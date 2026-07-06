"""Annotate pretraining shards with bot_sharp's static-eval value (NNUE-style).

Adds a `sharp_value` column: sharp's static positional eval mapped to
[-1, 1] from the SIDE-TO-MOVE's perspective (2*winprob-1) — the same perspective
as the game-outcome `value` column. pretrain.py blends them:
`value_target = (1-w)*outcome + w*sharp_value`.

Pipeline: our int8 board -> sharp position notation -> `sharp evalDump` (built
static-eval subcommand, one line "<eval> <winprob>" per position, side-to-move
perspective, winProbScale=4500) -> parse -> augmented .npz.

Usage:
    python tools/annotate_sharp.py --shards 'results/archive_ds/*.npz' \
        --out results/archive_ds_sharp
"""

import argparse
import glob
import os
import subprocess
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from jaxarimaa import constants as C  # noqa: E402
from tools import shard_io  # noqa: E402

SHARP = os.path.join(os.path.dirname(__file__), "..",
                     "third_party/bot_sharp/arimaasharp/build/sharp")
_RANK = ["R", "C", "D", "H", "M", "E"]  # rank 0..5; lowercase for silver


def board_to_sharp(board, player):
    """int8 [8,8] cell-code board (y=0 rank8) + player(0=gold) -> a sharp position
    record: a `1g`/`1s` side-to-move line then 8 board rows (rank 8 first)."""
    lines = ["1g" if player == 0 else "1s"]
    for y in range(8):
        row = []
        for x in range(8):
            code = int(board[y, x])
            if code == 0:
                row.append(".")
            else:
                color, rank = (code - 1) // C.N_RANKS, (code - 1) % C.N_RANKS
                ch = _RANK[rank]
                row.append(ch if color == 0 else ch.lower())
        lines.append("".join(row))
    return "\n".join(lines)


def eval_positions(boards, players, winprobscale):
    """Run sharp evalDump over all positions; return win-prob array [N] in
    side-to-move perspective (input order preserved)."""
    recs = [board_to_sharp(boards[i], int(players[i])) for i in range(len(boards))]
    with tempfile.NamedTemporaryFile("w", suffix=".pos", delete=False) as f:
        f.write("\n;\n".join(recs))
        posfile = f.name
    try:
        out = subprocess.run(
            [SHARP, "evalDump", posfile, "-winprobscale", str(winprobscale)],
            capture_output=True, text=True, check=True).stdout
    finally:
        os.unlink(posfile)
    winprobs = []
    for line in out.splitlines():
        line = line.strip()
        if not line:
            continue
        winprobs.append(float(line.split()[1]))  # "<eval_int> <winprob>"
    wp = np.asarray(winprobs, np.float32)
    if len(wp) != len(boards):
        raise RuntimeError(f"sharp returned {len(wp)} evals for {len(boards)} positions")
    return wp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--winprobscale", type=float, default=4500.0)
    ap.add_argument("--chunk", type=int, default=20000,
                    help="positions per evalDump invocation (bounds the temp file)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    files = [f for p in args.shards for f in sorted(glob.glob(p))]
    if not files:
        raise SystemExit(f"no shards matched {args.shards}")

    for f in files:
        d = dict(np.load(f))
        boards, players = d["board"], d["player"]
        N = len(boards)
        sv = np.empty(N, np.float32)
        for s in range(0, N, args.chunk):
            e = min(s + args.chunk, N)
            wp = eval_positions(boards[s:e], players[s:e], args.winprobscale)
            sv[s:e] = 2.0 * wp - 1.0  # win-prob -> value in [-1,1], side-to-move perspective
        d["sharp_value"] = sv
        outpath = os.path.join(args.out, os.path.basename(f))
        np.savez_compressed(outpath, **d)
        print(f"{os.path.basename(f)}: {N} positions | sharp_value "
              f"mean={sv.mean():+.3f} std={sv.std():.3f} "
              f"corr_with_outcome={np.corrcoef(sv, d['value'])[0,1]:+.3f} -> {outpath}",
              flush=True)


if __name__ == "__main__":
    main()
