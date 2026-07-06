"""Convert arimaa.com game-archive TSVs into supervised pretraining shards.

The public archive (arimaa.com/arimaa/download/gameData/, "for research and bot
development") is one TSV per year: one game per line with ratings, result,
termination and a movelist in standard notation (turns separated by literal
"\\n"; push/pull written as TWO tokens in execution order; trap captures as
"Rc3x" tokens; setup turns "1w"/"1b" list placements).

We replay each qualifying game through the LEGACY oracle engine
(games/arimaa.py), converting notation tokens into our 1393-action indices
(push/pull token pairs -> one combined action, with a tiny backtracking parser
for the plain-step-vs-pull ambiguity), validating every action against the
engine's legal set. Output shards store RAW state fields (board, player,
steps_left, turn_start_board) + labels (action, outcome value, moves-left), so
observation planes can be built later with any FeaturesConfig.

Usage:
    python tools/archive_dataset.py --years 2002 2003 --min-rating 1600 \
        --out results/archive_ds
"""

import argparse
import re
import os
import sys
import urllib.request

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from games import arimaa as a  # noqa: E402
from jaxarimaa.difftest import array_from_legacy_board  # noqa: E402

a.init_actions()

_UA = {"User-Agent": "Mozilla/5.0 (research; arimaa bot development)"}
_BASE = "http://arimaa.com/arimaa/download/gameData/"
# terminations with trustworthy move quality + known winner
_GOOD_TERM = {"g", "e", "m", "r"}  # goal, elimination, immobilization, resign


def _sq(tok):
    """Algebraic square 'e2' -> our (x, y) with y=0 at rank 8."""
    return ord(tok[0]) - 97, 8 - int(tok[1])


_DXY = {"n": (0, -1), "s": (0, 1), "e": (1, 0), "w": (-1, 0)}
_STEP_RE = re.compile(r"[RCDHMErcdhme][a-h][1-8][nsew]")


def _fetch_tgz(name, cache_dir):
    """Download + extract one archive; return the extracted txt path or None."""
    import tarfile
    txt = os.path.join(cache_dir, f"{name}.txt")
    if os.path.exists(txt):
        return txt
    tgz = os.path.join(cache_dir, f"{name}.tgz")
    try:
        req = urllib.request.Request(_BASE + f"{name}.tgz", headers=_UA)
        with urllib.request.urlopen(req) as r, open(tgz, "wb") as f:
            f.write(r.read())
    except urllib.error.HTTPError:
        return None
    with tarfile.open(tgz) as t:
        t.extractall(cache_dir)
    return txt if os.path.exists(txt) else None


def fetch_year(year, cache_dir):
    """Yearly file (early years) or 12 monthly files (high-volume years),
    concatenated into one TSV (header kept from the first part)."""
    os.makedirs(cache_dir, exist_ok=True)
    combined = os.path.join(cache_dir, f"allgames{year}.txt")
    if os.path.exists(combined):
        return combined
    if _fetch_tgz(f"allgames{year}", cache_dir):
        return combined
    parts = [_fetch_tgz(f"allgames{year}{m:02d}", cache_dir) for m in range(1, 13)]
    parts = [p for p in parts if p]
    if not parts:
        raise FileNotFoundError(f"no archive files for {year}")
    with open(combined, "w") as out:
        for i, p in enumerate(parts):
            with open(p, encoding="utf-8", errors="replace") as f:
                header = f.readline()
                if i == 0:
                    out.write(header)
                out.write(f.read())
    return combined


def _setup_board(setup_w, setup_b):
    """Build a legacy Board from the two setup turns' placement tokens."""
    board = a.Board()
    board.state = a.State()
    board.state.setup = False
    board.state.end = False
    board.state.player = 0
    board.state.left = 4
    for tokens, color in ((setup_w, 0), (setup_b, 1)):
        for tok in tokens:
            if len(tok) != 3:
                continue
            piece = a.char_to_piece(tok[0])
            board[_sq(tok[1:])] = piece
    return board


def _turn_tokens(turn_line):
    """Split a turn line into move tokens, dropping the turn tag + captures."""
    toks = turn_line.split()
    return [t for t in toks[1:] if not t.endswith("x")]


def _turn_moves(turn_line):
    """Turn line -> list of (move_token, captures_after) where captures_after
    is the list of capture tokens ("Rc3x") recorded after that move token.
    Captures are ground truth from the server and disambiguate pull-vs-plain
    parses whose intermediate boards fire traps differently."""
    toks = turn_line.split()[1:]
    moves = []
    for tok in toks:
        if tok.endswith("x"):
            if moves:
                moves[-1][1].append(tok)
        else:
            moves.append((tok, []))
    return moves


def _match_actions(board, moves):
    """Backtracking parse: (move_token, captures) list -> combined step actions.

    A token moving an ENEMY piece must start a push (pair with next token);
    a token moving OUR piece is either a plain step or the start of a pull
    (enemy token follows into the vacated square). Interpretations can differ
    in which traps fire mid-turn, so each candidate must reproduce EXACTLY the
    recorded captures (the server's ground truth) or it is rejected.
    Returns a list of action indices, or None if no parse validates.
    """
    if not moves:
        return []
    legal = {s for s in board.possible_steps()
             if board.step_cost(s) <= board.state.left}

    def tok_move(tok):
        frm = _sq(tok[1:3])
        d = _DXY[tok[3]]
        return frm, (frm[0] + d[0], frm[1] + d[1])

    frm0, to0 = tok_move(moves[0][0])
    piece0 = board[frm0]
    if piece0 is None:
        return None
    is_ours = (a.parse_piece(piece0)[0] == board.state.player)

    candidates = []
    if is_ours:
        # pull first (consumes 2 tokens): enemy follows into our vacated square
        if len(moves) >= 2:
            frm1, to1 = tok_move(moves[1][0])
            if to1 == frm0:
                spec = a.StepSpec(frm0, to0, frm1, to1)
                candidates.append((spec, 2))
        candidates.append((a.StepSpec(frm0, to0, None, None), 1))
    else:
        # enemy moves first: must be a push completed by our next token
        if len(moves) >= 2:
            frm1, to1 = tok_move(moves[1][0])
            if to1 == frm0:
                spec = a.StepSpec(frm1, to1, frm0, to0)
                candidates.append((spec, 2))

    for spec, n_tok in candidates:
        step = a.Step()
        step.oldPos, step.newPos = spec.old_pos, spec.new_pos
        step.opOldPos, step.opNewPos = spec.op_old_pos, spec.op_new_pos
        if not any(s.oldPos == step.oldPos and s.newPos == step.newPos
                   and s.opOldPos == step.opOldPos and s.opNewPos == step.opNewPos
                   for s in legal):
            continue
        # expected captures: all capture tokens attached to the consumed moves
        expected = set()
        for _, caps in moves[:n_tok]:
            for c in caps:
                expected.add((c[0], _sq(c[1:3])))
        saved = board.encode()
        saved_left = board.state.left
        before = {pos: board[pos] for pos in a.all_positions()
                  if board[pos] is not None}
        board.do_step(step)
        # Captures = (a) non-mover pieces that disappeared in place, plus
        # (b) movers missing from their destination (died on the trap there).
        movers = [(spec.old_pos, spec.new_pos)]
        if spec.op_old_pos is not None:
            movers.append((spec.op_old_pos, spec.op_new_pos))
        mover_sqs = {p for m in movers for p in m}
        vanished = set()
        for pos, piece in before.items():
            if pos in mover_sqs:
                continue
            if board[pos] is None:
                vanished.add((a.piece_to_char(piece), pos))
        for frm, dst in movers:
            if board[dst] != before[frm]:
                vanished.add((a.piece_to_char(before[frm]), dst))
        if vanished == expected:
            board.state.left = saved_left - (2 if spec.op_old_pos else 1)
            rest = _match_actions(board, moves[n_tok:])
            if rest is not None:
                return [a.ACTION_INDEX[spec]] + rest
        board.decode(saved)
        board.state.left = saved_left
    return None


def convert_game(movelist, result):
    """Replay one game; yield raw samples. Returns None on any parse failure."""
    turns = movelist.split("\\n")
    if len(turns) < 4:
        return None
    setup_w = _turn_tokens(turns[0]) if turns[0].startswith("1w") else None
    setup_b = _turn_tokens(turns[1]) if turns[1].startswith("1b") else None
    if not setup_w or not setup_b:
        return None
    board = _setup_board(setup_w, setup_b)
    winner = 0 if result == "w" else 1

    samples = []  # (board64, player, steps_left, turn_start64, action)
    for line in turns[2:]:
        line = line.strip()
        if not line or line[0] not in "0123456789":
            continue
        tokens = _turn_tokens(line)
        if "takeback" in tokens:
            return None
        if any(t in ("resigns", "resign") for t in tokens):
            break
        if not tokens:
            continue
        if not all(_STEP_RE.fullmatch(t) for t in tokens):
            return None  # malformed / rewound-setup lines (rare, drop the game)
        moves = _turn_moves(line)
        saved = board.encode()
        turn_start_arr = array_from_legacy_board(board)
        actions = _match_actions(board, moves)
        if actions is None:
            return None
        # _match_actions advanced the board; rewind and replay per-action so we
        # can record the state each action was taken FROM.
        board.decode(saved)
        board.state.left = 4
        left = 4
        for act in actions:
            samples.append((array_from_legacy_board(board), board.state.player,
                            left, turn_start_arr, act))
            spec = a.ACTION_LIST[act]
            step = a.Step()
            step.oldPos, step.newPos = spec.old_pos, spec.new_pos
            step.opOldPos, step.opNewPos = spec.op_old_pos, spec.op_new_pos
            board.do_step(step)
            left -= 2 if spec.op_old_pos else 1
        if left > 0:
            # explicit END_TURN action (turns of <4 step-points)
            samples.append((array_from_legacy_board(board), board.state.player,
                            left, turn_start_arr, a.ACTION_END_TURN))
        board.finish_turn(check_win=False)  # flips player (4-step turns too)
        board.state.left = 4
    return samples, winner


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", nargs="+", type=int, required=True)
    ap.add_argument("--min-rating", type=int, default=1600)
    ap.add_argument("--out", default="results/archive_ds")
    ap.add_argument("--cache", default="results/archive_raw")
    ap.add_argument("--limit", type=int, default=0, help="max games (debug)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    for year in args.years:
        path = fetch_year(year, args.cache)
        n_ok = n_skip = n_fail = 0
        boards, players, lefts, tstarts, acts, vals, mls = [], [], [], [], [], [], []
        with open(path, encoding="utf-8", errors="replace") as f:
            header = f.readline().rstrip("\n").split("\t")
            idx = {k: i for i, k in enumerate(header)}
            for line in f:
                row = line.rstrip("\n").split("\t")
                if len(row) < len(header):
                    continue
                g = lambda k: row[idx[k]]
                if g("result") not in ("w", "b") or g("termination") not in _GOOD_TERM:
                    n_skip += 1
                    continue
                try:
                    if (int(g("wrating")) < args.min_rating
                            or int(g("brating")) < args.min_rating
                            or int(g("plycount")) < 6):
                        n_skip += 1
                        continue
                except ValueError:
                    n_skip += 1
                    continue
                out = convert_game(g("movelist"), g("result"))
                if not out or not out[0]:
                    n_fail += 1
                    continue
                samples, winner = out
                n_ok += 1
                T = len(samples)
                for i, (b64, pl, left, ts, act) in enumerate(samples):
                    boards.append(b64)
                    players.append(pl)
                    lefts.append(left)
                    tstarts.append(ts)
                    acts.append(act)
                    vals.append(1.0 if pl == winner else -1.0)
                    mls.append(T - i)
                if args.limit and n_ok >= args.limit:
                    break
        if boards:
            np.savez_compressed(
                os.path.join(args.out, f"year{year}.npz"),
                board=np.asarray(boards, np.int8),
                player=np.asarray(players, np.int8),
                steps_left=np.asarray(lefts, np.int8),
                turn_start=np.asarray(tstarts, np.int8),
                action=np.asarray(acts, np.int16),
                value=np.asarray(vals, np.float32),
                moves_left=np.asarray(mls, np.int32),
            )
        print(f"{year}: games ok={n_ok} skipped={n_skip} parse-failed={n_fail} "
              f"samples={len(boards)}")


if __name__ == "__main__":
    main()
