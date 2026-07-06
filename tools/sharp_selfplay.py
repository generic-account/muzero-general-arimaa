"""Generate teacher games by playing bot_sharp against itself, into the same
supervised-pretraining shard format as tools/archive_dataset.py.

bot_sharp (David Wu, ~2015 Arimaa Challenge era) is the strongest classical
engine we have built + AEI-verified. Distilling its play gives the network a
strong, cheap ($0 of TPU — this is local CPU) initialization that skips the
value cold-start problem (see memory/cold-start-value-collapse); self-play then
improves past it via search (AlphaGo's bootstrap-then-surpass path).

Sharp emits moves in standard Arimaa notation — identical to the game archive —
so we reuse archive_dataset.convert_game verbatim: play a game with pyrimaa's
Game driver, reformat its move list into the archive movelist string, convert,
and validate every action against the JAX env legality (done by convert_game).

Usage:
    python tools/sharp_selfplay.py --games 200 --tc "4s/10s" --out results/sharp_ds
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from pyrimaa import aei  # noqa: E402
from pyrimaa.game import Game  # noqa: E402
from pyrimaa.util import TimeControl  # noqa: E402

from tools.archive_dataset import convert_game  # noqa: E402

# NB: the "aei" argument is REQUIRED — it puts sharp into AEI protocol mode
# (without it the engine never emits "aeiok" and the handshake times out).
SHARP = os.path.join(os.path.dirname(__file__), "..",
                     "third_party/bot_sharp/arimaasharp/build/sharp") + " aei"


def _movelist_from_game(moves):
    """pyrimaa Game.moves (['1g Ee2 ...', '1s ee7 ...', '2g ...', ...]) ->
    the archive movelist string: turns joined by the literal 2-char '\\n',
    color tags remapped g->w / s->b (convert_game only reads the setup tags
    and the leading turn digit; play-turn tags are otherwise ignored)."""
    fixed = []
    for m in moves:
        tag, _, rest = m.partition(" ")
        tag = tag.replace("g", "w").replace("s", "b")
        fixed.append(f"{tag} {rest}" if rest else tag)
    return "\\n".join(fixed)


def _make_engine():
    eng = aei.get_engine("stdio", SHARP, "sharp_selfplay.aei")
    return aei.EngineController(eng)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=200)
    ap.add_argument("--tc", default="4s/10s",
                    help="AEI time control per move/reserve (stronger = slower)")
    ap.add_argument("--out", default="results/sharp_ds")
    ap.add_argument("--shard-every", type=int, default=100,
                    help="flush a shard every N successfully converted games")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    tc = TimeControl(args.tc)

    boards, players, lefts, tstarts, acts, vals, mls = ([] for _ in range(7))
    n_ok = n_fail = shard = 0

    def flush():
        nonlocal shard, boards, players, lefts, tstarts, acts, vals, mls
        if not boards:
            return
        np.savez_compressed(
            os.path.join(args.out, f"sharp{shard:04d}.npz"),
            board=np.asarray(boards, np.int8), player=np.asarray(players, np.int8),
            steps_left=np.asarray(lefts, np.int8),
            turn_start=np.asarray(tstarts, np.int8),
            action=np.asarray(acts, np.int16), value=np.asarray(vals, np.float32),
            moves_left=np.asarray(mls, np.int32))
        shard += 1
        boards, players, lefts, tstarts, acts, vals, mls = ([] for _ in range(7))

    for gi in range(args.games):
        gold = silver = None
        try:
            gold, silver = _make_engine(), _make_engine()
            game = Game(gold, silver, timecontrol=tc, strict_setup=True)
            winner, term = game.play()
        except Exception as e:  # engine crash / illegal / timeout — skip the game
            print(f"game {gi}: engine error ({e}); skipping", flush=True)
            n_fail += 1
            continue
        finally:
            for eng in (gold, silver):
                if eng is not None:
                    try:
                        eng.quit(); eng.cleanup()
                    except Exception:
                        pass

        result = "w" if winner == 0 else "b"
        out = convert_game(_movelist_from_game(game.moves), result)
        if not out or not out[0]:
            n_fail += 1
            continue
        samples, _ = out
        n_ok += 1
        T = len(samples)
        for i, (b64, pl, left, ts, act) in enumerate(samples):
            boards.append(b64); players.append(pl); lefts.append(left)
            tstarts.append(ts); acts.append(act)
            vals.append(1.0 if pl == winner else -1.0)
            mls.append(T - i)
        if n_ok % args.shard_every == 0:
            flush()
        if (gi + 1) % 10 == 0:
            print(f"[{gi+1}/{args.games}] ok={n_ok} fail={n_fail} "
                  f"samples={len(boards) + shard*0}", flush=True)

    flush()
    print(f"DONE: games ok={n_ok} failed={n_fail} shards={shard}")


if __name__ == "__main__":
    main()
