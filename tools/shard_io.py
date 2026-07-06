"""Shared schema + read/write for supervised-pretraining shards.

One place defines the shard columns so the two producers (tools/archive_dataset.py,
tools/sharp_selfplay.py) and the consumer (tools/pretrain.py) can't drift — adding
a column (e.g. sharp_value) is a one-line change here, not three.

Each shard stores RAW state fields (board/player/steps_left/turn_start) + labels
(action, game-outcome value, moves-left), so observation planes are built later at
any FeaturesConfig.
"""

import numpy as np

# column name -> numpy dtype
SHARD_DTYPES = {
    "board": np.int8, "player": np.int8, "steps_left": np.int8,
    "turn_start": np.int8, "action": np.int16, "value": np.float32,
    "moves_left": np.int32,
}
SHARD_KEYS = list(SHARD_DTYPES)


def new_columns():
    """Fresh empty accumulator dict (one list per shard column)."""
    return {k: [] for k in SHARD_KEYS}


def add_game(cols, samples, winner):
    """Append one game's samples to the accumulator.

    `samples` is a list of (board64, player, steps_left, turn_start64, action) in
    play order; value = game outcome (+1 if that mover won, else -1); moves_left =
    plies from that position to game end."""
    T = len(samples)
    for i, (b64, pl, left, ts, act) in enumerate(samples):
        cols["board"].append(b64)
        cols["player"].append(pl)
        cols["steps_left"].append(left)
        cols["turn_start"].append(ts)
        cols["action"].append(act)
        cols["value"].append(1.0 if pl == winner else -1.0)
        cols["moves_left"].append(T - i)


def write_shard(path, cols):
    """Write an accumulator to a compressed .npz with the canonical dtypes."""
    np.savez_compressed(
        path, **{k: np.asarray(cols[k], SHARD_DTYPES[k]) for k in SHARD_KEYS})


def load_shards(files):
    """Concatenate shard files into one dict of arrays. Adds `sharp_value`
    (falls back to `value` when absent, so the pretrain blend is a no-op) and
    `_has_sharp`."""
    cols = {k: [] for k in SHARD_KEYS}
    sharp, has_sharp = [], True
    for f in files:
        d = np.load(f)
        for k in SHARD_KEYS:
            cols[k].append(d[k])
        if "sharp_value" in d:
            sharp.append(d["sharp_value"])
        else:
            has_sharp = False
    out = {k: np.concatenate(v) for k, v in cols.items()}
    out["sharp_value"] = np.concatenate(sharp) if has_sharp else out["value"].copy()
    out["_has_sharp"] = has_sharp
    return out
