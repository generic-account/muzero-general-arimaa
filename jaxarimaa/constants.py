"""Static board/action constants for the vectorized JAX Arimaa engine.

The action ordering matches the legacy engine's canonical action list
(``games.arimaa.ACTION_LIST`` + the trailing END_TURN action), so a JAX action
index and a legacy action index refer to the *same* move — the invariant that
makes differential testing against the oracle sound. The tables ship FROZEN in
``action_tables.npz`` (so deploy targets, e.g. TPU VMs, need neither the legacy
engine nor torch); when the file is absent they are re-derived from the legacy
engine. Regenerate after any action-space change with:
    python -m jaxarimaa.constants   # writes jaxarimaa/action_tables.npz

Board / cell encoding used throughout jaxarimaa (distinct from the legacy engine's
internal piece codes):
    cell 0            = empty
    cell 1 + rank + 6*color, rank in 0..5, color 0=GOLD 1=SILVER
    -> GOLD  rabbit..elephant = 1..6 ; SILVER rabbit..elephant = 7..12
Board arrays are stored row-major as ``board[y, x]`` with ``y=0`` the top rank (rank 8)
and ``x=0`` file 'a', matching the legacy engine's ``Board[(x, y)] == _data[y][x]``.
"""

import pathlib

import numpy as np

# ---------------------------------------------------------------------------
# Board geometry / piece encoding
# ---------------------------------------------------------------------------
BOARD = 8
N_CELLS = BOARD * BOARD
N_RANKS = 6  # rabbit, cat, dog, horse, camel, elephant
GOLD, SILVER = 0, 1
EMPTY = 0
# Trap squares as (x, y): c3, f3, c6, f6 -> matches legacy Board.TRAPS
TRAPS_XY = ((2, 2), (2, 5), (5, 2), (5, 5))

# Observation: 6 ranks * 2 colors + side-to-move plane (mirrors legacy get_observation)
OBS_PLANES = 13
OBS_SHAPE = (OBS_PLANES, BOARD, BOARD)


def cell_code(color: int, rank: int) -> int:
    return 1 + rank + N_RANKS * color


def decode_cell_code(code):
    """Inverse of cell_code: a non-empty cell code -> (color, rank).

    Works elementwise for Python ints, numpy, and jax arrays (empty cells, code 0,
    yield a nonsense pair that callers mask out with the occupancy test).
    """
    return (code - 1) // N_RANKS, (code - 1) % N_RANKS


# ---------------------------------------------------------------------------
# Action tables: frozen .npz (deploy) or derived from the legacy engine (dev)
# ---------------------------------------------------------------------------
_NEG = -1  # sentinel for "no coordinate / no push-pull / end-turn"
_NPZ_PATH = pathlib.Path(__file__).with_name("action_tables.npz")


def _derive_action_tables():
    """Build the tables from the legacy engine's canonical ACTION_LIST.

    For a plain step: (from)->(to) with op_* = -1.
    For a push/pull:  our piece (from)->(to) AND opponent (op_from)->(op_to).
    For END_TURN (last row): all fields = -1, is_end = True.
    """
    from games import arimaa as _legacy  # requires the dev environment (torch)

    _legacy.init_actions()
    action_list = _legacy.ACTION_LIST
    n = len(action_list) + 1  # + END_TURN
    t = {k: np.full(n, _NEG, np.int16)
         for k in ("frm_x", "frm_y", "to_x", "to_y",
                   "op_frm_x", "op_frm_y", "op_to_x", "op_to_y")}
    t["is_pushpull"] = np.zeros(n, np.bool_)
    t["is_end"] = np.zeros(n, np.bool_)
    t["cost"] = np.ones(n, np.int16)

    for i, spec in enumerate(action_list):
        t["frm_x"][i], t["frm_y"][i] = spec.old_pos
        t["to_x"][i], t["to_y"][i] = spec.new_pos
        if spec.op_old_pos is not None:
            t["op_frm_x"][i], t["op_frm_y"][i] = spec.op_old_pos
            t["op_to_x"][i], t["op_to_y"][i] = spec.op_new_pos
            t["is_pushpull"][i] = True
            t["cost"][i] = 2
    t["is_end"][n - 1] = True
    t["cost"][n - 1] = 0
    return t


if _NPZ_PATH.exists():
    with np.load(_NPZ_PATH) as _d:
        ACTION_TABLES = {k: _d[k] for k in _d.files}
else:
    ACTION_TABLES = _derive_action_tables()

N_ACTIONS = len(ACTION_TABLES["frm_x"])      # 1393
END_TURN = N_ACTIONS - 1                      # index 1392
N_STEP_ACTIONS = END_TURN                     # 1392


# Static trap-square plane [8,8] (1.0 at c3/f3/c6/f6), for the optional trap input plane.
TRAP_PLANE = np.zeros((BOARD, BOARD), np.float32)
for _tx, _ty in TRAPS_XY:
    TRAP_PLANE[_ty, _tx] = 1.0


def _build_sym_perm():
    """Action-index permutation induced by the left-right board mirror (x -> 7-x).

    Arimaa is symmetric under this reflection (traps c/f swap: 7-2=5), so mirroring
    (board, policy target) is a valid 2x data augmentation. y is unchanged; END_TURN
    maps to itself.
    """
    t = ACTION_TABLES
    fields = ("frm_x", "frm_y", "to_x", "to_y",
              "op_frm_x", "op_frm_y", "op_to_x", "op_to_y")

    def key(i):
        return tuple(int(t[f][i]) for f in fields)

    index = {key(i): i for i in range(N_STEP_ACTIONS)}
    perm = np.arange(N_ACTIONS, dtype=np.int32)

    def mx(v):
        return v if v < 0 else 7 - v  # mirror x, leave -1 sentinel

    for i in range(N_STEP_ACTIONS):
        mk = (mx(t["frm_x"][i]), t["frm_y"][i], mx(t["to_x"][i]), t["to_y"][i],
              mx(t["op_frm_x"][i]), t["op_frm_y"][i], mx(t["op_to_x"][i]), t["op_to_y"][i])
        perm[i] = index[tuple(int(x) for x in mk)]
    return perm


SYM_PERM = _build_sym_perm()  # numpy int32 [N_ACTIONS]


if __name__ == "__main__":
    # Freeze the legacy-derived tables to action_tables.npz (run in the dev env).
    tables = _derive_action_tables()
    np.savez_compressed(_NPZ_PATH, **tables)
    with np.load(_NPZ_PATH) as check:
        assert all(np.array_equal(check[k], tables[k]) for k in tables)
    print(f"wrote {_NPZ_PATH} ({_NPZ_PATH.stat().st_size} bytes, "
          f"{len(tables['frm_x'])} actions)")


# ---------------------------------------------------------------------------
# Zobrist hashing for the repetition rule (uint32; row 0 = empty contributes 0).
# ---------------------------------------------------------------------------
_ZRNG = np.random.RandomState(20260705)
ZOBRIST_CELLS = _ZRNG.randint(1, 2**32, size=(13, N_CELLS), dtype=np.uint32)
ZOBRIST_CELLS[0, :] = 0  # empty cells contribute nothing
ZOBRIST_PLAYER = _ZRNG.randint(1, 2**32, size=(2,), dtype=np.uint32)
REP_HISTORY = 64  # turn-end positions remembered per game (ring buffer)
MOVES_LEFT_CAP = 64.0  # plies-to-end cap for the moves-left head target normalization
