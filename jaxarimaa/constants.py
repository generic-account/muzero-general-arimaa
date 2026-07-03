"""Static board/action constants for the vectorized JAX Arimaa engine.

The action ordering is derived DIRECTLY from the legacy engine's canonical action
list (``games.arimaa.ACTION_LIST`` + the trailing END_TURN action). This guarantees
that a JAX action index and a legacy action index refer to the *same* move, which is
what makes differential testing against the oracle sound. (Later we can freeze these
tables to a .npz and drop the import-time dependency on the legacy engine.)

Board / cell encoding used throughout jaxarimaa (distinct from the legacy engine's
internal piece codes):
    cell 0            = empty
    cell 1 + rank + 6*color, rank in 0..5, color 0=GOLD 1=SILVER
    -> GOLD  rabbit..elephant = 1..6 ; SILVER rabbit..elephant = 7..12
Board arrays are stored row-major as ``board[y, x]`` with ``y=0`` the top rank (rank 8)
and ``x=0`` file 'a', matching the legacy engine's ``Board[(x, y)] == _data[y][x]``.
"""

import numpy as np

from games import arimaa as _legacy

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
# Action tables (derived from the legacy canonical ordering)
# ---------------------------------------------------------------------------
_legacy.init_actions()
_ACTION_LIST = _legacy.ACTION_LIST
N_STEP_ACTIONS = len(_ACTION_LIST)          # 1392
END_TURN = N_STEP_ACTIONS                    # index 1392
N_ACTIONS = N_STEP_ACTIONS + 1               # 1393

_NEG = -1  # sentinel for "no coordinate / no push-pull / end-turn"


def _build_action_tables():
    """Return numpy int16 tables of shape [N_ACTIONS] describing every action.

    For a plain step: (from)->(to) with op_* = -1.
    For a push/pull:  our piece (from)->(to) AND opponent (op_from)->(op_to).
    For END_TURN: all fields = -1, is_end = True.
    """
    frm_x = np.full(N_ACTIONS, _NEG, np.int16)
    frm_y = np.full(N_ACTIONS, _NEG, np.int16)
    to_x = np.full(N_ACTIONS, _NEG, np.int16)
    to_y = np.full(N_ACTIONS, _NEG, np.int16)
    op_frm_x = np.full(N_ACTIONS, _NEG, np.int16)
    op_frm_y = np.full(N_ACTIONS, _NEG, np.int16)
    op_to_x = np.full(N_ACTIONS, _NEG, np.int16)
    op_to_y = np.full(N_ACTIONS, _NEG, np.int16)
    is_pushpull = np.zeros(N_ACTIONS, np.bool_)
    is_end = np.zeros(N_ACTIONS, np.bool_)
    cost = np.ones(N_ACTIONS, np.int16)

    for i, spec in enumerate(_ACTION_LIST):
        (fx, fy) = spec.old_pos
        (tx, ty) = spec.new_pos
        frm_x[i], frm_y[i], to_x[i], to_y[i] = fx, fy, tx, ty
        if spec.op_old_pos is not None:
            (ofx, ofy) = spec.op_old_pos
            (otx, oty) = spec.op_new_pos
            op_frm_x[i], op_frm_y[i] = ofx, ofy
            op_to_x[i], op_to_y[i] = otx, oty
            is_pushpull[i] = True
            cost[i] = 2
    is_end[END_TURN] = True
    cost[END_TURN] = 0

    return {
        "frm_x": frm_x, "frm_y": frm_y, "to_x": to_x, "to_y": to_y,
        "op_frm_x": op_frm_x, "op_frm_y": op_frm_y,
        "op_to_x": op_to_x, "op_to_y": op_to_y,
        "is_pushpull": is_pushpull, "is_end": is_end, "cost": cost,
    }


ACTION_TABLES = _build_action_tables()


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
