"""Vectorized JAX Arimaa engine.

Implemented so far:
  * observe(state) -> (13,8,8) observation, byte-identical to the legacy
    ``ArimaaEnv.get_observation`` (validated by differential tests).
  * bridge helpers to build a State from an explicit board (for tests / setup).

STUBS (the hard branchless vectorization work — implemented next, validated against
the oracle at every step):
  * legal_action_mask(state) -> bool[N_ACTIONS]
  * step(state, action)      -> State

Everything here must be jittable and vmappable: fixed shapes, no data-dependent
Python control flow, use jnp.where / lax.cond / lax.select.
"""

import jax
import jax.numpy as jnp
import numpy as np

from . import constants as C
from .types import State

# ---------------------------------------------------------------------------
# Static action tables as jnp arrays + derived push/pull classification.
# Each action is fully specified; legality is a pure predicate over these.
# ---------------------------------------------------------------------------
_T = C.ACTION_TABLES
_FX = jnp.asarray(_T["frm_x"], jnp.int32)
_FY = jnp.asarray(_T["frm_y"], jnp.int32)
_TX = jnp.asarray(_T["to_x"], jnp.int32)
_TY = jnp.asarray(_T["to_y"], jnp.int32)
_OFX = jnp.asarray(_T["op_frm_x"], jnp.int32)
_OFY = jnp.asarray(_T["op_frm_y"], jnp.int32)
_OTX = jnp.asarray(_T["op_to_x"], jnp.int32)
_OTY = jnp.asarray(_T["op_to_y"], jnp.int32)
_COST = jnp.asarray(_T["cost"], jnp.int32)
_IS_END = jnp.asarray(_T["is_end"])

# Push:  StepSpec(old, enemy, enemy, op_new)  -> to == op_from
# Pull:  StepSpec(old, new,   enemy, old)     -> op_to == from
_np_pp = _T["is_pushpull"]
_np_push = _np_pp & (_T["to_x"] == _T["op_frm_x"]) & (_T["to_y"] == _T["op_frm_y"])
_np_pull = _np_pp & (_T["op_to_x"] == _T["frm_x"]) & (_T["op_to_y"] == _T["frm_y"])
_IS_PUSH = jnp.asarray(_np_push)
_IS_PULL = jnp.asarray(_np_pull)

_DIRS = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _neighbor(grid, dx, dy, fill):
    """Value of the cell at (x+dx, y+dy) for each (x,y); `fill` off-board.

    grid is [y, x]. dx, dy are static ints in {-1,0,1}.
    """
    padded = jnp.pad(grid, 1, constant_values=fill)
    return padded[1 + dy:9 + dy, 1 + dx:9 + dx]


def _gather(grid, vx, vy):
    """grid[y, x] gathered at coordinate arrays (vx, vy), clipped to the board.

    Sentinel -1 coords (plain-step op fields / END_TURN) clip to 0 and are ignored
    by the action-type masks that select them.
    """
    return grid[jnp.clip(vy, 0, 7), jnp.clip(vx, 0, 7)]


# ---------------------------------------------------------------------------
# Observation encoding  (mirrors legacy ArimaaEnv.get_observation)
# ---------------------------------------------------------------------------
_TRAP_PLANE = jnp.asarray(C.TRAP_PLANE)
_ZCELLS = jnp.asarray(C.ZOBRIST_CELLS)    # [13, 64] uint32
_ZPLAYER = jnp.asarray(C.ZOBRIST_PLAYER)  # [2] uint32


def position_hash(board, player):
    """Zobrist hash of (board, side-to-move). Empty cells contribute 0."""
    cell_keys = _ZCELLS[board.reshape(-1).astype(jnp.int32), jnp.arange(C.N_CELLS)]
    h = jax.lax.reduce(cell_keys, jnp.uint32(0), jax.lax.bitwise_xor, (0,))
    return h ^ _ZPLAYER[player.astype(jnp.int32)]


def observe(state: State, features=None) -> jnp.ndarray:
    """Return a (P, 8, 8) float32 observation for `state`.

    Base 13 planes (always): 6 GOLD-rank + 6 SILVER-rank one-hots + side-to-move.
    With `features` set, optional planes are appended (order fixed for reproducibility):
    frozen mask, trap squares, step-within-turn. `features=None` yields exactly the
    base 13 planes, byte-identical to the original encoding (the difftest invariant).
    Layout is [plane, y, x] with cell codes 1..6 GOLD, 7..12 SILVER.
    """
    codes = state.board.astype(jnp.int32)  # [8,8]; 0 empty, 1..12
    ids = jnp.arange(1, 2 * C.N_RANKS + 1)  # 1..12
    piece_planes = (codes[None] == ids[:, None, None]).astype(jnp.float32)  # (12,8,8)
    stm = jnp.where(state.player == C.GOLD, 1.0, 0.0).astype(jnp.float32)
    planes = [piece_planes, jnp.full((1, C.BOARD, C.BOARD), stm, jnp.float32)]

    if features is not None:
        if features.planes_frozen:
            occ, col, rnk = _piece_grids(state.board)
            planes.append(_frozen_grid(occ, col, rnk).astype(jnp.float32)[None])
        if features.planes_trap:
            planes.append(_TRAP_PLANE[None])
        if features.planes_step_in_turn:
            sip = (4.0 - state.steps_left.astype(jnp.float32)) / 4.0
            planes.append(jnp.full((1, C.BOARD, C.BOARD), sip, jnp.float32))
        if features.planes_moved:
            moved = (state.board != state.turn_start_board).astype(jnp.float32)
            planes.append(moved[None])

    return jnp.concatenate(planes, axis=0)  # (P,8,8)


def where_state(mask, if_true: State, if_false: State) -> State:
    """Per-leaf select between two batched State pytrees along the batch axis:
    return `if_true` where `mask` (a [B] boolean) is set, else `if_false`.

    Shared by search / self-play / evaluation for terminal-absorption and
    auto-reset, so the mask-broadcasting logic lives in exactly one place.
    """
    def sel(a, b):
        m = mask.reshape((-1,) + (1,) * (a.ndim - 1))
        return jnp.where(m, a, b)
    return jax.tree_util.tree_map(sel, if_true, if_false)


# ---------------------------------------------------------------------------
# Initialisation: random legal setup (both sides), jittable & vmappable.
# Setup is not (yet) part of the learned action space; we place the standard 16
# pieces per side randomly on their two home rows, matching the legacy engine's
# random-setup behaviour. Learned setup is a later extension (see scope doc).
# ---------------------------------------------------------------------------
# Standard per-side ranks: 8 rabbits, 2 cat, 2 dog, 2 horse, 1 camel, 1 elephant.
_SETUP_RANKS = jnp.asarray(
    [0] * 8 + [1, 1, 2, 2, 3, 3, 4, 5], dtype=jnp.int32
)  # length 16


def init_state(key) -> State:
    """Return a fresh State with a random legal setup; GOLD to move, 4 steps."""
    kg, ks = jax.random.split(key)
    board = jnp.zeros((C.BOARD, C.BOARD), jnp.int8)

    gold_ranks = jax.random.permutation(kg, _SETUP_RANKS)
    silver_ranks = jax.random.permutation(ks, _SETUP_RANKS)
    # GOLD home rows y=6,7 (bottom); SILVER home rows y=0,1 (top). Fill row-major.
    gold_codes = (1 + gold_ranks).astype(jnp.int8)          # 1..6
    silver_codes = (1 + C.N_RANKS + silver_ranks).astype(jnp.int8)  # 7..12
    board = board.at[6:8, :].set(gold_codes.reshape(2, C.BOARD))
    board = board.at[0:2, :].set(silver_codes.reshape(2, C.BOARD))

    rep_hist = jnp.zeros((C.REP_HISTORY,), jnp.uint32)
    rep_hist = rep_hist.at[0].set(position_hash(board, jnp.int8(C.GOLD)))
    return State(
        board=board,
        player=jnp.int8(C.GOLD),
        steps_left=jnp.int8(4),
        terminated=jnp.bool_(False),
        winner=jnp.int8(-1),
        turn_start_board=board,
        rep_hist=rep_hist,
        rep_ptr=jnp.int32(1),
    )


# ---------------------------------------------------------------------------
# Bridge helper: build a State from an explicit board array
# ---------------------------------------------------------------------------
def state_from_board(board, player, steps_left=4, terminated=False, winner=-1,
                     turn_start=None) -> State:
    board = jnp.asarray(board, jnp.int8)
    # turn_start defaults to the current board (fresh turn); pass the turn's starting
    # board explicitly for mid-turn states so the planes_moved feature is correct.
    ts = board if turn_start is None else jnp.asarray(turn_start, jnp.int8)
    return State(
        board=board,
        player=jnp.int8(player),
        steps_left=jnp.int8(steps_left),
        terminated=jnp.bool_(terminated),
        winner=jnp.int8(winner),
        turn_start_board=ts,
        rep_hist=jnp.zeros((C.REP_HISTORY,), jnp.uint32),
        rep_ptr=jnp.int32(0),
    )


# ---------------------------------------------------------------------------
# STUBS — the vectorization work goes here next.
# ---------------------------------------------------------------------------
def _piece_grids(board):
    """Return (occ, col, rnk) grids [8,8]. col/rnk are -1 on empty squares."""
    occ = board != 0
    raw_col, raw_rnk = C.decode_cell_code(board.astype(jnp.int32))
    col = jnp.where(occ, raw_col, -1)  # 0 gold, 1 silver
    rnk = jnp.where(occ, raw_rnk, -1)  # 0 rabbit .. 5 elephant
    return occ, col, rnk


def _frozen_grid(occ, col, rnk):
    """Per-cell frozen flag: occupied, has a stronger enemy neighbour, and no
    friendly neighbour (mirrors legacy Board.is_frozen)."""
    stronger = jnp.zeros((C.BOARD, C.BOARD), bool)
    friendly = jnp.zeros((C.BOARD, C.BOARD), bool)
    for dx, dy in _DIRS:
        n_occ = _neighbor(occ, dx, dy, False)
        n_col = _neighbor(col, dx, dy, -1)
        n_rnk = _neighbor(rnk, dx, dy, -1)
        stronger = stronger | (n_occ & (n_col != col) & (n_rnk > rnk))
        friendly = friendly | (n_occ & (n_col == col))
    return occ & stronger & (~friendly)


def legal_action_mask(state: State) -> jnp.ndarray:
    """bool[N_ACTIONS] legality mask, a pure function of (board, player, steps_left).

    Evaluates every action's legality in parallel against the static action table.
    Does NOT apply the stateful 3-fold-repetition restriction on END_TURN — that is
    the driver's / step-bookkeeping's job (see difftest for how this is isolated).
    """
    board = state.board
    player = state.player.astype(jnp.int32)
    left = state.steps_left.astype(jnp.int32)
    occ, col, rnk = _piece_grids(board)
    frozen = _frozen_grid(occ, col, rnk)

    # Mover (our unfrozen piece at `from`)
    f_occ = _gather(occ, _FX, _FY)
    f_col = _gather(col, _FX, _FY)
    f_rnk = _gather(rnk, _FX, _FY)
    f_frozen = _gather(frozen, _FX, _FY)
    ours = f_occ & (f_col == player) & (~f_frozen)

    # Destination emptiness
    to_empty = _gather(board, _TX, _TY) == 0
    op_to_empty = _gather(board, _OTX, _OTY) == 0

    # Rabbit cannot step backward (plain steps only; pushers/pullers are never rabbits)
    is_rabbit = f_rnk == 0
    backward = jnp.where(player == C.GOLD, _TY > _FY, _TY < _FY)
    rabbit_ok = ~(is_rabbit & backward)

    # Enemy for push/pull: adjacent, opposite colour, strictly weaker than mover
    e_occ = _gather(occ, _OFX, _OFY)
    e_col = _gather(col, _OFX, _OFY)
    e_rnk = _gather(rnk, _OFX, _OFY)
    enemy_ok = e_occ & (e_col != player) & (f_rnk > e_rnk)

    cost_ok = _COST <= left
    common = ours & cost_ok

    plain_legal = common & to_empty & rabbit_ok
    push_legal = common & enemy_ok & op_to_empty
    pull_legal = common & enemy_ok & to_empty
    step_legal = jnp.where(_IS_PUSH, push_legal,
                           jnp.where(_IS_PULL, pull_legal, plain_legal))

    # END_TURN: not at turn start, and ending now must not create a 3rd repetition
    # of (position, opponent-to-move) — the Arimaa repetition rule.
    end_hash = position_hash(board, (1 - player).astype(jnp.int8))
    would_repeat = jnp.sum(state.rep_hist == end_hash) >= 2
    end_det = (left != 4) & (~would_repeat)
    alive = ~state.terminated
    return jnp.where(_IS_END, end_det & alive, step_legal & alive)


def _apply_traps(board):
    """Remove any piece standing on a trap with no friendly orthogonal neighbour.

    Traps (c3,f3,c6,f6) are interior, and no two traps share a neighbour, so they are
    handled independently, all read from the same pre-removal board.
    """
    nb = board
    for (tx, ty) in C.TRAPS_XY:
        code = board[ty, tx]
        occ_t = code != 0
        ccol, _ = C.decode_cell_code(code)
        safe = jnp.bool_(False)
        for dx, dy in _DIRS:
            ncode = board[ty + dy, tx + dx]  # traps interior -> always in bounds
            ncol, _ = C.decode_cell_code(ncode)
            safe = safe | ((ncode != 0) & (ncol == ccol))
        remove = occ_t & (~safe)
        nb = jnp.where(remove, nb.at[ty, tx].set(jnp.int8(0)), nb)
    return nb


def _winner_after(board, mover, check_immobility=True):
    """Board-based win check after `mover` completed a turn (mirrors legacy
    check_win_reason ordering). Returns (has_winner, winner). Repetition (stateful)
    is handled elsewhere.

    `check_immobility=False` skips the (expensive) internal legal-mask evaluation;
    callers that already compute the next state's legal mask (the search's
    recurrent_fn) derive immobility from it instead — an empty mask means the
    side to move is immobilized (END_TURN is always legal mid-turn, so an empty
    mask can only occur at a fresh turn with no legal step).
    """
    A = mover.astype(jnp.int32)
    B = 1 - A
    A_rab = jnp.where(A == C.GOLD, 1, 7)   # gold rabbit=1, silver rabbit=7 (cell codes)
    B_rab = jnp.where(B == C.GOLD, 1, 7)
    A_goal = jnp.where(A == C.GOLD, 0, 7)  # gold goal row y=0, silver y=7
    B_goal = jnp.where(B == C.GOLD, 0, 7)

    goalA = jnp.any(board[A_goal] == A_rab)
    goalB = jnp.any(board[B_goal] == B_rab)
    norabB = ~jnp.any(board == B_rab)
    norabA = ~jnp.any(board == A_rab)

    if check_immobility:
        sub = State(board=board, player=B.astype(jnp.int8), steps_left=jnp.int8(4),
                    terminated=jnp.bool_(False), winner=jnp.int8(-1),
                    turn_start_board=board,
                    rep_hist=jnp.zeros((C.REP_HISTORY,), jnp.uint32),
                    rep_ptr=jnp.int32(0))
        immobB = ~jnp.any(legal_action_mask(sub)[:C.END_TURN])
    else:
        immobB = jnp.bool_(False)

    # apply in increasing priority so the highest-priority condition wins
    w = jnp.int32(-1)
    h = jnp.bool_(False)
    for cond, res in ((immobB, A), (norabA, B), (norabB, A), (goalB, B), (goalA, A)):
        w = jnp.where(cond, res, w)
        h = h | cond
    return h, w.astype(jnp.int8)


def step(state: State, action, defer_immobility=False) -> State:
    """Apply `action` (a scalar index) to `state`, returning the next State.

    Handles the board mutation (plain step / push / pull, in legacy do_step order),
    trap removal, step-cost accounting, turn transition, and board-based win
    detection. Assumes `action` is legal (see legal_action_mask). Pure & jittable.

    `defer_immobility=True` skips the internal immobilization check (which costs a
    full legal-mask evaluation): callers that compute the next state's legal mask
    anyway (the search) apply `terminated |= ~any(mask)`, `winner = 1 - player`
    themselves — halving the per-expansion mask work. Default is the exact,
    differential-tested behavior.
    """
    a = jnp.asarray(action, jnp.int32)
    fx, fy = _FX[a], _FY[a]
    tx, ty = _TX[a], _TY[a]
    ofx, ofy = jnp.clip(_OFX[a], 0, 7), jnp.clip(_OFY[a], 0, 7)
    otx, oty = jnp.clip(_OTX[a], 0, 7), jnp.clip(_OTY[a], 0, 7)
    is_end = _IS_END[a]
    is_pp = _IS_PUSH[a] | _IS_PULL[a]
    cost = _COST[a]

    board = state.board
    mover = board[fy, fx]
    enemy = board[ofy, ofx]

    # legacy do_step order: clear old, clear op_old, set new=mover, set op_new=enemy
    nb = board.at[fy, fx].set(jnp.int8(0))
    nb = jnp.where(is_pp, nb.at[ofy, ofx].set(jnp.int8(0)), nb)
    nb = nb.at[ty, tx].set(mover)
    nb = jnp.where(is_pp, nb.at[oty, otx].set(enemy), nb)
    moved = jnp.where(is_end, board, nb)
    moved = _apply_traps(moved)

    left_after = state.steps_left - cost.astype(jnp.int8)
    finish = is_end | (left_after == 0)
    has_win, winner = _winner_after(moved, state.player,
                                    check_immobility=not defer_immobility)

    # Repetition rule at turn end: a 3rd occurrence of (position, side-to-move)
    # loses for the REPEATER (rule-faithful anti-shuffling; NOTE the legacy env
    # awards this the other way — a known legacy quirk we deliberately diverge
    # from). Board-based wins (goal/immobilization/no-rabbits) take precedence.
    opp = (1 - state.player).astype(jnp.int8)
    end_hash = position_hash(moved, opp)
    rep3 = finish & (jnp.sum(state.rep_hist == end_hash) >= 2)
    winner = jnp.where(has_win, winner, opp)          # repeater loses if rep3
    terminal = finish & (has_win | rep3)

    # Record the turn-end position in the ring buffer (only on finish).
    slot = state.rep_ptr % C.REP_HISTORY
    new_hist = jnp.where(finish, state.rep_hist.at[slot].set(end_hash),
                         state.rep_hist)
    new_ptr = jnp.where(finish, state.rep_ptr + 1, state.rep_ptr)

    new_player = jnp.where(finish, opp, state.player)
    new_left = jnp.where(finish, jnp.int8(4), left_after)
    new_turn_start = jnp.where(finish, moved, state.turn_start_board)

    return State(
        board=moved,
        player=new_player,
        steps_left=new_left,
        terminated=terminal,
        winner=jnp.where(terminal, winner, jnp.int8(-1)),
        turn_start_board=new_turn_start,
        rep_hist=new_hist,
        rep_ptr=new_ptr,
    )
