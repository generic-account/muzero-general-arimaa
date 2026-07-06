"""State pytree for the vectorized JAX Arimaa engine."""

import chex
import jax.numpy as jnp
from flax import struct

from . import constants as C


@struct.dataclass
class State:
    """A single Arimaa game state. All fields are JAX arrays so State is a pytree
    that can be ``vmap``/``jit``/``scan``'d and batched across devices.

    Batched states are just States whose fields carry a leading batch dim.
    """

    board: chex.Array          # int8 [8, 8], cell codes (see constants); [y, x]
    player: chex.Array         # int8 scalar, 0=GOLD 1=SILVER (side to move)
    steps_left: chex.Array     # int8 scalar, steps remaining this turn (0..4)
    terminated: chex.Array     # bool scalar
    winner: chex.Array         # int8 scalar, 0/1 winner if terminated else -1
    # Turn bookkeeping:
    turn_start_board: chex.Array  # int8 [8, 8], board at the start of this turn
    # Repetition rule: Zobrist hashes of turn-end positions (ring buffer) + cursor.
    rep_hist: chex.Array          # uint32 [C.REP_HISTORY]
    rep_ptr: chex.Array           # int32 scalar, next write slot


def empty_state() -> State:
    return State(
        board=jnp.zeros((C.BOARD, C.BOARD), jnp.int8),
        player=jnp.int8(C.GOLD),
        steps_left=jnp.int8(4),
        terminated=jnp.bool_(False),
        winner=jnp.int8(-1),
        turn_start_board=jnp.zeros((C.BOARD, C.BOARD), jnp.int8),
        rep_hist=jnp.zeros((C.REP_HISTORY,), jnp.uint32),
        rep_ptr=jnp.int32(0),
    )
