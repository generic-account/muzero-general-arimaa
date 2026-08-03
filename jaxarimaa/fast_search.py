"""jaxarimaa glue for the batched Gumbel policy.

Everything algorithmic lives in `mctx_batched.py` (pure jax + mctx, kept
upstreamable). This module holds only what is specific to this project:

  * `pack_states` / `unpack_states` — the tree-embedding state packing
  * `run_search` — a drop-in replacement for `search.run_search`
"""

import functools

import jax
import jax.numpy as jnp
import mctx

from .mctx_batched import batched_gumbel_muzero_policy  # noqa: F401 (re-export)

# ---------------------------------------------------------------------------
# Packed tree embeddings: the per-simulation-round gather/scatter of parent
# node States dominated the whole device profile (58% of busy time on the
# s2pilot trace, 2026-07-12) because the State pytree stores 8 separate
# small-row arrays per node — int8 [8,8] boards are 64-byte gather rows,
# far below TPU tile size, so each gather ran at ~2% bandwidth efficiency
# and each round dispatched 8 of them. Packing the State into ONE
# uint32[..., 128] buffer (bit-exact bijection, 512-byte aligned rows) turns
# that into a single well-tiled gather/scatter per round.
# ---------------------------------------------------------------------------
_PACKED_WORDS = 128  # 98 used (16 board + 16 turn_start + 64 rep + 2), padded
                     # to the 128-lane vector width


def pack_states(states):
    """State pytree [...] -> uint32 [..., 128]. Exact bijection with
    unpack_states (bitcasts only; padding is zeros)."""

    def board_words(a):  # int8 [..., 8, 8] -> uint32 [..., 16]
        return jax.lax.bitcast_convert_type(
            a.reshape(a.shape[:-2] + (16, 4)), jnp.uint32)

    scalars = jnp.stack(
        [states.player, states.steps_left, states.winner,
         states.terminated.astype(jnp.int8)], axis=-1)          # int8 [..., 4]
    words = jnp.concatenate([
        board_words(states.board),                              # 0:16
        board_words(states.turn_start_board),                   # 16:32
        states.rep_hist,                                        # 32:96
        jax.lax.bitcast_convert_type(scalars, jnp.uint32)[..., None],  # 96
        jax.lax.bitcast_convert_type(states.rep_ptr, jnp.uint32)[..., None],  # 97
    ], axis=-1)
    pad = jnp.zeros(words.shape[:-1] + (_PACKED_WORDS - words.shape[-1],),
                    jnp.uint32)
    return jnp.concatenate([words, pad], axis=-1)


def unpack_states(words):
    """uint32 [..., 128] -> State pytree [...] (inverse of pack_states)."""
    from .types import State

    def words_board(w):  # uint32 [..., 16] -> int8 [..., 8, 8]
        return jax.lax.bitcast_convert_type(w, jnp.int8).reshape(
            w.shape[:-1] + (8, 8))

    scalars = jax.lax.bitcast_convert_type(words[..., 96], jnp.int8)  # [..., 4]
    return State(
        board=words_board(words[..., 0:16]),
        turn_start_board=words_board(words[..., 16:32]),
        rep_hist=words[..., 32:96],
        player=scalars[..., 0],
        steps_left=scalars[..., 1],
        winner=scalars[..., 2],
        terminated=scalars[..., 3] != 0,
        rep_ptr=jax.lax.bitcast_convert_type(words[..., 97], jnp.int32),
    )

# ---------------------------------------------------------------------------
# jaxarimaa wrapper: identical signature to search.run_search (drop-in).
# ---------------------------------------------------------------------------
@functools.partial(jax.jit, static_argnums=(0, 4, 5, 6, 7, 8))
def run_search(model, params, rng_key, states, num_simulations,
               max_num_considered_actions, features=None, prior_temp=1.0,
               ml_steer=0.0):
    from . import search as slow_search

    prior_logits, value, legal, _ = slow_search._eval(model, params, states, features)
    if prior_temp != 1.0:
        # Anti-self-sharpening (optima/KataGo): flatten the net's priors fed to
        # search so exploration survives the policy's own sharpening feedback.
        prior_logits = prior_logits / prior_temp
    root = mctx.RootFnOutput(prior_logits=prior_logits, value=value,
                             embedding=pack_states(states))
    inner_fn = slow_search.make_recurrent_fn(model, features, prior_temp,
                                             ml_steer)

    def recurrent_fn(params_, key, actions, packed):
        out, nstates = inner_fn(params_, key, actions, unpack_states(packed))
        return out, pack_states(nstates)

    return batched_gumbel_muzero_policy(
        params=params,
        rng_key=rng_key,
        root=root,
        recurrent_fn=recurrent_fn,
        num_simulations=num_simulations,
        invalid_actions=~legal,
        max_num_considered_actions=max_num_considered_actions,
    )
