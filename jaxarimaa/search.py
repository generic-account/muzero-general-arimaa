"""mctx AlphaZero search: PUCT/Gumbel MCTS over the REAL (exact) simulator.

Because our environment is an exact JAX simulator, the mctx "embedding" is the
actual game State and `recurrent_fn` steps it for real (no learned dynamics).

Two-player perspective handling: values/priors are always from the perspective of
the side to move at each node. Descending an edge multiplies by a `discount`:
  * +1 if the child has the SAME side to move (mid-turn: Arimaa turns are multi-step)
  * -1 if the side to move flipped (turn ended -> opponent)
  *  0 into a terminal node (its own value is irrelevant; the result is the reward)
Terminal reward is +1/-1 from the PARENT mover's perspective.

All functions are batched (leading batch dim) and jittable.
"""

import functools

import jax
import jax.numpy as jnp
import mctx

from . import env as jenv
from . import constants as C

_NEG = -1e9


def _mask_logits(logits, legal):
    """Set illegal-action logits to a large negative; if a row has NO legal action
    (terminal node), fall back to uniform zeros so softmax stays finite."""
    any_legal = jnp.any(legal, axis=-1, keepdims=True)
    masked = jnp.where(legal, logits, _NEG)
    return jnp.where(any_legal, masked, jnp.zeros_like(logits))


def _eval(model, params, states, features):
    """Batched network eval + legal-masking for a batch of States."""
    obs = jax.vmap(lambda s: jenv.observe(s, features))(states)
    logits, value, _ = jax.vmap(lambda o: model.apply(params, o))(obs)
    legal = jax.vmap(jenv.legal_action_mask)(states)
    return _mask_logits(logits, legal), value, legal


def make_recurrent_fn(model, features):
    def recurrent_fn(params, rng_key, action, embedding):
        state = embedding
        prev_player = state.player
        prev_term = state.terminated

        nstate = jax.vmap(jenv.step)(state, action)
        # Guard: never move from an already-terminal node (keep it absorbing).
        nstate = jenv.where_state(prev_term, state, nstate)

        now_term = nstate.terminated
        newly_term = now_term & (~prev_term)
        win = nstate.winner == prev_player.astype(nstate.winner.dtype)
        reward = jnp.where(newly_term, jnp.where(win, 1.0, -1.0), 0.0).astype(jnp.float32)
        same_player = nstate.player == prev_player
        discount = jnp.where(now_term, 0.0,
                             jnp.where(same_player, 1.0, -1.0)).astype(jnp.float32)

        prior_logits, value, _ = _eval(model, params, nstate, features)
        value = jnp.where(now_term, 0.0, value).astype(jnp.float32)
        out = mctx.RecurrentFnOutput(
            reward=reward, discount=discount, prior_logits=prior_logits, value=value
        )
        return out, nstate

    return recurrent_fn


@functools.partial(jax.jit, static_argnums=(0, 4, 5, 6))
def run_search(model, params, rng_key, states, num_simulations,
               max_num_considered_actions, features=None):
    """Run Gumbel-MuZero search (= AlphaZero here) from a batch of root States.

    `features` (a FeaturesConfig or None) selects the observation planes; it must
    match what the network's params were trained with. Returns the mctx PolicyOutput;
    use `.action` (chosen move) and `.action_weights` (improved policy target).
    """
    prior_logits, value, legal = _eval(model, params, states, features)
    root = mctx.RootFnOutput(prior_logits=prior_logits, value=value, embedding=states)
    return mctx.gumbel_muzero_policy(
        params=params,
        rng_key=rng_key,
        root=root,
        recurrent_fn=make_recurrent_fn(model, features),
        num_simulations=num_simulations,
        invalid_actions=~legal,
        max_num_considered_actions=max_num_considered_actions,
    )
