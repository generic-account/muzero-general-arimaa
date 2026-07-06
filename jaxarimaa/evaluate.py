"""Evaluation: play the current network (via search) against a uniform-random
opponent, vectorized over a batch of games. Returns win/loss/unfinished counts.

Our side controls every step while it is the side to move (Arimaa turns are
multi-step); the random opponent picks a uniform legal action on its steps.
"""

import functools

import jax
import jax.numpy as jnp

from . import env as jenv
from . import search
from . import constants as C


def _random_legal_action(rng, states):
    legal = jax.vmap(jenv.legal_action_mask)(states)
    g = jax.random.gumbel(rng, legal.shape)
    scored = jnp.where(legal, g, -jnp.inf)
    return jnp.argmax(scored, axis=-1)


def _run_search_impl(fast):
    if fast:
        from . import fast_search
        return fast_search.run_search
    return search.run_search


@functools.partial(jax.jit, static_argnums=(0, 4, 5, 6, 7, 8, 9))
def play_vs_random(model, params, rng, our_color, n_games, max_steps,
                   num_sims, max_considered, features=None, fast=False):
    rng, kinit = jax.random.split(rng)
    states = jax.vmap(jenv.init_state)(jax.random.split(kinit, n_games))
    done = jnp.zeros((n_games,), bool)
    result = jnp.full((n_games,), -1, jnp.int8)

    def body(carry, _):
        states, done, result, rng = carry
        rng, ks, kr = jax.random.split(rng, 3)
        out = _run_search_impl(fast)(model, params, ks, states, num_sims,
                                     max_considered, features)
        rnd = _random_legal_action(kr, states)
        use_ours = states.player == our_color
        action = jnp.where(use_ours, out.action, rnd)

        nstates = jax.vmap(jenv.step)(states, action)
        # freeze already-finished lanes
        nstates = jenv.where_state(done, states, nstates)

        newly = nstates.terminated & (~done)
        result = jnp.where(newly, nstates.winner, result)
        done = done | nstates.terminated
        return (nstates, done, result, rng), None

    (states, done, result, _), _ = jax.lax.scan(
        body, (states, done, result, rng), None, length=max_steps
    )
    wins = jnp.sum((result == our_color) & done)
    losses = jnp.sum((result == (1 - our_color)) & done)
    unfinished = jnp.sum(~done)
    return wins, losses, unfinished


@functools.partial(jax.jit, static_argnums=(0, 4, 5, 6, 7, 8, 9, 10))
def play_match(model, params_a, params_b, rng, a_color, n_games, max_steps,
               num_sims, max_considered, features=None, fast=False):
    """Play `params_a` (as `a_color`) vs `params_b` via search, batched over games.
    Returns (a_wins, b_wins, unfinished). Used for the arena Elo metric (learner vs anchor).
    """
    rng, kinit = jax.random.split(rng)
    states = jax.vmap(jenv.init_state)(jax.random.split(kinit, n_games))
    done = jnp.zeros((n_games,), bool)
    result = jnp.full((n_games,), -1, jnp.int8)

    def body(carry, _):
        states, done, result, rng = carry
        rng, ka, kb = jax.random.split(rng, 3)
        impl = _run_search_impl(fast)
        out_a = impl(model, params_a, ka, states, num_sims, max_considered, features)
        out_b = impl(model, params_b, kb, states, num_sims, max_considered, features)
        action = jnp.where(states.player == a_color, out_a.action, out_b.action)
        nstates = jax.vmap(jenv.step)(states, action)
        nstates = jenv.where_state(done, states, nstates)  # freeze finished lanes
        newly = nstates.terminated & (~done)
        result = jnp.where(newly, nstates.winner, result)
        done = done | nstates.terminated
        return (nstates, done, result, rng), None

    (states, done, result, _), _ = jax.lax.scan(
        body, (states, done, result, rng), None, length=max_steps)
    a_wins = jnp.sum((result == a_color) & done)
    b_wins = jnp.sum((result == (1 - a_color)) & done)
    unfinished = jnp.sum(~done)
    return a_wins, b_wins, unfinished
