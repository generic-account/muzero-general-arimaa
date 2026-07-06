"""Vectorized self-play, sharded across the device mesh.

`_rollout` plays a per-device batch of games with lax.scan (search -> record ->
step -> auto-reset), then computes value targets by a reverse scan (game outcome
back-propagated, net-value bootstrap for truncated tails).

`make_generate` wraps `_rollout` in `shard_map` over the 'data' axis so EACH device
plays a DISTINCT subset of games (folding the device's global axis index — and the
host's process index — into the rng). On 1 device this is a no-op. This is what makes
self-play actually scale with the slice (otherwise every chip replays the same games).
"""

import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P

from . import env as jenv
from . import search


def _rollout(model, params, rng, batch, max_steps, mcts, features, sp_knobs):
    """Self-play `batch` games for `max_steps`; return dict of [T, batch, ...].

    Optional (feature-gated): playout-cap randomization (a batch-uniform coin picks a
    cheap `fast_sims` search vs the `full` search each step; only full-search moves get
    training weight) and resign/adjudication (end a game once |root value| clears a
    threshold, so decided games finish and the lane resets to a fresh game).
    """
    num_sims, max_considered = mcts
    resign_thresh, full_prob, fast_sims, outcome_w = sp_knobs
    playout_cap = features is not None and features.playout_cap
    resign = features is not None and features.resign
    if features is not None and features.fast_search:
        from . import fast_search as search_impl
    else:
        search_impl = search
    rng, kinit = jax.random.split(rng)
    states = jax.vmap(jenv.init_state)(jax.random.split(kinit, batch))

    def _search(sims):
        def branch(operand):
            s, k = operand
            out = search_impl.run_search(model, params, k, s, sims, max_considered,
                                         features)
            return out.action, out.action_weights, out.search_tree.node_values[:, 0]
        return branch

    def body(carry, _):
        states, rng = carry
        if playout_cap:
            rng, ks, kr, kc = jax.random.split(rng, 4)
            is_full = jax.random.bernoulli(kc, full_prob)  # batch-uniform coin
            action, weights, root_v = jax.lax.cond(
                is_full, _search(num_sims), _search(fast_sims), (states, ks))
            weight_step = jnp.where(is_full, 1.0, 0.0)
        else:
            rng, ks, kr = jax.random.split(rng, 3)  # no extra draw when disabled
            action, weights, root_v = _search(num_sims)((states, ks))
            weight_step = 1.0

        rec = {
            "obs": jax.vmap(lambda s: jenv.observe(s, features))(states),
            "policy_target": weights,
            "player": states.player,
            "weight": jnp.full((batch,), weight_step, jnp.float32),
        }
        nstates = jax.vmap(jenv.step)(states, action)
        if resign:
            adj = jnp.abs(root_v) > resign_thresh
            adj_winner = jnp.where(root_v > 0, states.player,
                                   1 - states.player).astype(jnp.int8)
            nstates = nstates.replace(
                terminated=nstates.terminated | adj,
                winner=jnp.where(adj & (~nstates.terminated), adj_winner, nstates.winner))
        rec["term"] = nstates.terminated
        rec["winner"] = nstates.winner
        rec["root_v"] = root_v.astype(jnp.float32)
        fresh = jax.vmap(jenv.init_state)(jax.random.split(kr, batch))
        nstates = jenv.where_state(nstates.terminated, fresh, nstates)  # auto-reset
        return (nstates, rng), rec

    (final_states, _), recs = jax.lax.scan(body, (states, rng), None, length=max_steps)

    # Bootstrap value for truncated tails: net value at the final rolled-to states.
    fobs = jax.vmap(lambda s: jenv.observe(s, features))(final_states)
    _, boot_val, _ = jax.vmap(lambda o: model.apply(params, o))(fobs)

    # Reverse scan producing, per step from the side-to-move perspective:
    #  value_target in [-1,1] (terminal -> outcome; else next value sign-flipped iff the
    #  mover changed; truncated tail -> bootstrap), and moves_left_target = normalized
    #  plies to game end (terminal -> 0; else next+1; capped; truncated tail -> capped).
    MLCAP = 64.0

    def back(carry, step):
        next_player, v, ml = carry
        player, term, winner = step["player"], step["term"], step["winner"]
        outcome = jnp.where(winner == player, 1.0, -1.0)
        sign = jnp.where(player == next_player, 1.0, -1.0)
        v_t = jnp.where(term, outcome, sign * v).astype(jnp.float32)
        ml_t = jnp.where(term, 0.0, jnp.minimum(ml + 1.0, MLCAP))
        return (player, v_t, ml_t), (v_t, (ml_t / MLCAP).astype(jnp.float32))

    _, (value_target, moves_left_target) = jax.lax.scan(
        back,
        (final_states.player, boot_val.astype(jnp.float32),
         jnp.full(boot_val.shape, MLCAP, jnp.float32)),
        {"player": recs["player"], "term": recs["term"], "winner": recs["winner"]},
        reverse=True,
    )
    # Blend the (often truncation-bootstrapped) outcome target with the SEARCH
    # root value: search values incorporate real lookahead, breaking the
    # self-confirming value collapse seen when most games truncate. Terminal
    # steps keep their exact ground-truth outcome.
    blended = outcome_w * value_target + (1.0 - outcome_w) * recs["root_v"]
    value_target = jnp.where(recs["term"], value_target, blended)

    # Only store aux targets when their head is enabled — otherwise they are dead
    # weight in the replay buffer (HBM). Baseline (heads off) carries neither.
    out = {
        "obs": recs["obs"],
        "policy_target": recs["policy_target"],
        "value_target": value_target,
        "weight": recs["weight"],
    }
    if features is not None and features.moves_left_head:
        out["moves_left_target"] = moves_left_target
    if features is not None and features.mtp:
        # MTP target: the NEXT step's value; masked at the last step and at game
        # boundaries (where the next step belongs to a freshly-reset game).
        T = value_target.shape[0]
        out["mtp_value_target"] = jnp.concatenate([value_target[1:], value_target[-1:]], 0)
        not_last = (jnp.arange(T) < T - 1)[:, None]
        out["mtp_mask"] = (not_last & (~recs["term"])).astype(jnp.float32)
    return out


def make_generate(mesh, model, batch_size, max_steps, mcts, features=None,
                  sp_knobs=(0.9, 0.25, 8, 0.5)):
    """Build a jitted, sharded self-play function `(params, rng) -> recs [T,B,...]`.

    Compiled once and reused across iterations (model/sizes/features/knobs are static).
    `sp_knobs` = (resign_threshold, full_search_prob, fast_sims, value_target_outcome_weight).
    """
    n = mesh.shape["data"]
    if batch_size % n:
        raise ValueError(f"batch_size {batch_size} must divide mesh size {n}")
    per = batch_size // n

    def per_shard(params, rng):
        r = jax.random.fold_in(rng, jax.lax.axis_index("data"))
        return _rollout(model, params, r, per, max_steps, mcts, features, sp_knobs)

    sharded = jax.shard_map(per_shard, mesh=mesh, in_specs=(P(), P()),
                            out_specs=P(None, "data"), check_vma=False)
    jitted = jax.jit(sharded)

    def generate(params, rng):
        return jitted(params, jax.random.fold_in(rng, jax.process_index()))

    return generate


def flatten_samples(recs):
    """[T, B, ...] -> [B*T, ...], keeping the 'data'-sharded game axis leading and
    contiguous (swap T/B first) so downstream sharding stays clean."""
    def f(x):
        x = jnp.swapaxes(x, 0, 1)              # [B, T, ...]
        return x.reshape((-1,) + x.shape[2:])  # [B*T, ...]
    return {k: f(v) for k, v in recs.items()}
