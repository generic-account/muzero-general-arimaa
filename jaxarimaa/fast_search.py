"""Batched sequential halving for Gumbel MuZero — a drop-in, wave-parallel policy.

The observation (see docs/JAX_REBASE_SCOPE.md and the mctx feasibility study):
`mctx.gumbel_muzero_policy` runs `num_simulations` strictly sequential
simulation steps, each costing one `[B]` `recurrent_fn` call plus a round of
small tree ops. But Sequential Halving visits the considered root actions in
*rounds* (visit every currently-considered action once, halve, repeat), and
within a round the visits are independent:

  * the set visited in a round is exactly {actions with visit_count == cv};
  * each visit descends only into its own root-action subtree;
  * root-level backups are commutative (count-weighted mean).

So the per-simulation loop can be regrouped into one batched `recurrent_fn`
call of shape `[B * K_round]` per round — `num_simulations=32,
max_num_considered_actions=16` collapses from 32 sequential network/env calls
to 4 — while building the *same tree* (up to floating-point summation order
within a round) and therefore preserving Gumbel's policy-improvement guarantee
and mctx's training targets.

`batched_gumbel_muzero_policy` below mirrors the `mctx.gumbel_muzero_policy`
signature and returns a real `mctx.PolicyOutput` over a real `mctx` tree; it
depends only on jax + mctx internals (no project code), so it can be upstreamed.

Known divergence from mctx (documented, tested): mctx shrinks the halving
schedule per batch row when a row has fewer than `max_num_considered_actions`
valid actions. Rounds here are static-width; rows with fewer valid actions
re-visit their best valid action for the surplus slots (mctx itself re-expands
nodes at max_depth similarly). This only affects rows near terminal states.
"""

import functools

import jax
import jax.numpy as jnp
import mctx
from mctx._src import action_selection as mctx_action_selection
from mctx._src import base as mctx_base
from mctx._src import qtransforms as mctx_qtransforms
from mctx._src import search as mctx_search
from mctx._src import seq_halving as mctx_seq_halving
from mctx._src.tree import Tree


def _rounds_from_schedule(max_num_considered_actions, num_simulations):
    """Run-length encode mctx's visit schedule into (considered_visit, width) rounds.

    Derived from mctx's own `get_sequence_of_considered_visits`, so the phase
    widths (including the max(2, m//2) and extra-visit quirks) match exactly.
    """
    seq = mctx_seq_halving.get_sequence_of_considered_visits(
        max_num_considered_actions, num_simulations)
    rounds = []
    i = 0
    while i < len(seq):
        j = i
        while j < len(seq) and seq[j] == seq[i]:
            j += 1
        rounds.append((seq[i], j - i))
        i = j
    return rounds


def _mask_invalid_actions(logits, invalid_actions):
    """mctx's masking: renormalize by max, set invalid to dtype-min."""
    if invalid_actions is None:
        return logits
    logits = logits - jnp.max(logits, axis=-1, keepdims=True)
    min_logit = jnp.finfo(logits.dtype).min
    return jnp.where(invalid_actions, min_logit, logits)


def _write_node(tree, parent_index, action, next_node_index, prior_logits,
                value, reward, discount, embedding):
    """The tail of mctx.search.expand: write an evaluated node + edge stats."""
    batch_range = jnp.arange(parent_index.shape[0])
    tree = mctx_search.update_tree_node(
        tree, next_node_index, prior_logits, value, embedding)
    batch_update = mctx_search.batch_update
    return tree.replace(
        children_index=batch_update(
            tree.children_index, next_node_index, parent_index, action),
        children_rewards=batch_update(
            tree.children_rewards, reward, parent_index, action),
        children_discounts=batch_update(
            tree.children_discounts, discount, parent_index, action),
        parents=batch_update(tree.parents, parent_index, next_node_index),
        action_from_parent=batch_update(
            tree.action_from_parent, action, next_node_index),
    )


def _make_forced_simulate(interior_fn):
    """A vmapped tree-descent like mctx.search.simulate, but the depth-0 (root)
    action is a per-batch input instead of coming from a selection function —
    the round's considered action. Body mirrors mctx._src.search.simulate.
    """

    @functools.partial(jax.vmap, in_axes=[0, 0, 0, None], out_axes=0)
    def simulate_forced(rng_key, tree, forced_action, max_depth):
        def selection_fn(key, t, node_index, depth):
            interior = interior_fn(key, t, node_index, depth)
            return jnp.where(depth == 0, forced_action, interior).astype(jnp.int32)

        class _State(dict):
            pass

        def cond_fun(state):
            return state["is_continuing"]

        def body_fun(state):
            node_index = state["next_node_index"]
            rng, sel_key = jax.random.split(state["rng_key"])
            action = selection_fn(sel_key, tree, node_index, state["depth"])
            next_node_index = tree.children_index[node_index, action]
            depth = state["depth"] + 1
            return {
                "rng_key": rng,
                "node_index": node_index,
                "action": action,
                "next_node_index": next_node_index,
                "depth": depth,
                "is_continuing": jnp.logical_and(
                    next_node_index != Tree.UNVISITED, depth < max_depth),
            }

        root_index = jnp.array(Tree.ROOT_INDEX, dtype=jnp.int32)
        state = {
            "rng_key": rng_key,
            "node_index": jnp.full((), Tree.NO_PARENT, jnp.int32),
            "action": jnp.full((), Tree.NO_PARENT, jnp.int32),
            "next_node_index": root_index,
            "depth": jnp.zeros((), jnp.int32),
            "is_continuing": jnp.array(True),
        }
        end = jax.lax.while_loop(cond_fun, body_fun, state)
        return end["node_index"], end["action"]

    return simulate_forced


def batched_gumbel_muzero_policy(
    params,
    rng_key,
    root,
    recurrent_fn,
    num_simulations,
    invalid_actions=None,
    max_depth=None,
    qtransform=mctx_qtransforms.qtransform_completed_by_mix_value,
    max_num_considered_actions=16,
    gumbel_scale=1.0,
):
    """Wave-parallel Gumbel MuZero. Mirrors `mctx.gumbel_muzero_policy`.

    `recurrent_fn` must be shape-polymorphic in its leading (batch) dimension:
    it is called with batch `B * K_round` instead of `B`. Any vmapped
    recurrent_fn (the common case) satisfies this.
    """
    batch_size = root.value.shape[0]
    num_actions = root.prior_logits.shape[-1]
    if invalid_actions is None:
        invalid_actions = jnp.zeros_like(root.prior_logits)
    if max_depth is None:
        max_depth = num_simulations

    # Same masking + gumbel sampling (and rng consumption order) as mctx, so
    # results are directly comparable under the same rng_key.
    root = root.replace(
        prior_logits=_mask_invalid_actions(root.prior_logits, invalid_actions))
    rng_key, gumbel_rng = jax.random.split(rng_key)
    gumbel = gumbel_scale * jax.random.gumbel(
        gumbel_rng, shape=root.prior_logits.shape, dtype=root.prior_logits.dtype)

    extra_data = mctx_action_selection.GumbelMuZeroExtraData(root_gumbel=gumbel)
    tree = mctx_search.instantiate_tree_from_root(
        root, num_simulations, root_invalid_actions=invalid_actions,
        extra_data=extra_data)

    interior_fn = functools.partial(
        mctx_action_selection.gumbel_muzero_interior_action_selection,
        qtransform=qtransform)
    simulate_forced = _make_forced_simulate(interior_fn)
    batch_range = jnp.arange(batch_size)
    rounds = _rounds_from_schedule(max_num_considered_actions, num_simulations)

    sim_offset = 0
    for considered_visit, width in rounds:
        # --- select this round's considered set (all actions with visits == cv,
        # ranked by gumbel + logits + completed Q; exactly K of them on-schedule).
        summary_visits = tree.children_visits[:, Tree.ROOT_INDEX]
        completed_q = jax.vmap(qtransform, in_axes=[0, None])(
            tree, Tree.ROOT_INDEX)
        scores = mctx_seq_halving.score_considered(
            considered_visit, gumbel, tree.children_prior_logits[:, Tree.ROOT_INDEX],
            completed_q, summary_visits)
        _, top_actions = jax.lax.top_k(scores, width)  # [B, width]
        # Rows with fewer valid actions than the schedule expects: fall back to
        # the row's best action (top-1 is always valid when any action is).
        selected_invalid = jnp.take_along_axis(
            invalid_actions, top_actions, axis=1).astype(bool)
        top_actions = jnp.where(
            selected_invalid, top_actions[:, :1], top_actions)

        # --- descend each considered action's subtree (independent per action).
        parents, actions, next_idxs = [], [], []
        for k in range(width):
            rng_key, simulate_key = jax.random.split(rng_key)
            simulate_keys = jax.random.split(simulate_key, batch_size)
            parent_index, action = simulate_forced(
                simulate_keys, tree, top_actions[:, k], max_depth)
            next_node_index = tree.children_index[batch_range, parent_index, action]
            next_node_index = jnp.where(
                next_node_index == Tree.UNVISITED, sim_offset + k + 1,
                next_node_index)
            parents.append(parent_index)
            actions.append(action)
            next_idxs.append(next_node_index)

        # --- ONE batched recurrent_fn call for the whole round: [B * width].
        parents_f = jnp.stack(parents, axis=1).reshape(-1)        # b-major
        actions_f = jnp.stack(actions, axis=1).reshape(-1)
        batch_f = jnp.repeat(batch_range, width)
        embedding_f = jax.tree_util.tree_map(
            lambda x: x[batch_f, parents_f], tree.embeddings)
        rng_key, expand_key = jax.random.split(rng_key)
        step, new_embedding_f = recurrent_fn(
            params, expand_key, actions_f, embedding_f)

        # --- write the round's nodes + backups (cheap tree ops, no forwards).
        for k in range(width):
            step_k_logits = step.prior_logits.reshape(
                batch_size, width, num_actions)[:, k]
            step_k_value = step.value.reshape(batch_size, width)[:, k]
            step_k_reward = step.reward.reshape(batch_size, width)[:, k]
            step_k_discount = step.discount.reshape(batch_size, width)[:, k]
            emb_k = jax.tree_util.tree_map(
                lambda x: x.reshape((batch_size, width) + x.shape[1:])[:, k],
                new_embedding_f)
            tree = _write_node(tree, parents[k], actions[k], next_idxs[k],
                               step_k_logits, step_k_value, step_k_reward,
                               step_k_discount, emb_k)
            tree = mctx_search.backward(tree, next_idxs[k])
        sim_offset += width

    # --- outputs: verbatim mctx.policies.gumbel_muzero_policy tail.
    summary = tree.summary()
    considered_visit = jnp.max(summary.visit_counts, axis=-1, keepdims=True)
    completed_qvalues = jax.vmap(qtransform, in_axes=[0, None])(
        tree, Tree.ROOT_INDEX)
    to_argmax = mctx_seq_halving.score_considered(
        considered_visit, gumbel, root.prior_logits, completed_qvalues,
        summary.visit_counts)
    action = mctx_action_selection.masked_argmax(to_argmax, invalid_actions)
    completed_search_logits = _mask_invalid_actions(
        root.prior_logits + completed_qvalues, invalid_actions)
    action_weights = jax.nn.softmax(completed_search_logits)
    return mctx_base.PolicyOutput(
        action=action, action_weights=action_weights, search_tree=tree)


# ---------------------------------------------------------------------------
# jaxarimaa wrapper: identical signature to search.run_search (drop-in).
# ---------------------------------------------------------------------------
@functools.partial(jax.jit, static_argnums=(0, 4, 5, 6))
def run_search(model, params, rng_key, states, num_simulations,
               max_num_considered_actions, features=None):
    from . import search as slow_search

    prior_logits, value, legal = slow_search._eval(model, params, states, features)
    root = mctx.RootFnOutput(prior_logits=prior_logits, value=value,
                             embedding=states)
    return batched_gumbel_muzero_policy(
        params=params,
        rng_key=rng_key,
        root=root,
        recurrent_fn=slow_search.make_recurrent_fn(model, features),
        num_simulations=num_simulations,
        invalid_actions=~legal,
        max_num_considered_actions=max_num_considered_actions,
    )
