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

v2 additionally collapses the *tree ops* within a round from K-sequential to
wave-parallel (v1 batched only the recurrent_fn call):

  * descent: read-only on the round's tree snapshot, so all K lanes descend
    with one nested-vmap `[K, B]` lockstep while_loop instead of K calls;
  * node writes: the K new nodes and the K (parent, action) edges are distinct
    on-schedule, so the K per-array scatters fuse into ONE `[B*K]`-wide scatter
    per tree array (mirroring mctx's `update_tree_node` + `expand` tail);
  * backups: the K leaf->root paths are disjoint except at the root. One
    lockstep `[B, K]` scan walks all lanes up at once, recording per hop the
    (parent, action, propagated leaf_value, updated child value); non-root
    stats are then written with disjoint scatters, visit counts with
    scatter-adds, and the root value with the associative closed form
    v_new = (v*n + sum_k leaf_k) / (n + K) — mathematically identical to
    mctx's K sequential incremental means (fp summation order differs at the
    root only; root selection reads raw_values/children stats, not
    node_values, so visit counts still match mctx exactly).

`batched_gumbel_muzero_policy` below mirrors the `mctx.gumbel_muzero_policy`
signature and returns a real `mctx.PolicyOutput` over a real `mctx` tree; it
depends only on jax + mctx internals (no project code), so it can be upstreamed.

Known divergences from mctx (documented, tested — both are properties of the
per-ROUND regrouping present since v1, not of the v2 op batching):

  1. mctx shrinks the halving schedule per batch row when a row has fewer than
     `max_num_considered_actions` valid actions. Rounds here are static-width;
     rows with fewer valid actions re-visit their best valid action for the
     surplus slots (mctx itself re-expands nodes at max_depth similarly).
     This only affects rows near terminal states.
  2. mctx recomputes the root completed-Q transform after EVERY simulation, so
     within a halving round (2w candidates at the considered visit level, only
     w visits) the transform's global terms (visit_scale via max visits,
     rescale min/max, mixed value) drift between picks and mctx's sequential
     argmax can select a different w-subset than the round-start top-w used
     here. The visited SET can then differ on near-tied candidates (likelier
     for untrained networks whose Q-values are nearly equal). Rounds that
     visit ALL candidates at the level (extra-visit rounds, and round 0 where
     all candidates share the mixed completed value) are order-independent and
     exact. With a drift-free qtransform (e.g. value_scale=0) the whole search
     matches mctx's visit counts exactly — see tests_fast_search_v2.py.
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


def _write_nodes_batched(tree, batch_f, parent_f, action_f, next_f,
                         prior_logits_f, value_f, reward_f, discount_f,
                         embedding_f):
    """The tail of mctx.search.expand + update_tree_node, for a whole round.

    All arguments are flat `[B*K]` (b-major); the K new node indices per row
    are distinct and the K (parent, action) edges per row are distinct
    on-schedule, so each of mctx's K sequential per-array scatters collapses
    into one K-wide scatter. `node_visits` uses `.add` (identical to mctx's
    read-modify-write `set(old+1)` for distinct indices, and correct for the
    documented off-schedule duplicate re-expansions where `set` would race).
    """
    return tree.replace(
        # update_tree_node fields (at the new nodes).
        children_prior_logits=tree.children_prior_logits.at[
            batch_f, next_f].set(prior_logits_f),
        raw_values=tree.raw_values.at[batch_f, next_f].set(value_f),
        node_values=tree.node_values.at[batch_f, next_f].set(value_f),
        node_visits=tree.node_visits.at[batch_f, next_f].add(1),
        embeddings=jax.tree_util.tree_map(
            lambda t, s: t.at[batch_f, next_f].set(s),
            tree.embeddings, embedding_f),
        # expand tail fields (at the (parent, action) edges / new nodes).
        children_index=tree.children_index.at[
            batch_f, parent_f, action_f].set(next_f),
        children_rewards=tree.children_rewards.at[
            batch_f, parent_f, action_f].set(reward_f),
        children_discounts=tree.children_discounts.at[
            batch_f, parent_f, action_f].set(discount_f),
        parents=tree.parents.at[batch_f, next_f].set(parent_f),
        action_from_parent=tree.action_from_parent.at[
            batch_f, next_f].set(action_f),
    )


def _backward_batched(tree, leaf_indices, num_hops):
    """mctx.search.backward for K leaves per batch row at once.

    Walks all `[B, K]` lanes leaf->root in lockstep for `num_hops` hops
    (a static bound on the max leaf depth this round), recording per hop the
    (parent, action, propagated leaf_value, updated child value). The K paths
    are disjoint except at the root, so:

      * non-root `node_values` and `children_values` follow mctx's exact
        single-visitor expressions (bitwise identical on-schedule);
      * visit counts are scatter-adds (exact, integer);
      * the root value uses the associative closed form
        (v*n + sum_k leaf_k) / (n + K), equal to mctx's K sequential
        incremental means up to fp summation order.

    Lanes whose paths are shorter than `num_hops` are masked by routing their
    scatter indices out of bounds (`mode="drop"`).
    """
    batch_size, num_lanes = leaf_indices.shape
    num_nodes = tree.node_values.shape[1]
    b_idx = jnp.arange(batch_size)[:, None]  # [B, 1], broadcasts over K

    node_values = tree.node_values
    node_visits = tree.node_visits

    def hop(carry, _):
        index, leaf_value, child_value, active = carry  # each [B, K]
        parent = tree.parents[b_idx, index]
        action = tree.action_from_parent[b_idx, index]
        reward = tree.children_rewards[b_idx, parent, action]
        discount = tree.children_discounts[b_idx, parent, action]
        leaf_value = reward + discount * leaf_value
        is_root = parent == Tree.ROOT_INDEX
        record = (parent, action, leaf_value, child_value, active)
        # The parent's updated value: only this lane touches a non-root parent,
        # so this matches mctx's (v*count + leaf_value) / (count + 1) exactly.
        # It becomes the NEXT hop's children_values write (mctx writes
        # tree.node_values[index] *after* the previous hop updated it).
        count = node_visits[b_idx, parent]
        parent_value = (node_values[b_idx, parent] * count + leaf_value) / (
            count + 1.0)
        carry = (parent, leaf_value, parent_value,
                 jnp.logical_and(active, ~is_root))
        return carry, record

    init = (
        leaf_indices,
        node_values[b_idx, leaf_indices],  # backward starts from the leaf value
        node_values[b_idx, leaf_indices],  # first children_values write
        jnp.ones(leaf_indices.shape, dtype=bool),
    )
    _, (parent_h, action_h, leaf_h, child_h, active_h) = jax.lax.scan(
        hop, init, None, length=num_hops)  # each [H, B, K]

    # Flatten hops x lanes; mask by sending dropped entries out of bounds
    # (masked parents can be NO_PARENT == -1, which would WRAP, so this is
    # required for correctness, not just hygiene).
    mask = active_h.reshape(-1)
    par = jnp.where(mask, parent_h.reshape(-1), num_nodes)
    act = action_h.reshape(-1)
    bat = jnp.broadcast_to(b_idx[None, :, :], active_h.shape).reshape(-1)
    leaf_v = leaf_h.reshape(-1)
    child_v = child_h.reshape(-1)
    one = jnp.ones((), dtype=tree.children_visits.dtype)

    # Per-node visit increments and leaf_value sums (root gets all K lanes,
    # non-root nodes exactly one on-schedule).
    cnt = jnp.zeros_like(node_visits).at[bat, par].add(
        one, mode="drop")
    leaf_sum = jnp.zeros_like(node_values).at[bat, par].add(
        leaf_v, mode="drop")
    new_node_values = jnp.where(
        cnt > 0,
        (node_values * node_visits + leaf_sum) / (node_visits + cnt),
        node_values)

    return tree.replace(
        node_values=new_node_values,
        node_visits=node_visits + cnt,
        children_values=tree.children_values.at[bat, par, act].set(
            child_v, mode="drop"),
        children_visits=tree.children_visits.at[bat, par, act].add(
            one, mode="drop"),
    )


def _completed_q_and_score_subset(
    tree, batch_range, considered, gumbel, prior, considered_visit,
    logit_max_full, value_scale=0.1, maxvisit_init=50.0, epsilon=1e-8):
    """Bit-identical `score_considered(considered_visit, ...)` over a column
    subset, for the DEFAULT `qtransform_completed_by_mix_value` (value_scale=0.1,
    maxvisit_init=50, rescale_values=True, use_mixed_value=True, epsilon=1e-8).

    Rounds 1+ of Sequential Halving only ever consider actions with
    `visit == considered_visit >= 1`; those are always a subset of the round-0
    top-m `considered` set (the only root actions the search ever visits), and
    every other action scores `-inf` (score_considered's penalty) so it can
    never enter the top-k. So the full-1393-wide `vmap(qtransform)` +
    `score_considered` collapse to work over the `[B, m]` `considered` columns.

    Reproduces mctx's global terms from the subset (see qtransforms.py):
      * root prior softmax and gumbel are STATIC across rounds; passed in
        already gathered at `considered`.
      * every `considered` action stays visited>0 the whole search, so the
        completed array is {qvalue[considered]} plus the mixed value at the
        (always >= 1, since 1393 >> m) unvisited actions; hence the full-width
        rescale min/max reduce to min/max over (qvalue[considered], mixed) and
        `max(visit_counts)` reduces to the max over `considered`.
      * score_considered subtracts the FULL-width logit max (`logit_max_full`).
    """
    b = batch_range[:, None]
    # qvalues over the considered columns: rewards + discount * value at ROOT.
    q = (tree.children_rewards[b, Tree.ROOT_INDEX, considered]
         + tree.children_discounts[b, Tree.ROOT_INDEX, considered]
         * tree.children_values[b, Tree.ROOT_INDEX, considered])   # [B, m]
    visits = tree.children_visits[b, Tree.ROOT_INDEX, considered]   # [B, m]
    raw_value = tree.raw_values[:, Tree.ROOT_INDEX]                 # [B]

    # _compute_mixed_value over the subset (all considered actions are visited,
    # so `where(visit>0, .)` masks are all-true on the subset).
    sum_visit_counts = jnp.sum(visits, axis=-1)                     # [B]
    prior = jnp.maximum(jnp.finfo(prior.dtype).tiny, prior)
    sum_probs = jnp.sum(prior, axis=-1)                            # [B]
    weighted_q = jnp.sum(
        prior * q / jnp.where(sum_probs[:, None] > 0, sum_probs[:, None], 1.0),
        axis=-1)
    mixed = (raw_value + sum_visit_counts * weighted_q) / (sum_visit_counts + 1)

    # _rescale_qvalues: min/max over the full completed array = min/max over
    # (qvalue[considered], mixed), since all other (unvisited) actions complete
    # to `mixed` and there is always >= 1 unvisited action (1393 >> m).
    min_value = jnp.minimum(jnp.min(q, axis=-1), mixed)[:, None]
    max_value = jnp.maximum(jnp.max(q, axis=-1), mixed)[:, None]
    rescaled = (q - min_value) / jnp.maximum(max_value - min_value, epsilon)
    maxvisit = jnp.max(visits, axis=-1)                            # [B]
    visit_scale = (maxvisit_init + maxvisit)[:, None]
    completed_q = visit_scale * value_scale * rescaled            # [B, m]

    # score_considered over the subset (full-width logit max already given).
    logits_norm = tree.children_prior_logits[b, Tree.ROOT_INDEX, considered] \
        - logit_max_full[:, None]
    penalty = jnp.where(visits == considered_visit, 0.0, -jnp.inf)
    return jnp.maximum(-1e9, gumbel + logits_norm + completed_q) + penalty


def _make_forced_simulate(interior_fn):
    """A vmapped tree-descent like mctx.search.simulate, but the depth-0 (root)
    action is a per-batch input instead of coming from a selection function —
    the round's considered action. Body mirrors mctx._src.search.simulate.

    Doubly vmapped `[K, B]`: descent is read-only on the round's tree snapshot
    and the K considered subtrees are disjoint below the root (the root action
    is forced), so all K lanes run one lockstep while_loop over a broadcast
    tree instead of K sequential descents.
    """

    @functools.partial(jax.vmap, in_axes=[0, None, 0, None], out_axes=0)  # K
    @functools.partial(jax.vmap, in_axes=[0, 0, 0, None], out_axes=0)     # B
    def simulate_forced(rng_key, tree, forced_action, max_depth):
        def selection_fn(key, t, node_index, depth):
            interior = interior_fn(key, t, node_index, depth)
            return jnp.where(depth == 0, forced_action, interior).astype(jnp.int32)

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

    # Fast subset path is only exercised for the default completed-by-mix-value
    # qtransform with default params (see _completed_q_and_score_subset). For
    # any other qtransform we keep the general full-width vmap(qtransform) path.
    use_subset = (qtransform
                  is mctx_qtransforms.qtransform_completed_by_mix_value)
    # Root priors and their softmax never change during search -> static.
    root_logits = tree.children_prior_logits[:, Tree.ROOT_INDEX]  # [B, 1393]
    logit_max_full = jnp.max(root_logits, axis=-1)               # [B]
    prior_full = jax.nn.softmax(root_logits, axis=-1)            # [B, 1393]

    def select_full_width(considered_visit, width):
        summary_visits = tree.children_visits[:, Tree.ROOT_INDEX]
        completed_q = jax.vmap(qtransform, in_axes=[0, None])(
            tree, Tree.ROOT_INDEX)
        scores = mctx_seq_halving.score_considered(
            considered_visit, gumbel, root_logits, completed_q, summary_visits)
        _, top_actions = jax.lax.top_k(scores, width)  # [B, width]
        return _apply_fallback(top_actions)

    def _apply_fallback(top_actions):
        # Rows with fewer valid actions than the schedule expects: fall back to
        # the row's best action (top-1 is always valid when any action is).
        selected_invalid = jnp.take_along_axis(
            invalid_actions, top_actions, axis=1).astype(bool)
        return jnp.where(selected_invalid, top_actions[:, :1], top_actions)

    considered = None       # [B, m] round-0 top-m action ids (post-fallback)
    considered_prior = None  # [B, m] prior_full gathered at `considered`
    gumbel_sub = None        # [B, m] gumbel gathered at `considered`
    sim_offset = 0
    for round_index, (considered_visit, width) in enumerate(rounds):
        # --- select this round's considered set (all actions with visits == cv,
        # ranked by gumbel + logits + completed Q; exactly K of them on-schedule).
        if round_index == 0 or not use_subset:
            top_actions = select_full_width(considered_visit, width)
            if round_index == 0 and use_subset:
                # Remember the round-0 selection (post-fallback) as `considered`:
                # every subsequent round only visits a subset of it. Sort by
                # action id so that any duplicate ids from the fallback map to
                # adjacent columns (harmless: duplicates carry equal scores, so
                # the round 1+ scatter into the full-width score array below is
                # deterministic regardless of which duplicate wins the `.set`).
                considered = jnp.sort(top_actions, axis=-1)          # [B, m]
                considered_prior = jnp.take_along_axis(
                    prior_full, considered, axis=1)
                gumbel_sub = jnp.take_along_axis(gumbel, considered, axis=1)
        else:
            # Rounds 1+: only actions with `visit == considered_visit >= 1` are
            # eligible, always a subset of the round-0 `considered` set (the only
            # actions the search ever visits); every other action scores -inf.
            # So the O(1393) qtransform/softmax/score work is done over just the
            # `considered` columns, then scattered into a full-width -inf score
            # array so the O(1393) top_k reproduces mctx's exact fill/tie-break
            # for the off-schedule (fewer-eligible-than-width) rows too.
            scores_sub = _completed_q_and_score_subset(
                tree, batch_range, considered, gumbel_sub, considered_prior,
                considered_visit, logit_max_full)              # [B, m]
            scores = jnp.full(
                (batch_size, num_actions), -jnp.inf, dtype=scores_sub.dtype)
            scores = scores.at[batch_range[:, None], considered].set(scores_sub)
            _, top_actions = jax.lax.top_k(scores, width)      # [B, width]
            top_actions = _apply_fallback(top_actions)

        # --- descend all K considered subtrees at once (read-only, disjoint
        # below the root): one lockstep [K, B] while_loop.
        rng_key, simulate_rng = jax.random.split(rng_key)
        simulate_keys = jax.random.split(simulate_rng, width * batch_size)
        simulate_keys = simulate_keys.reshape(
            (width, batch_size) + simulate_keys.shape[1:])
        parent_kb, action_kb = simulate_forced(
            simulate_keys, tree, top_actions.T, max_depth)  # [K, B]
        parents = parent_kb.T                               # [B, K]
        actions = action_kb.T
        next_idxs = tree.children_index[batch_range[:, None], parents, actions]
        next_idxs = jnp.where(
            next_idxs == Tree.UNVISITED,
            sim_offset + jnp.arange(width, dtype=next_idxs.dtype)[None, :] + 1,
            next_idxs)                                      # [B, K]

        # --- ONE batched recurrent_fn call for the whole round: [B * width].
        parents_f = parents.reshape(-1)                     # b-major
        actions_f = actions.reshape(-1)
        next_f = next_idxs.reshape(-1)
        batch_f = jnp.repeat(batch_range, width)
        embedding_f = jax.tree_util.tree_map(
            lambda x: x[batch_f, parents_f], tree.embeddings)
        rng_key, expand_key = jax.random.split(rng_key)
        step, new_embedding_f = recurrent_fn(
            params, expand_key, actions_f, embedding_f)

        # --- write the round's K nodes with ONE scatter per tree array.
        tree = _write_nodes_batched(
            tree, batch_f, parents_f, actions_f, next_f,
            step.prior_logits, step.value, step.reward, step.discount,
            new_embedding_f)

        # --- back up all K lanes at once. A leaf expanded in round r is at
        # depth <= r+1 (each considered subtree gains at most one node per
        # round), so the lockstep walk needs at most that many hops.
        num_hops = min(round_index + 1, max_depth)
        tree = _backward_batched(tree, next_idxs, num_hops)
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
