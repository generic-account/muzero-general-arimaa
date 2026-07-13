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
        # NOTE: gumbel interior selection is deterministic (mctx deletes the
        # key), so no per-hop rng split is carried — the old per-hop
        # jax.random.split was a [K, B]-wide threefry chain per hop feeding a
        # deleted argument (pure dispatch waste, visible in traces as 8-byte
        # slice/threefry ops). `rng_key` stays in the signature so callers'
        # key-stream consumption (and thus all search outputs) is unchanged.
        def selection_fn(t, node_index, depth):
            interior = interior_fn(rng_key, t, node_index, depth)
            return jnp.where(depth == 0, forced_action, interior).astype(jnp.int32)

        def cond_fun(state):
            return state["is_continuing"]

        def body_fun(state):
            node_index = state["next_node_index"]
            action = selection_fn(tree, node_index, state["depth"])
            next_node_index = tree.children_index[node_index, action]
            depth = state["depth"] + 1
            return {
                "node_index": node_index,
                "action": action,
                "next_node_index": next_node_index,
                "depth": depth,
                "is_continuing": jnp.logical_and(
                    next_node_index != Tree.UNVISITED, depth < max_depth),
            }

        root_index = jnp.array(Tree.ROOT_INDEX, dtype=jnp.int32)
        state = {
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
# v3 "compact" search: no [B, N, A] children tables.
#
# Motivation (s2pilot round-2 trace, 2026-07-13): the full-width children
# tables ([B, 129, 1393] f32/s32 = 368MB each) were (a) physically transposed
# by XLA every halving round to reconcile the scatter-side layout with the
# descent while_loop's preferred layout (~34% of device time as anonymous
# reshape+copy pairs), and (b) gathered full-width by the interior action
# selection (~15%). v3 stores per-NODE compact state instead:
#
#   * child stats are DERIVED: children_values/visits of an edge are exact
#     mirrors of the child node's own node_values/node_visits (mctx's backward
#     maintains that invariant; we simply read it), and children_rewards/
#     discounts are per-child scalars (node_reward/node_discount).
#   * each node keeps its expanded-children list ([B, N, C] action/id/logit,
#     C = max children possible under the halving schedule) — replaces
#     children_index.
#   * interior Gumbel selection works on the candidate subset: among UNVISITED
#     actions the completed-Q is one shared constant (the mixed value), so
#     ordering = prior-logit ordering, and the argmax over 1393 actions equals
#     the argmax over {visited children} + {best unvisited candidate}. Each
#     node stores its top-(C+1) prior candidates + exact softmax pieces
#     (row max and sum-exp, captured at expansion) to reproduce mctx's math.
#
# Exactness contract: identical arithmetic per term; softmax normalizations
# regroup fp summation over the subset instead of the full row, so results
# are exact under a drift-free qtransform (value_scale=0) and drift only on
# fp near-ties under the default one — the same documented category as the
# v1/v2 round-regrouping drift. Tie-breaks reproduce argmax's lowest-index
# rule. The returned PolicyOutput tail is computed on a bitwise-identical
# root row (1-node mctx tree), so action/action_weights consumers are
# unaffected.
# ---------------------------------------------------------------------------
_UNVISITED = -1


def _cmax_from_schedule(rounds, m):
    """Max children any node can have on-schedule: the root has exactly the
    round-0 width (== m); an interior node gains at most one child per visit,
    and its visits are bounded by the schedule's max considered-visit + 1."""
    max_cv = max(cv for cv, _ in rounds)
    return max(m, max_cv + 1)


def _interior_select_compact(t, node, value_scale=0.1):
    """gumbel_muzero_interior_action_selection + qtransform_completed_by_mix_value
    over the node's compact children/candidates. Unbatched (runs under the
    [K, B] double vmap of the descent loop). `value_scale` mirrors the
    qtransform's (0.0 = the drift-free test transform)."""
    eps, maxvisit_init = 1e-8, 50.0
    va = t["child_actions"][node]                     # [C]
    vids = t["child_ids"][node]
    vlog = t["child_logits"][node]
    valid = va != _UNVISITED
    safe_ids = jnp.where(valid, vids, 0)
    vvis = jnp.where(valid, t["node_visits"][safe_ids], 0)
    vval = t["node_values"][safe_ids]
    q_c = t["node_reward"][safe_ids] + t["node_discount"][safe_ids] * vval

    raw = t["raw_values"][node]
    pm = t["prior_max"][node]
    s_all = t["prior_sumexp"][node]
    # softmax probs of the visited actions (exact: same exp/max/sum pieces
    # jax.nn.softmax uses, captured at expansion).
    vexp = jnp.where(valid, jnp.exp(vlog - pm), 0.0)
    vprob = jnp.maximum(jnp.finfo(jnp.float32).tiny, vexp / s_all)

    # _compute_mixed_value over the subset (visited slots all have visits>0).
    sum_visits = jnp.sum(vvis)
    vis_pos = valid & (vvis > 0)
    sum_probs = jnp.sum(jnp.where(vis_pos, vprob, 0.0))
    weighted_q = jnp.sum(jnp.where(
        vis_pos, vprob * q_c / jnp.where(vis_pos, sum_probs, 1.0), 0.0))
    mixed = (raw + sum_visits * weighted_q) / (sum_visits + 1)

    # completed + rescale: unvisited all complete to `mixed` and at least one
    # unvisited action always exists (1393 >> C), so full-row min/max reduce
    # to min/max over (visited q, mixed).
    mn = jnp.minimum(jnp.min(jnp.where(valid, q_c, jnp.inf)), mixed)
    mx = jnp.maximum(jnp.max(jnp.where(valid, q_c, -jnp.inf)), mixed)
    denom = jnp.maximum(mx - mn, eps)
    visit_scale = (maxvisit_init + jnp.max(vvis)) * value_scale
    cq_v = visit_scale * ((q_c - mn) / denom)
    cq_u = visit_scale * ((mixed - mn) / denom)

    # Best unvisited action = highest-logit candidate not in the child list
    # (candidate list is logit-descending; ties inherit top_k's index order,
    # matching the full-width argmax's lowest-index rule).
    ca = t["cand_actions"][node]                      # [Cc]
    cl = t["cand_logits"][node]
    taken = jnp.any(
        (ca[:, None] == va[None, :]) & valid[None, :], axis=1)
    first_free = jnp.argmax(~taken)                   # first (best) free slot
    a_bu = ca[first_free]
    l_bu = cl[first_free]

    # softmax(prior_logits + completed_q) over the full row, regrouped:
    # visited terms explicit; the unvisited mass is e^{cq_u} * (sum-exp of all
    # logits minus the visited ones).
    m_full = jnp.maximum(
        jnp.max(jnp.where(valid, vlog + cq_v, -jnp.inf)), l_bu + cq_u)
    z_vis = jnp.sum(jnp.where(valid, jnp.exp(vlog + cq_v - m_full), 0.0))
    s_unvis = jnp.maximum(s_all - jnp.sum(vexp), 0.0)
    z_unvis = jnp.exp(cq_u + pm - m_full) * s_unvis
    z = z_vis + z_unvis

    score_v = jnp.where(
        valid,
        jnp.exp(vlog + cq_v - m_full) / z
        - vvis.astype(jnp.float32) / (1.0 + sum_visits),
        -jnp.inf)
    score_u = jnp.exp(l_bu + cq_u - m_full) / z       # visits == 0
    scores = jnp.concatenate([score_v, score_u[None]])
    acts = jnp.concatenate([va, a_bu[None]])
    best = jnp.max(scores)
    # argmax tie-break: lowest action id among the tied maxima.
    return jnp.min(jnp.where(scores == best, acts, jnp.iinfo(jnp.int32).max)
                   ).astype(jnp.int32)


def _lookup_child(t, node, action):
    """children_index equivalent: the node id of `action` under `node`, or
    _UNVISITED. Unbatched."""
    va = t["child_actions"][node]
    eq = (va == action) & (va != _UNVISITED)
    slot = jnp.argmax(eq)
    return jnp.where(jnp.any(eq), t["child_ids"][node, slot],
                     jnp.int32(_UNVISITED))


def _make_compact_simulate(max_depth, value_scale=0.1):
    """[K, B] lockstep descent over the compact tree (mirrors
    _make_forced_simulate)."""

    @functools.partial(jax.vmap, in_axes=[0, None, 0], out_axes=0)  # K
    @functools.partial(jax.vmap, in_axes=[0, 0, 0], out_axes=0)     # B
    def simulate(rng_key, t, forced_action):
        del rng_key

        def body(state):
            node = state["next"]
            interior = _interior_select_compact(t, node, value_scale)
            action = jnp.where(state["depth"] == 0, forced_action,
                               interior).astype(jnp.int32)
            nxt = _lookup_child(t, node, action)
            depth = state["depth"] + 1
            return {"node": node, "action": action, "next": nxt,
                    "depth": depth,
                    "cont": jnp.logical_and(nxt != _UNVISITED,
                                            depth < max_depth)}

        state = {"node": jnp.int32(_UNVISITED), "action": jnp.int32(_UNVISITED),
                 "next": jnp.int32(Tree.ROOT_INDEX), "depth": jnp.int32(0),
                 "cont": jnp.array(True)}
        end = jax.lax.while_loop(lambda s: s["cont"], body, state)
        return end["node"], end["action"]

    return simulate


def _root_stats_at(t, batch_range, actions):
    """(q, visits) of root children at `actions` [B, m], derived from node
    stats (bitwise equal to the v2 children-table reads)."""
    b = batch_range[:, None]
    va = t["child_actions"][:, Tree.ROOT_INDEX]        # [B, C]
    vids = t["child_ids"][:, Tree.ROOT_INDEX]
    valid = va != _UNVISITED
    eq = (actions[:, :, None] == va[:, None, :]) & valid[:, None, :]
    found = jnp.any(eq, axis=-1)                       # [B, m]
    slot = jnp.argmax(eq, axis=-1)
    ids = jnp.take_along_axis(vids, slot, axis=1)      # [B, m]
    ids = jnp.where(found, ids, 0)
    vis = jnp.where(found, t["node_visits"][b, ids], 0)
    q = t["node_reward"][b, ids] + t["node_discount"][b, ids] \
        * t["node_values"][b, ids]
    return q, vis, found


def compact_gumbel_muzero_policy(
    params, rng_key, root, recurrent_fn, num_simulations,
    invalid_actions=None, max_depth=None, max_num_considered_actions=16,
    value_scale=0.1):
    """v3: batched sequential halving over the compact tree (see header)."""
    batch_size = root.value.shape[0]
    num_actions = root.prior_logits.shape[-1]
    if invalid_actions is None:
        invalid_actions = jnp.zeros_like(root.prior_logits)
    if max_depth is None:
        max_depth = num_simulations

    # Mirror v2: the masked logits ARE the root's logits from here on
    # (the output tail below must see the masked version, as v2's did).
    root = root.replace(
        prior_logits=_mask_invalid_actions(root.prior_logits, invalid_actions))
    root_logits_masked = root.prior_logits
    rng_key, gumbel_rng = jax.random.split(rng_key)
    gumbel = jax.random.gumbel(gumbel_rng, shape=root_logits_masked.shape,
                               dtype=root_logits_masked.dtype)

    rounds = _rounds_from_schedule(max_num_considered_actions, num_simulations)
    m = max_num_considered_actions
    c_max = _cmax_from_schedule(rounds, m)
    cc = c_max + 1
    n_nodes = num_simulations + 1
    B, N, C = batch_size, n_nodes, c_max

    def node_stats(logits):  # [R, A] -> (pm, sumexp, cand_actions, cand_logits)
        pm = jnp.max(logits, axis=-1)
        s = jnp.sum(jnp.exp(logits - pm[:, None]), axis=-1)
        cl, ca = jax.lax.top_k(logits, cc)
        return pm, s, ca.astype(jnp.int32), cl

    r_pm, r_s, r_ca, r_cl = node_stats(root_logits_masked)
    t = {
        "node_values": jnp.zeros((B, N), jnp.float32
                                 ).at[:, Tree.ROOT_INDEX].set(root.value),
        "raw_values": jnp.zeros((B, N), jnp.float32
                                ).at[:, Tree.ROOT_INDEX].set(root.value),
        "node_visits": jnp.zeros((B, N), jnp.int32
                                 ).at[:, Tree.ROOT_INDEX].set(1),
        "node_reward": jnp.zeros((B, N), jnp.float32),
        "node_discount": jnp.zeros((B, N), jnp.float32),
        "prior_max": jnp.zeros((B, N), jnp.float32
                               ).at[:, Tree.ROOT_INDEX].set(r_pm),
        "prior_sumexp": jnp.ones((B, N), jnp.float32
                                 ).at[:, Tree.ROOT_INDEX].set(r_s),
        "cand_actions": jnp.zeros((B, N, cc), jnp.int32
                                  ).at[:, Tree.ROOT_INDEX].set(r_ca),
        "cand_logits": jnp.zeros((B, N, cc), jnp.float32
                                 ).at[:, Tree.ROOT_INDEX].set(r_cl),
        "child_actions": jnp.full((B, N, C), _UNVISITED, jnp.int32),
        "child_ids": jnp.zeros((B, N, C), jnp.int32),
        "child_logits": jnp.zeros((B, N, C), jnp.float32),
        "child_count": jnp.zeros((B, N), jnp.int32),
        "parents": jnp.full((B, N), Tree.NO_PARENT, jnp.int32),
        "action_from_parent": jnp.full((B, N), Tree.NO_PARENT, jnp.int32),
        "embeddings": jnp.zeros((B, N) + root.embedding.shape[1:],
                                root.embedding.dtype
                                ).at[:, Tree.ROOT_INDEX].set(root.embedding),
    }

    batch_range = jnp.arange(B)
    logit_max_full = jnp.max(root_logits_masked, axis=-1)

    def _apply_fallback(top_actions):
        selected_invalid = jnp.take_along_axis(
            invalid_actions, top_actions, axis=1).astype(bool)
        return jnp.where(selected_invalid, top_actions[:, :1], top_actions)

    def score_root(considered_visit, considered, cons_prior, cons_gumbel):
        """Bit-identical _completed_q_and_score_subset over derived stats."""
        eps = 1e-8
        q, visits, _ = _root_stats_at(t, batch_range, considered)   # [B, m]
        raw_value = root.value                                      # [B]
        sum_visit_counts = jnp.sum(visits, axis=-1)
        prior = jnp.maximum(jnp.finfo(cons_prior.dtype).tiny, cons_prior)
        sum_probs = jnp.sum(prior, axis=-1)
        weighted_q = jnp.sum(
            prior * q / jnp.where(sum_probs[:, None] > 0,
                                  sum_probs[:, None], 1.0), axis=-1)
        mixed = (raw_value + sum_visit_counts * weighted_q) / (
            sum_visit_counts + 1)
        min_value = jnp.minimum(jnp.min(q, axis=-1), mixed)[:, None]
        max_value = jnp.maximum(jnp.max(q, axis=-1), mixed)[:, None]
        rescaled = (q - min_value) / jnp.maximum(max_value - min_value, eps)
        visit_scale = (50.0 + jnp.max(visits, axis=-1))[:, None]
        completed_q = visit_scale * value_scale * rescaled
        logits_norm = jnp.take_along_axis(root_logits_masked, considered,
                                          axis=1) - logit_max_full[:, None]
        penalty = jnp.where(visits == considered_visit, 0.0, -jnp.inf)
        return jnp.maximum(-1e9, cons_gumbel + logits_norm + completed_q) \
            + penalty

    prior_full = jax.nn.softmax(root_logits_masked, axis=-1)
    considered = considered_prior = gumbel_sub = None
    sim_offset = 0
    simulate = _make_compact_simulate(max_depth, value_scale)

    for round_index, (considered_visit, width) in enumerate(rounds):
        if round_index == 0:
            # Round 0: no root children yet -> completed_q is exactly 0
            # (all-equal completed values rescale to 0); scores = gumbel +
            # normalized logits, mctx's score_considered form.
            logits_norm = root_logits_masked - logit_max_full[:, None]
            scores = jnp.maximum(-1e9, gumbel + logits_norm)
            _, top_actions = jax.lax.top_k(scores, width)
            top_actions = _apply_fallback(top_actions)
            considered = jnp.sort(top_actions, axis=-1)
            considered_prior = jnp.take_along_axis(prior_full, considered,
                                                   axis=1)
            gumbel_sub = jnp.take_along_axis(gumbel, considered, axis=1)
        else:
            scores_sub = score_root(considered_visit, considered,
                                    considered_prior, gumbel_sub)
            scores = jnp.full((B, num_actions), -jnp.inf,
                              dtype=scores_sub.dtype)
            scores = scores.at[batch_range[:, None], considered].set(scores_sub)
            _, top_actions = jax.lax.top_k(scores, width)
            top_actions = _apply_fallback(top_actions)

        rng_key, simulate_rng = jax.random.split(rng_key)
        sim_keys = jax.random.split(simulate_rng, width * B)
        sim_keys = sim_keys.reshape((width, B) + sim_keys.shape[1:])
        parent_kb, action_kb = simulate(sim_keys, t, top_actions.T)
        parents = parent_kb.T                                  # [B, K]
        actions = action_kb.T

        # existing-child lookup (vectorized): (parent, action) -> id or new
        pa_va = t["child_actions"][batch_range[:, None], parents]   # [B,K,C]
        pa_ids = t["child_ids"][batch_range[:, None], parents]
        eq = (pa_va == actions[:, :, None]) & (pa_va != _UNVISITED)
        exists = jnp.any(eq, axis=-1)
        exist_slot = jnp.argmax(eq, axis=-1)
        exist_id = jnp.take_along_axis(pa_ids, exist_slot[..., None],
                                       axis=-1)[..., 0]
        new_ids = jnp.broadcast_to(
            sim_offset + jnp.arange(width, dtype=jnp.int32)[None, :] + 1,
            (B, width))
        # Off-schedule fallback rows can put the SAME (parent, action) in two
        # lanes of one round; those lanes share the first occurrence's node id
        # (their recurrent outputs are identical, and node_visits.add counts
        # each lane — preserving mctx's edge-visit sums).
        key_pa = parents * num_actions + actions               # [B, K]
        eq_pa = key_pa[:, :, None] == key_pa[:, None, :]       # [B, K, K]
        first_ix = jnp.argmax(eq_pa, axis=-1)                  # [B, K]
        new_ids = jnp.take_along_axis(new_ids, first_ix, axis=1)
        next_idxs = jnp.where(exists, exist_id, new_ids)       # [B, K]

        # --- one batched recurrent_fn call (packed-embedding gather as v2).
        parents_f = parents.reshape(-1)
        actions_f = actions.reshape(-1)
        next_f = next_idxs.reshape(-1)
        batch_f = jnp.repeat(batch_range, width)
        embedding_f = t["embeddings"][batch_f, parents_f]
        rng_key, expand_key = jax.random.split(rng_key)
        step, new_embedding_f = recurrent_fn(params, expand_key, actions_f,
                                             embedding_f)

        # --- node writes (all node-indexed at next_f; distinct on-schedule).
        pm_f, s_f, ca_f, cl_f = node_stats(step.prior_logits)
        # parent-side edge write: overwrite the existing slot on re-expansion
        # (mirrors mctx's children_index pointer overwrite), else append.
        # Lanes in one round can SHARE a parent (e.g. round 0: all K under the
        # root), so appended slots are count + the lane's rank among same-
        # parent new lanes this round (actions are distinct on-schedule).
        count_f = t["child_count"][batch_f, parents_f]
        lane = jnp.arange(width)
        is_first = first_ix == lane[None, :]                   # [B, K]
        new_first = (~exists) & is_first
        same_parent = parents[:, :, None] == parents[:, None, :]
        rank = jnp.sum(same_parent & new_first[:, None, :]
                       & (lane[None, None, :] < lane[None, :, None]),
                       axis=-1)                                # [B, K]
        slot_new = jnp.minimum(count_f.reshape(B, width) + rank, C - 1)
        # duplicates inherit the first occurrence's slot; existing edges
        # overwrite in place (mctx pointer-overwrite semantics).
        slot_new = jnp.take_along_axis(slot_new, first_ix, axis=1)
        new_f = ~exists.reshape(-1)
        slot_f = jnp.where(new_f, slot_new.reshape(-1), exist_slot.reshape(-1))
        # child's logit under the parent: from the parent's stored rows —
        # cand list first, else the (re-expanded) existing slot's logit.
        cand_pa = t["cand_actions"][batch_f, parents_f]         # [F, Cc]
        cand_pl = t["cand_logits"][batch_f, parents_f]
        in_cand = cand_pa == actions_f[:, None]
        cand_hit = jnp.any(in_cand, axis=-1)
        cand_slot = jnp.argmax(in_cand, axis=-1)
        logit_from_cand = jnp.take_along_axis(
            cand_pl, cand_slot[:, None], axis=-1)[:, 0]
        old_logit = t["child_logits"][batch_f, parents_f, slot_f]
        # root round-0 actions may fall outside the parent's top-(C+1) cand
        # list; fetch those from the full root logits row (root only).
        root_logit = root_logits_masked[batch_f, actions_f]
        fallback_logit = jnp.where(parents_f == Tree.ROOT_INDEX, root_logit,
                                   old_logit)
        edge_logit = jnp.where(cand_hit, logit_from_cand, fallback_logit)

        t = dict(
            t,
            node_values=t["node_values"].at[batch_f, next_f].set(step.value),
            raw_values=t["raw_values"].at[batch_f, next_f].set(step.value),
            node_visits=t["node_visits"].at[batch_f, next_f].add(1),
            node_reward=t["node_reward"].at[batch_f, next_f].set(step.reward),
            node_discount=t["node_discount"].at[batch_f, next_f].set(
                step.discount),
            prior_max=t["prior_max"].at[batch_f, next_f].set(pm_f),
            prior_sumexp=t["prior_sumexp"].at[batch_f, next_f].set(s_f),
            cand_actions=t["cand_actions"].at[batch_f, next_f].set(ca_f),
            cand_logits=t["cand_logits"].at[batch_f, next_f].set(cl_f),
            child_actions=t["child_actions"].at[
                batch_f, parents_f, slot_f].set(actions_f),
            child_ids=t["child_ids"].at[batch_f, parents_f, slot_f].set(next_f),
            child_logits=t["child_logits"].at[
                batch_f, parents_f, slot_f].set(edge_logit),
            child_count=t["child_count"].at[batch_f, parents_f].add(
                jnp.where(new_first.reshape(-1), 1, 0)),
            parents=t["parents"].at[batch_f, next_f].set(parents_f),
            action_from_parent=t["action_from_parent"].at[
                batch_f, next_f].set(actions_f),
            embeddings=t["embeddings"].at[batch_f, next_f].set(new_embedding_f),
        )

        # --- backward: node stats only (children_* are derived views).
        num_hops = min(round_index + 1, max_depth)
        t = _backward_compact(t, next_idxs, num_hops)
        sim_offset += width

    # --- outputs: verbatim v2 tail on a 1-node mctx tree whose root row is
    # scattered back from the compact stats (bitwise-identical values).
    va = t["child_actions"][:, Tree.ROOT_INDEX]                # [B, C]
    vids = t["child_ids"][:, Tree.ROOT_INDEX]
    valid = va != _UNVISITED
    safe_a = jnp.where(valid, va, num_actions)                 # OOB -> drop
    ids0 = jnp.where(valid, vids, 0)
    bidx = batch_range[:, None]
    full_visits = jnp.zeros((B, num_actions), jnp.int32).at[
        bidx, safe_a].set(jnp.where(valid, t["node_visits"][bidx, ids0], 0),
                          mode="drop")
    full_values = jnp.zeros((B, num_actions), jnp.float32).at[
        bidx, safe_a].set(t["node_values"][bidx, ids0], mode="drop")
    full_rewards = jnp.zeros((B, num_actions), jnp.float32).at[
        bidx, safe_a].set(t["node_reward"][bidx, ids0], mode="drop")
    full_discounts = jnp.zeros((B, num_actions), jnp.float32).at[
        bidx, safe_a].set(t["node_discount"][bidx, ids0], mode="drop")

    extra_data = mctx_action_selection.GumbelMuZeroExtraData(root_gumbel=gumbel)
    mini = Tree(
        node_visits=t["node_visits"][:, :1],
        raw_values=t["raw_values"][:, :1],
        node_values=t["node_values"][:, :1],
        parents=jnp.full((B, 1), Tree.NO_PARENT, jnp.int32),
        action_from_parent=jnp.full((B, 1), Tree.NO_PARENT, jnp.int32),
        children_index=jnp.full((B, 1, num_actions), _UNVISITED, jnp.int32),
        children_prior_logits=root_logits_masked[:, None, :],
        children_visits=full_visits[:, None, :],
        children_rewards=full_rewards[:, None, :],
        children_discounts=full_discounts[:, None, :],
        children_values=full_values[:, None, :],
        embeddings=jax.tree_util.tree_map(lambda x: x[:, :1], t["embeddings"]),
        root_invalid_actions=invalid_actions,
        extra_data=extra_data,
    )
    summary = mini.summary()
    considered_visit = jnp.max(summary.visit_counts, axis=-1, keepdims=True)
    completed_qvalues = jax.vmap(
        functools.partial(mctx_qtransforms.qtransform_completed_by_mix_value,
                          value_scale=value_scale),
        in_axes=[0, None])(mini, Tree.ROOT_INDEX)
    to_argmax = mctx_seq_halving.score_considered(
        considered_visit, gumbel, root.prior_logits, completed_qvalues,
        summary.visit_counts)
    action = mctx_action_selection.masked_argmax(to_argmax, invalid_actions)
    completed_search_logits = _mask_invalid_actions(
        root.prior_logits + completed_qvalues, invalid_actions)
    action_weights = jax.nn.softmax(completed_search_logits)
    return mctx_base.PolicyOutput(
        action=action, action_weights=action_weights, search_tree=mini)


def _backward_compact(t, leaf_indices, num_hops):
    """_backward_batched without the children-table writes (derived views).
    Node math identical (bitwise)."""
    batch_size, num_lanes = leaf_indices.shape
    num_nodes = t["node_values"].shape[1]
    b_idx = jnp.arange(batch_size)[:, None]
    node_values = t["node_values"]
    node_visits = t["node_visits"]

    def hop(carry, _):
        index, leaf_value, active = carry
        parent = t["parents"][b_idx, index]
        reward = t["node_reward"][b_idx, index]
        discount = t["node_discount"][b_idx, index]
        leaf_value = reward + discount * leaf_value
        is_root = parent == Tree.ROOT_INDEX
        record = (parent, leaf_value, active)
        carry = (parent, leaf_value, jnp.logical_and(active, ~is_root))
        return carry, record

    init = (leaf_indices, node_values[b_idx, leaf_indices],
            jnp.ones(leaf_indices.shape, dtype=bool))
    _, (parent_h, leaf_h, active_h) = jax.lax.scan(
        hop, init, None, length=num_hops)

    mask = active_h.reshape(-1)
    par = jnp.where(mask, parent_h.reshape(-1), num_nodes)
    bat = jnp.broadcast_to(b_idx[None, :, :], active_h.shape).reshape(-1)
    leaf_v = leaf_h.reshape(-1)
    one = jnp.ones((), dtype=node_visits.dtype)

    cnt = jnp.zeros_like(node_visits).at[bat, par].add(one, mode="drop")
    leaf_sum = jnp.zeros_like(node_values).at[bat, par].add(
        leaf_v, mode="drop")
    new_node_values = jnp.where(
        cnt > 0,
        (node_values * node_visits + leaf_sum) / (node_visits + cnt),
        node_values)
    return dict(t, node_values=new_node_values,
                node_visits=node_visits + cnt)


# ---------------------------------------------------------------------------
# jaxarimaa wrapper: identical signature to search.run_search (drop-in).
# ---------------------------------------------------------------------------
@functools.partial(jax.jit, static_argnums=(0, 4, 5, 6, 7))
def run_search(model, params, rng_key, states, num_simulations,
               max_num_considered_actions, features=None, prior_temp=1.0):
    from . import search as slow_search

    prior_logits, value, legal = slow_search._eval(model, params, states, features)
    if prior_temp != 1.0:
        # Anti-self-sharpening (optima/KataGo): flatten the net's priors fed to
        # search so exploration survives the policy's own sharpening feedback.
        prior_logits = prior_logits / prior_temp
    root = mctx.RootFnOutput(prior_logits=prior_logits, value=value,
                             embedding=pack_states(states))
    inner_fn = slow_search.make_recurrent_fn(model, features, prior_temp)

    def recurrent_fn(params_, key, actions, packed):
        out, nstates = inner_fn(params_, key, actions, unpack_states(packed))
        return out, pack_states(nstates)

    policy = (compact_gumbel_muzero_policy
              if (features is not None and features.compact_search)
              else batched_gumbel_muzero_policy)
    return policy(
        params=params,
        rng_key=rng_key,
        root=root,
        recurrent_fn=recurrent_fn,
        num_simulations=num_simulations,
        invalid_actions=~legal,
        max_num_considered_actions=max_num_considered_actions,
    )
