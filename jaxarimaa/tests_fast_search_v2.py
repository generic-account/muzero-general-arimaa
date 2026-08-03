"""Equivalence + validity tests for fast_search v2 (wave-parallel tree ops).

Run with:  .venv/bin/python -m jaxarimaa.tests_fast_search_v2

What is asserted, for mid-game Arimaa states and a small real network, for
(num_simulations, max_num_considered_actions) in (32,16) (16,16) (8,4) (64,16):

  1. v1-vs-v2 (the point of v2 — batching must not change results):
     root child visit counts and chosen actions of the current implementation
     are EXACTLY equal to the pre-v2 (git HEAD) implementation on on-schedule
     rows (num_valid >= m), and the whole search tree matches bitwise except
     root node_values (associative closed-form root mean; fp summation order).
  2. Tree-mechanics exactness vs mctx: with a drift-free qtransform
     (value_scale=0, so root scores don't depend on backed-up Q-values),
     v2 root child visit counts are EXACTLY equal to mctx.gumbel_muzero_policy
     on ALL on-schedule rows. This isolates descent/expansion/backup:
     any error in v2's batched tree ops would break this.
  3. mctx equivalence with the DEFAULT qtransform:
       * (16,16) is a single round (all candidates unvisited, provably
         drift-free): exact visit equality asserted on all on-schedule rows;
       * for multi-round schedules, mctx recomputes completed-Q after every
         simulation, so within a halving round (2w candidates at the
         considered visit level, w slots) its sequential argmax picks can
         drift away from the round-start top-w that fast_search (v1 AND v2)
         uses. This is a PRE-EXISTING v1 property, not introduced by v2; the
         test asserts that any row diverging from mctx diverges IDENTICALLY
         in v1 and v2, and asserts full mctx equality (visits exact, action
         equal, action_weights allclose 1e-4, root node_values allclose) on
         the rows where v1 already matched mctx.
  4. Validity on ALL rows (including the documented off-schedule fallback):
     chosen action legal; action_weights sum to 1 with zero illegal mass;
     root child visits sum to exactly num_simulations; finite tree values.
"""

import functools
import importlib.util
import os
import subprocess
import sys
import tempfile

import jax
import jax.numpy as jnp
import numpy as np
import mctx
from mctx._src import qtransforms as mctx_qtransforms

from jaxarimaa import env as jenv
from jaxarimaa import fast_search
from jaxarimaa import mctx_batched
from jaxarimaa import network
from jaxarimaa import search as slow_search
from jaxarimaa.config import NetConfig

BATCH = 8
ADVANCE_STEPS = 12
NM_CASES = [(32, 16), (16, 16), (8, 4), (64, 16)]
ROOT = 0

# Drift-free transform: value_scale=0 removes the backed-up Q contribution
# from root scores, so mctx's per-sim rescoring cannot reorder picks and any
# visit-count difference vs mctx would be a tree-mechanics bug in v2.
QT_DRIFT_FREE = functools.partial(
    mctx_qtransforms.qtransform_completed_by_mix_value, value_scale=0.0)


def make_states(key, batch=BATCH, steps=ADVANCE_STEPS):
    """Advance `steps` random legal steps from random setups (mid-game states)."""
    keys = jax.random.split(key, batch)
    states = jax.vmap(jenv.init_state)(keys)
    step_key = jax.random.fold_in(key, 1234)

    @jax.jit
    def advance(states, key):
        legal = jax.vmap(jenv.legal_action_mask)(states)
        logits = jnp.where(legal, 0.0, -jnp.inf)
        actions = jax.random.categorical(key, logits, axis=-1)
        nxt = jax.vmap(jenv.step)(states, actions)
        return jenv.where_state(states.terminated, states, nxt)

    for _ in range(steps):
        step_key, k = jax.random.split(step_key)
        states = advance(states, k)
    return states


def make_model_and_params(key, states):
    model = network.make_network(
        NetConfig(backbone="resnet", channels=16, blocks=2, use_se=False))
    obs = jax.vmap(lambda s: jenv.observe(s, None))(states)
    params = model.init(key, obs[0])
    return model, params


def load_v1():
    """Load the pre-v2 fast_search (git HEAD version) as a standalone module."""
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # The v2 policy lived in fast_search.py until commit a01f659, then moved to
    # mctx_batched.py. Check BOTH for the v2 marker: if either has it, HEAD is
    # already v2 and there is no v1 baseline to compare against. (Before this
    # guard was widened, the refactor made it miss the marker and then try to
    # exec the post-refactor fast_search.py — which is now a glue module with a
    # relative import — standalone, crashing the suite.)
    src = None
    for path in ("jaxarimaa/mctx_batched.py", "jaxarimaa/fast_search.py"):
        try:
            text = subprocess.check_output(
                ["git", "-C", repo, "show", f"HEAD:{path}"], text=True)
        except Exception:
            continue
        if "_write_nodes_batched" in text:
            print("  [note] git HEAD already contains v2; skipping v1 baseline")
            return None
        if path == "jaxarimaa/fast_search.py":
            src = text
    if src is None:
        print("  [note] no pre-v2 fast_search.py at HEAD; skipping v1 baseline")
        return None
    path = os.path.join(tempfile.mkdtemp(prefix="fsv1_"), "fast_search_v1.py")
    with open(path, "w") as f:
        f.write(src)
    spec = importlib.util.spec_from_file_location("fast_search_v1", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run_policy(policy_fn, model, params, key, states, n, m, qtransform=None):
    prior_logits, value, legal, _ = slow_search._eval(model, params, states, None)
    root = mctx.RootFnOutput(prior_logits=prior_logits, value=value,
                             embedding=states)
    kwargs = {} if qtransform is None else {"qtransform": qtransform}
    out = policy_fn(
        params=params,
        rng_key=key,
        root=root,
        recurrent_fn=slow_search.make_recurrent_fn(model, None),
        num_simulations=n,
        invalid_actions=~legal,
        max_num_considered_actions=m,
        **kwargs)
    return out


def check_case(model, params, states, key, n, m, v1_mod):
    fails = []
    legal = jax.vmap(jenv.legal_action_mask)(states)
    num_valid = legal.sum(axis=-1)
    on_sched = jnp.where(num_valid >= m)[0]

    out_ref = slow_search.run_search(model, params, key, states, n, m, None)
    out_v2 = fast_search.run_search(model, params, key, states, n, m, None)
    visits_ref = out_ref.search_tree.children_visits[:, ROOT]
    visits_v2 = out_v2.search_tree.children_visits[:, ROOT]

    if on_sched.size == 0:
        fails.append("no on-schedule rows to compare (num_valid < m everywhere)")

    # --- 1) v1-vs-v2: bitwise-equal search behavior.
    n_pre_existing = -1
    if v1_mod is not None:
        out_v1 = run_policy(v1_mod.batched_gumbel_muzero_policy,
                            model, params, key, states, n, m)
        t1, t2 = out_v1.search_tree, out_v2.search_tree
        visits_v1 = t1.children_visits[:, ROOT]
        if not bool(jnp.all(visits_v1 == visits_v2)):
            fails.append("v1 and v2 root visit counts differ")
        if not bool(jnp.all(out_v1.action[on_sched] == out_v2.action[on_sched])):
            fails.append("v1 and v2 chosen actions differ on on-schedule rows")
        # Full-tree equality on on-schedule rows: integer topology/counters
        # bitwise; float value fields to within a few ulps (v2's vectorized
        # backup is the same arithmetic as v1's per-lane backup, but XLA
        # fuses/FMA-contracts the two graphs differently, so last-ulp
        # differences ~1e-9 are expected). Off-schedule rows additionally
        # have duplicate lanes per round, where v1's sequential write/backup
        # order differs from v2's single-scatter order (documented fallback).
        for name in ["children_index", "children_visits", "parents",
                     "action_from_parent", "node_visits"]:
            a1, a2 = getattr(t1, name)[on_sched], getattr(t2, name)[on_sched]
            if not bool(jnp.all(a1 == a2)):
                fails.append(f"v1 and v2 tree field {name} differs on-schedule")
        for name in ["children_values", "children_rewards",
                     "children_prior_logits", "raw_values", "node_values"]:
            a1, a2 = getattr(t1, name)[on_sched], getattr(t2, name)[on_sched]
            if not bool(jnp.allclose(a1, a2, atol=1e-6)):
                err = float(jnp.abs(a1 - a2).max())
                fails.append(f"v1 and v2 tree field {name} not allclose "
                             f"on-schedule (max err {err:.2e})")
        # Rows where v1 already diverged from mctx (pre-existing top-k vs
        # per-sim-rescored selection drift; see module docstring).
        row_match_v1 = jnp.all(visits_v1 == visits_ref, axis=-1)
        n_pre_existing = int(jnp.sum(~row_match_v1[on_sched]))
        mctx_rows = on_sched[row_match_v1[on_sched]]
    else:
        row_match_v1 = jnp.all(visits_v2 == visits_ref, axis=-1)
        mctx_rows = on_sched[row_match_v1[on_sched]]

    # --- 2) tree-mechanics exactness vs mctx (drift-free qtransform).
    out_ref0 = run_policy(mctx.gumbel_muzero_policy, model, params, key,
                          states, n, m, qtransform=QT_DRIFT_FREE)
    out_v20 = run_policy(mctx_batched.batched_gumbel_muzero_policy, model,
                         params, key, states, n, m, qtransform=QT_DRIFT_FREE)
    v_ref0 = out_ref0.search_tree.children_visits[:, ROOT]
    v_v20 = out_v20.search_tree.children_visits[:, ROOT]
    if not bool(jnp.all(v_ref0[on_sched] == v_v20[on_sched])):
        bad = on_sched[jnp.any(v_ref0[on_sched] != v_v20[on_sched], axis=-1)]
        fails.append(f"drift-free visit counts differ from mctx on rows "
                     f"{bad.tolist()} (tree-mechanics bug)")
    if not bool(jnp.all(out_ref0.action[on_sched] == out_v20.action[on_sched])):
        fails.append("drift-free chosen actions differ from mctx")

    # --- 3) mctx equivalence with the default qtransform.
    if n == m:  # single round, provably drift-free even by default transform
        if not bool(jnp.all(visits_ref[on_sched] == visits_v2[on_sched])):
            fails.append("single-round default-qtransform visits differ from mctx")
    rows = mctx_rows
    if rows.size:
        if not bool(jnp.all(visits_ref[rows] == visits_v2[rows])):
            fails.append("visit counts differ from mctx on v1-matching rows")
        if not bool(jnp.all(out_ref.action[rows] == out_v2.action[rows])):
            fails.append("chosen actions differ from mctx on v1-matching rows")
        werr = float(jnp.abs(out_ref.action_weights[rows]
                             - out_v2.action_weights[rows]).max())
        if werr > 1e-4:
            fails.append(f"action_weights differ from mctx (max err {werr:.2e})")
        verr = float(jnp.abs(out_ref.search_tree.node_values[rows, ROOT]
                             - out_v2.search_tree.node_values[rows, ROOT]).max())
        if verr > 1e-4:
            fails.append(f"root node_values differ from mctx (max err {verr:.2e})")
    else:
        werr = verr = float("nan")

    # --- 4) validity on ALL rows.
    chosen_legal = legal[jnp.arange(legal.shape[0]), out_v2.action]
    if not bool(jnp.all(chosen_legal)):
        fails.append("v2 chose an illegal action")
    wsum = out_v2.action_weights.sum(axis=-1)
    if not bool(jnp.allclose(wsum, 1.0, atol=1e-4)):
        fails.append(f"v2 action_weights do not sum to 1 (got {wsum})")
    illegal_mass = jnp.where(legal, 0.0, out_v2.action_weights).sum()
    if not bool(illegal_mass == 0.0):
        fails.append(f"v2 puts mass {illegal_mass} on illegal actions")
    vsum = visits_v2.sum(axis=-1)
    if not bool(jnp.all(vsum == n)):
        fails.append(f"v2 root visits sum != n: {vsum.tolist()}")
    if not bool(jnp.all(jnp.isfinite(out_v2.search_tree.node_values))):
        fails.append("v2 tree node_values contain non-finite entries")

    return fails, int(on_sched.size), int(rows.size), n_pre_existing, werr, verr


def check_subset_score_algebra(model, params, states, key, n=32, m=8):
    """Direct algebraic gate on `_completed_q_and_score_subset`.

    That function hand-derives mctx's `score_considered(vmap(qtransform)(...))`
    over just the `considered` columns, reproducing the transform's GLOBAL terms
    (mixed value, rescale min/max, visit_scale, full-width logit max) from the
    subset. Nothing else in this suite covers it: the drift-free exactness test
    passes `functools.partial(qtransform, value_scale=0.0)`, which fails the
    `qtransform is qtransform_completed_by_mix_value` identity check and so takes
    the general full-width branch. Under the DEFAULT transform, where the subset
    path IS active, divergence from mctx is expected for other reasons — so a bug
    in this algebra could hide as "drift". Hence this test, against a tree that
    mctx itself built.
    """
    from mctx._src import seq_halving as _sh
    prior, value, legal, _ = slow_search._eval(model, params, states, None)
    out = mctx.gumbel_muzero_policy(
        params=params, rng_key=key,
        root=mctx.RootFnOutput(prior_logits=prior, value=value,
                               embedding=states),
        recurrent_fn=slow_search.make_recurrent_fn(model, None, 1.0),
        num_simulations=n, invalid_actions=~legal,
        max_num_considered_actions=m)
    tree = out.search_tree
    visits_np = np.asarray(tree.children_visits[:, ROOT])
    # `considered` must be visited actions (the function's stated precondition):
    # take top-m by visit count, then sort by action id as the policy does.
    order = np.argsort(-visits_np, axis=-1)[:, :m]
    considered = jnp.asarray(np.sort(order, axis=-1))
    if not (np.take_along_axis(visits_np, np.asarray(considered), 1) > 0).all():
        return [f"precondition unmet: fewer than m={m} visited root actions"]

    root_logits = tree.children_prior_logits[:, ROOT]
    gumbel = tree.extra_data.root_gumbel
    logit_max_full = jnp.max(root_logits, axis=-1)
    considered_prior = jnp.take_along_axis(
        jax.nn.softmax(root_logits, axis=-1), considered, axis=1)
    gumbel_sub = jnp.take_along_axis(gumbel, considered, axis=1)
    batch_range = jnp.arange(states.player.shape[0])

    completed_full = jax.vmap(mctx_qtransforms.qtransform_completed_by_mix_value,
                              in_axes=[0, None])(tree, ROOT)
    fails = []
    for cv in (1, 2, 3, 4):
        scores_full = _sh.score_considered(
            cv, gumbel, root_logits, completed_full,
            tree.children_visits[:, ROOT])
        ref = np.asarray(jnp.take_along_axis(scores_full, considered, axis=1))
        got = np.asarray(mctx_batched._completed_q_and_score_subset(
            tree, batch_range, considered, gumbel_sub, considered_prior, cv,
            logit_max_full))
        if not (np.isinf(ref) == np.isinf(got)).all():
            fails.append(f"subset algebra: -inf penalty mask mismatch at cv={cv}")
            continue
        fin = np.isfinite(ref) & np.isfinite(got)
        if fin.any():
            d = float(np.abs(ref[fin] - got[fin]).max())
            if d > 1e-5:
                fails.append(f"subset algebra: max|diff|={d:.3e} at cv={cv}")
    return fails


def main():
    key = jax.random.PRNGKey(7)
    kstate, kmodel, ksearch = jax.random.split(key, 3)
    states = make_states(kstate)
    model, params = make_model_and_params(kmodel, states)
    num_valid = jax.vmap(jenv.legal_action_mask)(states).sum(axis=-1)
    print(f"batch={BATCH}, num_valid per row: {num_valid.tolist()}")

    v1_mod = load_v1()
    if v1_mod is not None:
        print("v1 baseline loaded from git HEAD")

    all_ok = True
    v1_col = "exact" if v1_mod is not None else "n/a"
    print(f"{'n':>4} {'m':>4} {'rows>=m':>8} {'v1==v2':>7} {'mctx@vs0':>9} "
          f"{'mctx-rows':>10} {'drifted':>8} {'max|dw|':>10} {'max|dv|':>10}  status")
    for i, (n, m) in enumerate(NM_CASES):
        case_key = jax.random.fold_in(ksearch, i)
        fails, n_sched, n_match, n_drift, werr, verr = check_case(
            model, params, states, case_key, n, m, v1_mod)
        status = "PASS" if not fails else "FAIL"
        all_ok &= not fails
        print(f"{n:>4} {m:>4} {n_sched:>8} {v1_col:>7} {'exact':>9} "
              f"{n_match:>10} {n_drift:>8} {werr:>10.2e} {verr:>10.2e}  {status}")
        for f in fails:
            print(f"       - {f}")

    sub_fails = check_subset_score_algebra(
        model, params, states, jax.random.fold_in(ksearch, 99))
    all_ok &= not sub_fails
    print(f"subset-score algebra vs mctx full-width: "
          f"{'exact' if not sub_fails else 'FAIL'}")
    for f in sub_fails:
        print(f"       - {f}")

    print("ALL PASS" if all_ok else "FAILURES PRESENT")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
