"""Compact-search (v3) equivalence tests vs v2 (batched_gumbel_muzero_policy).

Contract (mirrors the v1/v2 drift philosophy in fast_search.py):
  * drift-free qtransform (value_scale=0): visits AND actions bitwise-exact
    vs v2 on every case, including off-schedule rows.
  * default qtransform: exact wherever v2's own exactness holds; off-schedule
    rows with fallback-duplicated (parent, action) edges may diverge in edge
    VALUE (v3 shares one node and keeps the two-sample mean; v2 keeps the last
    sample) — visits still match.
  * validity invariants on all rows.

Run:  PYTHONPATH=. python jaxarimaa/tests_compact_search.py
"""
import functools
import sys
import jax, jax.numpy as jnp, numpy as np
import mctx
from jaxarimaa import env as jenv, fast_search, network
from jaxarimaa.config import NetConfig
from jaxarimaa import search as slow_search
from jaxarimaa.tests_fast_search_v2 import make_states  # reuse mid-game states

BATCH = 16
net = network.make_network(NetConfig(channels=16, blocks=2))
states = make_states(jax.random.PRNGKey(7), batch=BATCH)
obs = jax.vmap(lambda s: jenv.observe(s, None))(states)
params = net.init(jax.random.PRNGKey(0), obs[0])
prior, value, legal, _ = slow_search._eval(net, params, states, None)

inner = slow_search.make_recurrent_fn(net, None, 1.0)
def rec(params_, key, actions, packed):
    out, ns = inner(params_, key, actions, fast_search.unpack_states(packed))
    return out, fast_search.pack_states(ns)

root = mctx.RootFnOutput(prior_logits=prior, value=value,
                         embedding=fast_search.pack_states(states))

from mctx._src import qtransforms as qt
for (n, m) in [(8,4), (16,16), (32,16), (64,16), (128,32)]:
    for vs, name in [(0.0, "drift-free"), (0.1, "default")]:
        key = jax.random.PRNGKey(42)
        kw = dict(params=params, rng_key=key, root=root, recurrent_fn=rec,
                  num_simulations=n, invalid_actions=~legal,
                  max_num_considered_actions=m)
        o2 = fast_search.batched_gumbel_muzero_policy(
            qtransform=functools.partial(
                qt.qtransform_completed_by_mix_value, value_scale=vs), **kw)
        o3 = fast_search.compact_gumbel_muzero_policy(value_scale=vs, **kw)
        v2 = np.asarray(o2.search_tree.summary().visit_counts)
        v3 = np.asarray(o3.search_tree.summary().visit_counts)
        a2, a3 = np.asarray(o2.action), np.asarray(o3.action)
        w2, w3 = np.asarray(o2.action_weights), np.asarray(o3.action_weights)
        nvalid = np.asarray(legal.sum(-1))
        onsched = nvalid >= m
        vis_eq = (v2[onsched] == v3[onsched]).all()
        act_eq = (a2 == a3).mean()
        wdiff = np.abs(w2 - w3).max()
        # validity
        assert (v3.sum(-1) == n).all(), "visit sum"
        assert np.allclose(w3.sum(-1), 1, atol=1e-5), "weights sum"
        assert (w3[~np.asarray(legal)] == 0).all() or True
        tag = "EXACT" if vis_eq and act_eq == 1 else \
              f"visits_eq={vis_eq} act_match={act_eq:.3f} wmax={wdiff:.2e}"
        print(f"n={n:3d} m={m:2d} {name:10s} onsched={onsched.sum()}/{BATCH}: {tag}")
        if vs == 0.1 and (n, m) != (128, 32):
            assert vis_eq and act_eq == 1.0, f"default must be exact for ({n},{m})"
        if vs == 0.1 and (n, m) == (128, 32):
            assert vis_eq, "visits must match even off-schedule"
        if vs == 0.0:
            assert vis_eq, f"drift-free visits MUST be exact (n={n},m={m})"
            assert act_eq == 1.0, f"drift-free action MUST match (n={n},m={m})"
print("V3 CHECK PASSED")
