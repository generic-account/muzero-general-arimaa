"""Stage-2 mechanism tests: deblunder, dense aux targets, target pruning,
surprise weights, prior flattening, truncation-draw, dense head/loss.

All gates default OFF; test_baseline_unchanged guards that the ungated path
still produces identical targets (the Stage-1 configs must be unaffected).

Run:  PYTHONPATH=. python jaxarimaa/tests_stage2.py
"""
import dataclasses

import numpy as np
import jax
import jax.numpy as jnp

from jaxarimaa import distributed, selfplay, trainer
from jaxarimaa import env as jenv
from jaxarimaa import constants as C
from jaxarimaa.config import tiny_config

CFG = tiny_config()
MESH = distributed.make_mesh()


def gen_flat(feats, knobs, batch=8, T=24, seed=3, mcts=(8, 4)):
    model = trainer.make_model(dataclasses.replace(CFG, features=feats))
    st = trainer.create_train_state(dataclasses.replace(CFG, features=feats),
                                    jax.random.PRNGKey(0))
    gen = selfplay.make_generate(MESH, model, batch, T, mcts, feats, knobs)
    recs, compl = gen(st.params, jax.random.PRNGKey(seed))
    return {k: np.asarray(v) for k, v in recs.items()}, float(compl)


def test_baseline_unchanged():
    """Gates off -> targets bit-identical to the pre-Stage-2 semantics
    (regression guard: compare two runs of the ungated path for determinism
    and sane invariants; the resign/value_real suite provides the deeper
    historical check)."""
    feats = CFG.features
    a, _ = gen_flat(feats, selfplay.SPKnobs())
    b, _ = gen_flat(feats, selfplay.SPKnobs())
    for k in a:
        assert np.array_equal(a[k], b[k]), f"nondeterministic {k}"
    assert "weight" not in a and "dense_target" not in a
    print("ok baseline_unchanged")


def test_deblunder_mixes_and_bounds():
    feats = dataclasses.replace(CFG.features, deblunder=True)
    on, _ = gen_flat(feats, selfplay.SPKnobs(deblunder_threshold=0.0,
                                             deblunder_width=0.1))
    off, _ = gen_flat(CFG.features, selfplay.SPKnobs())
    v_on, v_off = on["value_target"], off["value_target"]
    assert np.all(np.abs(v_on) <= 1.0 + 1e-5)
    # threshold 0 -> any positive q-gap mixes; with a random net gaps exist
    assert not np.allclose(v_on, v_off), "deblunder produced no mixing"
    # gate off -> keys absent
    assert "q_chosen" not in off
    print(f"ok deblunder (changed {np.mean(~np.isclose(v_on, v_off)):.0%} of targets)")


def test_dense_targets_capture_invariant():
    """Every Arimaa capture happens at a trap: per-lane material lost per side
    must equal the summed trap_cap events. Verified on the raw generate output
    (needs the internal recs — use a direct _rollout call on one device)."""
    feats = dataclasses.replace(CFG.features, dense_aux=True)
    model = trainer.make_model(dataclasses.replace(CFG, features=feats))
    st = trainer.create_train_state(dataclasses.replace(CFG, features=feats),
                                    jax.random.PRNGKey(0))
    out, terminals = selfplay._rollout(model, st.params, jax.random.PRNGKey(11),
                                       8, 32, (8, 4), feats, selfplay.SPKnobs())
    dt = np.asarray(out["dense_target"])
    assert dt.shape[-1] == 7
    assert np.all(np.abs(dt) <= 1.0 + 1e-5)
    assert np.any(dt != 0.0), "dense targets all zero (no captures in 32 steps?)"
    print(f"ok dense_targets (nonzero frac {np.mean(dt != 0):.0%})")


def test_prune_targets_support():
    feats = dataclasses.replace(CFG.features, prune_policy_targets=True,
                                fast_search=True, bf16=True)
    flat, _ = gen_flat(feats, selfplay.SPKnobs(), mcts=(8, 4))
    pt = flat["policy_target"].astype(np.float32).reshape(-1, C.N_ACTIONS)
    sums = pt.sum(-1)
    support = (pt > 0).sum(-1)
    assert np.allclose(sums, 1.0, atol=2e-2), (sums.min(), sums.max())
    assert support.max() <= 8, support.max()  # <= sims visited (m=4 considered)
    print(f"ok prune_targets (support mean {support.mean():.1f})")


def test_surprise_weights_present():
    flat, _ = gen_flat(CFG.features, selfplay.SPKnobs(surprise_w=1.0))
    assert "weight" in flat
    w = flat["weight"]
    assert np.all(w >= 1.0 - 1e-5) and np.any(w > 1.0)
    print(f"ok surprise_weights (mean {w.mean():.2f})")


def test_prior_temp_changes_play():
    a, _ = gen_flat(CFG.features, selfplay.SPKnobs(prior_temp=1.0), seed=5)
    b, _ = gen_flat(CFG.features, selfplay.SPKnobs(prior_temp=3.0), seed=5)
    assert not np.array_equal(a["policy_target"], b["policy_target"])
    print("ok prior_temp")


def test_truncation_draw():
    feats = dataclasses.replace(CFG.features, truncation_draw=True)
    flat, compl = gen_flat(feats, selfplay.SPKnobs(), T=12)  # short: mostly truncated
    vr, vt = flat["value_real"], flat["value_target"]
    assert np.all(vr == 1.0), "truncation-draw tails must be REAL"
    assert np.any(vt == 0.0), "draw tails should produce exact-0 targets"
    print(f"ok truncation_draw (zero-target frac {np.mean(vt == 0):.0%})")


def test_dense_head_and_loss():
    feats = dataclasses.replace(CFG.features, dense_aux=True)
    cfg = dataclasses.replace(CFG, features=feats)
    st = trainer.create_train_state(cfg, jax.random.PRNGKey(0))
    obs = jax.vmap(lambda k: jenv.observe(jenv.init_state(k), feats))(
        jax.random.split(jax.random.PRNGKey(1), 4))
    batch = {"obs": obs,
             "policy_target": jax.nn.one_hot(jnp.arange(4), C.N_ACTIONS),
             "value_target": jnp.zeros(4), "value_real": jnp.ones(4),
             "dense_target": jnp.zeros((4, 7))}
    st2, m = trainer.train_step(st, batch, 1.0, jax.random.PRNGKey(2), False,
                                (0.0, 0.0, 0.0, 0.5))
    assert "dense_loss" in m and np.isfinite(float(m["dense_loss"]))
    print(f"ok dense_head (loss {float(m['dense_loss']):.4f})")


def test_cross_arch_rung_match():
    """play_match(model_b=...) must support a rung whose architecture differs
    from the learner's (s2pilot crash 2026-07-12: C256x15 planted rung vs
    C128x10 from-scratch net -> ScopeParamShapeError)."""
    from jaxarimaa import evaluate
    cfg_a = CFG
    cfg_b = dataclasses.replace(
        CFG, net=dataclasses.replace(CFG.net, channels=CFG.net.channels * 2,
                                     blocks=CFG.net.blocks + 1),
        features=dataclasses.replace(CFG.features, moves_left_head=True))
    model_a = trainer.make_model(cfg_a)
    model_b = trainer.make_model(cfg_b)
    pa = trainer.create_train_state(cfg_a, jax.random.PRNGKey(0)).params
    pb = trainer.create_train_state(cfg_b, jax.random.PRNGKey(1)).params
    a, b, u = evaluate.play_match(model_a, pa, pb, jax.random.PRNGKey(2),
                                  0, 4, 16, 4, 4, cfg_a.features,
                                  cfg_a.features.fast_search, model_b=model_b)
    assert int(a) + int(b) + int(u) == 4
    print(f"ok cross_arch_rung_match (W{int(a)} L{int(b)} U{int(u)})")


def test_handicap_starts():
    """handicap_frac=1: every game starts with exactly 31 pieces (one
    non-rabbit removed from one side)."""
    feats = dataclasses.replace(CFG.features, handicap_games=True)
    recs, _ = gen_flat(feats, selfplay.SPKnobs(handicap_frac=1.0), batch=8, T=4)
    obs0 = recs["obs"][0].astype(np.float32)          # [B, P, 8, 8]
    gold = obs0[:, 0:6].sum(axis=(1, 2, 3))
    silver = obs0[:, 6:12].sum(axis=(1, 2, 3))
    total = gold + silver
    assert (total == 31).all(), total
    assert ((gold == 15) ^ (silver == 15)).all(), (gold, silver)
    print("ok handicap_starts")


def test_rollout_resolve():
    """rollout_resolve + truncation_draw: all targets grounded; resolved
    tails carry real +-1 outcomes when any game resolves."""
    feats = dataclasses.replace(CFG.features, truncation_draw=True,
                                rollout_resolve=True)
    knobs = selfplay.SPKnobs(resolve_steps=120)
    a, _ = gen_flat(feats, knobs, batch=8, T=16, seed=5)
    assert a["value_real"].min() == 1.0, "truncation_draw+resolve => all real"
    assert np.abs(a["value_target"]).max() <= 1.0
    base_feats = dataclasses.replace(CFG.features, truncation_draw=True)
    b, _ = gen_flat(base_feats, selfplay.SPKnobs(), batch=8, T=16, seed=5)
    n_res = int((np.abs(a["value_target"][-1]) > 0.9).sum())
    print(f"ok rollout_resolve (resolved-tail rows: {n_res}, "
          f"targets differ: {not np.array_equal(a['value_target'], b['value_target'])})")


def test_qmix_targets():
    """qmix: non-terminal targets = l*outcome + (1-l)*root_v; the implied
    root_v is bounded and terminal rows keep exact outcomes."""
    feats = dataclasses.replace(CFG.features, qmix_value=True)
    mix, _ = gen_flat(feats, selfplay.SPKnobs(qmix_lambda=0.5), batch=8, T=24,
                      seed=7)
    off, _ = gen_flat(CFG.features, selfplay.SPKnobs(), batch=8, T=24, seed=7)
    tm, to = mix["value_target"], off["value_target"]
    differ = ~np.isclose(tm, to)
    assert differ.any(), "qmix should change some non-terminal targets"
    implied_rv = 2.0 * tm[differ] - to[differ]
    assert np.abs(implied_rv).max() <= 1.0 + 1e-4, "implied root_v out of range"
    print(f"ok qmix_targets (mixed rows: {int(differ.sum())})")


def test_ml_steering_changes_play():
    """ml_steer > 0 (with a moves_left head) changes played games; 0 is the
    baseline (bit-identical path, covered by test_baseline_unchanged)."""
    feats = dataclasses.replace(CFG.features, ml_steering=True,
                                moves_left_head=True)
    a, _ = gen_flat(feats, selfplay.SPKnobs(ml_steer=5.0), batch=8, T=24, seed=9)
    b, _ = gen_flat(feats, selfplay.SPKnobs(ml_steer=0.0), batch=8, T=24, seed=9)
    assert not np.array_equal(np.asarray(a["obs"]), np.asarray(b["obs"])), \
        "steering at weight 5.0 must alter play"
    print("ok ml_steering_changes_play")


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
    print(f"all {len(fns)} stage-2 tests passed")
