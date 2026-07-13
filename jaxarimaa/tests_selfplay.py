"""Self-play target-semantics tests (promoted from scratchpad after the temp
dir purge ate the originals — these are permanent preflight members now).

1. resign/value_real: belief-adjudicated games must NOT be marked real
   (the self-confirming-value poison, fixed 2026-07-08).
2. resume: kill->relaunch restores anneal sidecar incl. tier/arena state.

Run:  PYTHONPATH=. python jaxarimaa/tests_selfplay.py
"""
import dataclasses
import shutil
import subprocess
import sys

import numpy as np
import jax

from jaxarimaa import distributed, selfplay, trainer
from jaxarimaa.config import tiny_config


def test_resign_rows_not_value_real():
    cfg = tiny_config()
    feats = dataclasses.replace(cfg.features, resign=True,
                                adjudicate_truncation=True)
    cfg = dataclasses.replace(cfg, features=feats)
    mesh = distributed.make_mesh()
    model = trainer.make_model(cfg)
    st = trainer.create_train_state(cfg, jax.random.PRNGKey(0))
    knobs = selfplay.SPKnobs(resign_thresh=0.0, full_prob=0.25, fast_sims=2,
                             greedy_after=0)
    gen = selfplay.make_generate(mesh, model, 8, 40, (4, 4), feats, knobs)
    recs, completed = gen(st.params, jax.random.PRNGKey(1))
    vr = np.asarray(recs["value_real"])
    assert float(completed) > 0.5, "resign@0 should adjudicate nearly every game"
    assert vr.max() == 0.0, "belief-adjudicated rows must be value_real=0"
    print("ok resign_not_real")


def test_resume_restores_sidecar():
    ck = "/tmp/jaxarimaa_tests_resume"
    shutil.rmtree(ck, ignore_errors=True)
    prog = '''
import dataclasses, sys
from jaxarimaa import train
from jaxarimaa.config import tiny_config
cfg = tiny_config()
cfg = dataclasses.replace(cfg,
    train=dataclasses.replace(cfg.train, iterations=int(sys.argv[1]),
        arena_interval=1, arena_games=2, ckpt_interval=1,
        ckpt_dir="{ck}/ckpt", anneal_stages=10,
        value_loss_weight=0.25, value_tail_weight=0.0),
    features=dataclasses.replace(cfg.features, arena_gating=True))
train.train(cfg, out_path="{ck}/model.pkl", eval_every=0, verbose=True,
            logdir="{ck}/tb")
'''.format(ck=ck)
    r1 = subprocess.run([sys.executable, "-c", prog, "2"],
                        capture_output=True, text=True, timeout=900)
    assert "[anneal]" in r1.stdout, r1.stdout[-500:] + r1.stderr[-500:]
    r2 = subprocess.run([sys.executable, "-c", prog, "4"],
                        capture_output=True, text=True, timeout=900)
    assert "resumed from checkpoint" in r2.stdout, r2.stdout[-500:]
    assert "restored anneal state" in r2.stdout, r2.stdout[-500:]
    print("ok resume_sidecar")


def test_probation_counter_and_switch():
    ck = "/tmp/jaxarimaa_tests_probation"
    shutil.rmtree(ck, ignore_errors=True)
    prog = '''
import dataclasses
from jaxarimaa import train
from jaxarimaa.config import tiny_config
cfg = tiny_config()
cfg = dataclasses.replace(cfg,
    train=dataclasses.replace(cfg.train, iterations=3,
        arena_interval=1, arena_games=2, arena_threshold=1.01,
        probation_after=2, ckpt_interval=1, ckpt_dir="{ck}/ckpt"),
    features=dataclasses.replace(cfg.features, arena_gating=True,
                                 certification=True))
train.train(cfg, out_path="{ck}/model.pkl", eval_every=0, verbose=True,
            logdir="{ck}/tb")
'''.format(ck=ck)
    r = subprocess.run([sys.executable, "-c", prog],
                       capture_output=True, text=True, timeout=900)
    assert "[probation] 2 consecutive" in r.stdout, \
        r.stdout[-800:] + r.stderr[-800:]
    import json as _json
    with open(f"{ck}/ckpt/anneal.json") as f:
        assert _json.load(f)["consec_failed"] >= 2
    print("ok probation_switch")


def test_packed_state_roundtrip():
    """fast_search packs tree-node States into one uint32 buffer (the 58%
    gather fix); the bijection must be exact on every field, including after
    real play (nonzero rep ring, mid-turn steps_left)."""
    import jax.numpy as jnp
    from jaxarimaa import env as jenv, fast_search
    from jaxarimaa.types import State
    states = jax.vmap(jenv.init_state)(jax.random.split(jax.random.PRNGKey(0), 16))
    for i in range(40):
        legal = jax.vmap(jenv.legal_action_mask)(states)
        g = jax.random.gumbel(jax.random.PRNGKey(i), legal.shape)
        states = jax.vmap(jenv.step)(states, jnp.argmax(
            jnp.where(legal, g, -jnp.inf), axis=-1))
    rt = fast_search.unpack_states(fast_search.pack_states(states))
    for f in State.__dataclass_fields__:
        a, b = getattr(states, f), getattr(rt, f)
        assert a.dtype == b.dtype and a.shape == b.shape, f
        assert np.array_equal(np.asarray(a), np.asarray(b)), f
    print("ok packed_state_roundtrip")


if __name__ == "__main__":
    test_resign_rows_not_value_real()
    test_resume_restores_sidecar()
    test_probation_counter_and_switch()
    test_packed_state_roundtrip()
    print("all selfplay tests passed")
