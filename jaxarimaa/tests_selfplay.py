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


if __name__ == "__main__":
    test_resign_rows_not_value_real()
    test_resume_restores_sidecar()
    print("all selfplay tests passed")
