"""Static archive:self-play mix-ratio sweep — the forgetting-vs-mixing curve.

Fixed endpoints (T512 stream probe): pure self-play targets = 0.406 arena vs
pretrained (-66 Elo/300 steps); archive mixing was the strongest positive
lever ever measured live (corpus-mix arm was the only rising curve). This
sweeps the ratio ON STATIC DATA (no loop confounds) to (a) set the big run's
corpus_mix floor by measurement, (b) decide whether policy training is
mix-salvageable at high ratios.

Arms (production loss weights, 300 steps, batch 1024, per-step Bernoulli
batch source exactly like the live loop): mix = 1.00 / 0.75 / 0.50 fraction
of ARCHIVE batches. Arena = 128 games vs pretrained (same protocol as the
0.406 endpoint).

Run (TPU VM): PYTHONPATH=. ~/venv/bin/python -u tools/tpu_mix_sweep.py
"""
import functools

import numpy as np
import jax
import jax.numpy as jnp

from jaxarimaa import checkpoint, distributed, evaluate, selfplay, trainer
from jaxarimaa.config import (Config, FeaturesConfig, MCTSConfig, NetConfig,
                              TrainConfig)
from jaxarimaa.corpus import CorpusSampler

feats = FeaturesConfig(
    bf16=True, fast_search=True, resign=True, playout_cap=True,
    adjudicate_truncation=True, moves_left_head=True, planes_frozen=True,
    planes_trap=True, planes_step_in_turn=True, planes_moved=True)
cfg = Config(net=NetConfig(channels=256, blocks=15), features=feats,
             mcts=MCTSConfig(num_simulations=32, max_num_considered_actions=16),
             train=TrainConfig(train_batch_size=1024, lr=5e-4, warmup_steps=20,
                               iterations=1, train_steps_per_iter=300))
params, _ = checkpoint.load("results/jaxarimaa/pretrained_c256.pkl")
mesh = distributed.make_mesh()
model = trainer.make_model(cfg)
ndev = len(jax.devices())
shard = functools.partial(distributed.shard_batch, mesh)

print("generating production-shaped self-play cache (T=512)...", flush=True)
gen = selfplay.make_generate(mesh, model, 512 * ndev, 512, (32, 16), feats,
                             selfplay.SPKnobs(resign_thresh=0.90, full_prob=0.25,
                                              fast_sims=8, greedy_after=15))
recs, compl = gen(params, jax.random.PRNGKey(0))
flat = {k: np.asarray(v) for k, v in selfplay.flatten_samples(recs).items()}
N = len(flat["value_target"])
print(f"cache: {N} rows, completion {float(compl):.2f}", flush=True)

print("loading archive corpus...", flush=True)
corpus = CorpusSampler("results/archive_ds_sharp/year*.npz", feats)
print(f"corpus: {corpus.n:,} rows", flush=True)


def finetune(mix, tag, steps=300):
    st = trainer.create_train_state(cfg, jax.random.PRNGKey(1)).replace(params=params)
    rng = np.random.default_rng(2)
    key = jax.random.PRNGKey(3)
    for _ in range(steps):
        key, ka = jax.random.split(key)
        if rng.random() < mix:
            batch = corpus.sample(rng, 1024, shard_fn=shard)
        else:
            idx = rng.integers(0, N, size=1024)
            batch = shard({k: jnp.asarray(v[idx]) for k, v in flat.items()})
        st, m = trainer.train_step(st, batch, 0.25, ka, False, (0.15, 0.0, 0.0),
                                   1.0, 0.0)
    print(f"[{tag}] pol={float(m['policy_loss']):.3f} "
          f"val={float(m['value_loss']):.3f}", flush=True)
    return st.params


def arena(p, tag):
    k1, k2 = jax.random.split(jax.random.PRNGKey(7))
    a1, b1, u1 = evaluate.play_match(model, p, params, k1, 0, 64, 384, 32, 16,
                                     feats, True)
    a2, b2, u2 = evaluate.play_match(model, params, p, k2, 0, 64, 384, 32, 16,
                                     feats, True)
    w, l = int(a1) + int(b2), int(b1) + int(a2)
    d = int(u1) + int(u2)
    s = (w + 0.5 * d) / max(w + l + d, 1)
    print(f"[{tag}] vs pretrained: W{w} L{l} D{d} score={s:.3f}", flush=True)
    return s


results = {}
for mix in (1.00, 0.75, 0.50):
    tag = f"mix{int(mix * 100)}"
    p = finetune(mix, tag)
    results[tag] = arena(p, tag)
print("=== MIX SWEEP VERDICT (score vs pretrained; selfplay-only known 0.406) ===",
      flush=True)
for tag, s in results.items():
    print(f"{tag:8s} {s:.3f}", flush=True)
