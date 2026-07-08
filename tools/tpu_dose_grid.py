"""Data-utilization grid: training-dose x cache-size, at the validated mix.

The live loop generates 262k trained-quality rows/iter but consumes ~5k
self-play samples (~2% utilization; generation is 99.7% of iteration cost).
Two remedies compete: MORE TRAINING per batch (free in wall-clock, bounded by
drift dose) vs LESS GENERATION (cuts the dominant cost, bounded by data
diversity). This grid measures both marginal curves in one shot:

  doses  {64, 192, 576, 1728} steps @ batch 1024, mix 0.70
         (64 ~= 4 live iters of 16 steps; 1728 ~= 2 epochs of the cache's
          self-play rows at the 30% stream share)
  caches {full 262k, subsampled 65k}  (65k ~= 4x smaller generation)

All arms fine-tune the re-grounded init; arena = 128 games/color vs the
UNCHANGED init (the mix-sweep protocol, whose numbers matched live behavior).
Read-out: where the dose curve bends = the steps/iter to run; if 65k tracks
262k at that dose, generation can shrink ~4x on top.

Run (TPU VM): PYTHONPATH=. ~/venv/bin/python -u tools/tpu_dose_grid.py
"""
import functools

import numpy as np
import jax
import jax.numpy as jnp

from jaxarimaa import checkpoint, distributed, evaluate, selfplay, trainer
from jaxarimaa.config import (Config, FeaturesConfig, MCTSConfig, NetConfig,
                              TrainConfig)
from jaxarimaa.corpus import CorpusSampler

MIX = 0.70
DOSES = (64, 192, 576, 1728)
ARENA_G = 128  # per color

feats = FeaturesConfig(
    bf16=True, fast_search=True, resign=True, playout_cap=True,
    adjudicate_truncation=True, moves_left_head=True, planes_frozen=True,
    planes_trap=True, planes_step_in_turn=True, planes_moved=True)
cfg = Config(net=NetConfig(channels=256, blocks=15), features=feats,
             mcts=MCTSConfig(num_simulations=32, max_num_considered_actions=16),
             train=TrainConfig(train_batch_size=1024, lr=3e-4, warmup_steps=20,
                               iterations=1, train_steps_per_iter=1))
params, _ = checkpoint.load("results/jaxarimaa/regrounded_c256.pkl")
mesh = distributed.make_mesh()
model = trainer.make_model(cfg)
ndev = len(jax.devices())
shard = functools.partial(distributed.shard_batch, mesh)

print("generating production-shaped cache (T=512)...", flush=True)
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


def finetune(steps, pool, tag):
    st = trainer.create_train_state(cfg, jax.random.PRNGKey(1)).replace(params=params)
    rng = np.random.default_rng(2)
    key = jax.random.PRNGKey(3)
    for _ in range(steps):
        key, ka = jax.random.split(key)
        if rng.random() < MIX:
            batch = corpus.sample(rng, 1024, shard_fn=shard)
        else:
            idx = rng.integers(0, pool, size=1024)  # pool<=N: subsampled cache
            batch = shard({k: jnp.asarray(v[idx]) for k, v in flat.items()})
        st, m = trainer.train_step(st, batch, 0.25, ka, False, (0.15, 0.0, 0.0),
                                   1.0, 0.0)
    print(f"[{tag}] pol={float(m['policy_loss']):.3f} "
          f"val={float(m['value_loss']):.3f}", flush=True)
    return st.params


def arena(p, tag):
    k1, k2 = jax.random.split(jax.random.PRNGKey(7))
    a1, b1, u1 = evaluate.play_match(model, p, params, k1, 0, ARENA_G, 384,
                                     32, 16, feats, True)
    a2, b2, u2 = evaluate.play_match(model, params, p, k2, 0, ARENA_G, 384,
                                     32, 16, feats, True)
    w, l = int(a1) + int(b2), int(b1) + int(a2)
    d = int(u1) + int(u2)
    s = (w + 0.5 * d) / max(w + l + d, 1)
    print(f"[{tag}] vs init: W{w} L{l} D{d} score={s:.3f}", flush=True)
    return s


results = {}
for dose in DOSES:
    for pool, ptag in ((N, "full"), (N // 4, "sub65k")):
        tag = f"d{dose}_{ptag}"
        p = finetune(dose, pool, tag)
        results[tag] = arena(p, tag)
print("=== DOSE GRID VERDICT (score vs re-grounded init) ===", flush=True)
for tag, s in results.items():
    print(f"{tag:16s} {s:.3f}", flush=True)
