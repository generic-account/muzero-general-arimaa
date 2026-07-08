"""Stream probe on PRODUCTION-SHAPED data (TPU): which gradient stream, on
T=512 self-play data, damages the pretrained policy?

The original (local) probe cleared policy/value/ml streams individually but its
cache was generated at T=128 (early-game, in-book) WITHOUT symmetry aug; the
full loop (spotcheck50) still crashed -411 with all loss protections on. The
untested discriminators are symmetry aug, data depth, and freshness. This
probe re-runs channel attribution on real-depth data:

  cache:  one production-knob rollout (T=512, resign .90, greedy 15,
          playout-cap .25) from the pretrained params
  arms:   300 fine-tune steps each on the SAME cache, then 128-game arena
          vs the pretrained params (score ~0.5 = harmless, <0.4 = poison)

Run (TPU VM): PYTHONPATH=. ~/venv/bin/python -u tools/tpu_stream_probe.py
"""
import numpy as np
import jax
import jax.numpy as jnp

from jaxarimaa import checkpoint, distributed, evaluate, selfplay, trainer
from jaxarimaa.config import (Config, FeaturesConfig, MCTSConfig, NetConfig,
                              TrainConfig)

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

print("generating production-shaped cache (T=512)...", flush=True)
gen = selfplay.make_generate(mesh, model, 512 * ndev, 512, (32, 16), feats,
                             selfplay.SPKnobs(resign_thresh=0.90, full_prob=0.25,
                                              fast_sims=8, greedy_after=15))
recs, compl = gen(params, jax.random.PRNGKey(0))
flat = {k: np.asarray(v) for k, v in selfplay.flatten_samples(recs).items()}
N = len(flat["value_target"])
vr = flat["value_real"]
print(f"cache: {N} rows, completion {float(compl):.2f}, "
      f"value_real frac {float(vr.mean()):.2f}", flush=True)

shard = lambda b: distributed.shard_batch(mesh, b)


def finetune(pw, vw, mlw, sym, tag, steps=300):
    st = trainer.create_train_state(cfg, jax.random.PRNGKey(1)).replace(params=params)
    rng = np.random.default_rng(2)
    key = jax.random.PRNGKey(3)
    for _ in range(steps):
        idx = rng.integers(0, N, size=1024)
        batch = shard({k: jnp.asarray(v[idx]) for k, v in flat.items()})
        key, ka = jax.random.split(key)
        st, m = trainer.train_step(st, batch, vw, ka, sym, (mlw, 0.0, 0.0),
                                   pw, 0.0)  # value_tail_weight=0 (production)
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


ARMS = [
    ("pol_nosym",  (1.0, 0.0, 0.0, False)),
    ("pol_sym",    (1.0, 0.0, 0.0, True)),
    ("val025",     (0.0, 0.25, 0.0, False)),
    ("ml015",      (0.0, 0.0, 0.15, False)),
    ("all_nosym",  (1.0, 0.25, 0.15, False)),
    ("all_sym",    (1.0, 0.25, 0.15, True)),
]
results = {}
for tag, (pw, vw, mlw, sym) in ARMS:
    p = finetune(pw, vw, mlw, sym, tag)
    results[tag] = arena(p, tag)
print("=== STREAM PROBE T512 VERDICT ===", flush=True)
for tag, s in results.items():
    print(f"{tag:12s} {s:.3f} {'POISON' if s < 0.40 else 'ok'}", flush=True)
