"""Arena measurement-variance probe: replay ONE fixed pairing under K seeds.

The rehearsal's unbiased rung readings swung +77 -> -34 Elo (iter 9 -> 19),
~3.5 sigma under the BINOMIAL error bar (+-22 at 256 games). Near-clone
games may correlate (shared openings/lines), making the true per-match sigma
much larger. This probe separates measurement variance from genuine strength
change: fixed params, only the match rng varies.

  pairing: anchor.pkl (the iter-9 promoted net) vs regrounded init
  K=6 replays x 256 games -> empirical sigma of the score

sigma ~0.031 = binomial -> readings were real (genuine oscillation).
sigma >~0.06 = correlated-game noise -> the metric needs bigger/decorrelated
matches and the ratchet bands need recalibration; "oscillation" is mostly
measurement.

Run (TPU VM): PYTHONPATH=. ~/venv/bin/python -u tools/tpu_replay_variance.py \
    <params_a.pkl> <params_b.pkl> [K=6] [games_per_color=128]
"""
import sys

import numpy as np
import jax

from jaxarimaa import checkpoint, evaluate, trainer
from jaxarimaa.config import Config, FeaturesConfig, MCTSConfig, NetConfig

A = sys.argv[1]
B = sys.argv[2]
K = int(sys.argv[3]) if len(sys.argv) > 3 else 6
G = int(sys.argv[4]) if len(sys.argv) > 4 else 128

feats = FeaturesConfig(
    bf16=True, fast_search=True, resign=True, playout_cap=True,
    adjudicate_truncation=True, moves_left_head=True, planes_frozen=True,
    planes_trap=True, planes_step_in_turn=True, planes_moved=True)
cfg = Config(net=NetConfig(channels=256, blocks=15), features=feats,
             mcts=MCTSConfig(num_simulations=32, max_num_considered_actions=16))
pa, _ = checkpoint.load(A)
pb, _ = checkpoint.load(B)
model = trainer.make_model(cfg)

scores = []
for k in range(K):
    k1, k2 = jax.random.split(jax.random.PRNGKey(1000 + k))
    a1, b1, u1 = evaluate.play_match(model, pa, pb, k1, 0, G, 384, 32, 16,
                                     feats, True)
    a2, b2, u2 = evaluate.play_match(model, pb, pa, k2, 0, G, 384, 32, 16,
                                     feats, True)
    w, l = int(a1) + int(b2), int(b1) + int(a2)
    d = int(u1) + int(u2)
    s = (w + 0.5 * d) / max(w + l + d, 1)
    scores.append(s)
    print(f"replay {k}: W{w} L{l} D{d} score={s:.3f}", flush=True)

s = np.asarray(scores)
n = 2 * G
binom = float(np.sqrt(s.mean() * (1 - s.mean()) / n))
print("=== REPLAY VARIANCE VERDICT ===", flush=True)
print(f"scores: {[f'{x:.3f}' for x in scores]}", flush=True)
print(f"mean={s.mean():.3f} empirical_sigma={s.std(ddof=1):.4f} "
      f"binomial_sigma={binom:.4f} "
      f"inflation={s.std(ddof=1) / max(binom, 1e-9):.2f}x", flush=True)
