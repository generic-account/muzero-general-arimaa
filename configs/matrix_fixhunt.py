"""Fix-hunt matrix: one-variable arms against the spotcheck50 failure.

spotcheck50 (control curve, gs://.../runs/spotcheck50/): warm-started self-play
drops to ~-400 Elo vs the pretrained anchor within 4 iters, then holds a noisy
-350 floor. Diagnosis evidence: value head real-row MSE 0.85 on self-play
positions (vs 0.15 on corpus/archive) -> Gumbel action_weights reweight the
expert prior by out-of-distribution Q; the local probe missed it because its
cache was generated at T=128 (early-game, in-book positions only).

Arms (15 iters each, arena vs pretrained anchor every 3, anneal OFF):
  vhead  - AlphaGo-style phase 0: calibrate value/aux HEADS on frozen
           trunk+policy (freeze_trunk_policy), value on real self-play rows
           only, no corpus. Success = value_real_mse falling hard; arena is
           free to move (search uses the recalibrated value).
  visits - policy target = root visit counts (optima/AZ; no Q-imputation for
           unvisited actions). Success = no cliff (arena ~>= -50 and stable).
  shape  - greedy_after=0 + resign 0.97: longer, more diverse, more real-
           terminal games (in-book-er positions, more value_real rows).
  replay - replay_capacity 262k->1M (buffer 1 iter -> 4 iters deep on v5e-4;
           the 4-chip scale-up silently cut depth 4x) + train_steps 32->16.

Run:  PYTHONPATH=. python -u configs/matrix_fixhunt.py <arm>
"""
import sys

import jax

from jaxarimaa import train
from jaxarimaa.config import (Config, FeaturesConfig, MCTSConfig, NetConfig,
                              SelfPlayConfig, TrainConfig)

ARM = sys.argv[1]

tr = dict(train_batch_size=1024, iterations=15, train_steps_per_iter=32,
          replay_capacity=262144, warmup_steps=100, lr=5e-4,
          eval_max_steps=384, arena_interval=3, arena_games=64,
          arena_threshold=0.55,
          max_steps_tiers=(256, 384, 512), completion_target=0.65,
          value_loss_weight=0.25, value_tail_weight=0.0,
          corpus_mix=0.2, corpus_path="results/archive_ds_sharp/year*.npz",
          compile_cache_dir="gs://arimaa-tpu-2026-artifacts/compile-cache")
ft = dict(bf16=True, fast_search=True, resign=True, playout_cap=True,
          symmetry_aug=True, arena_gating=True, adjudicate_truncation=True,
          moves_left_head=True, planes_frozen=True, planes_trap=True,
          planes_step_in_turn=True, planes_moved=True)
sp = dict(batch_size=512 * len(jax.devices()), max_steps=384,
          resign_threshold=0.90, full_search_prob=0.25, fast_sims=8,
          greedy_after_turns=15)

if ARM == "vhead":
    tr.update(freeze_trunk_policy=True, policy_loss_weight=0.0,
              value_loss_weight=1.0, corpus_mix=0.0)
elif ARM == "visits":
    ft.update(visit_policy_targets=True)
elif ARM == "shape":
    sp.update(greedy_after_turns=0, resign_threshold=0.97)
elif ARM == "replay":
    tr.update(replay_capacity=1048576, train_steps_per_iter=16)
else:
    raise SystemExit(f"unknown arm {ARM!r} (vhead|visits|shape|replay)")

cfg = Config(net=NetConfig(channels=256, blocks=15),
             mcts=MCTSConfig(num_simulations=32, max_num_considered_actions=16),
             selfplay=SelfPlayConfig(**sp), train=TrainConfig(**tr),
             features=FeaturesConfig(**ft))
train.train(cfg, out_path=f"results/jaxarimaa/matrix_{ARM}.pkl", eval_every=0,
            logdir=f"results/jaxarimaa/matrix_{ARM}_tb",
            init_params="results/jaxarimaa/pretrained_c256.pkl")
