"""Completion/perf sweep from the PRETRAINED init: (num_simulations, max_steps).

All prior completion-rate data came from random-strength nets; the pretrained
net plays differently. This measures, per (n, T) combo: game-completion
fraction, env-steps/s, value-target magnitude, and eval-vs-random — the data
that picks the big run's sims + max_steps tiers (and validates the pretrained
net's strength on TPU).

Usage (TPU VM): PYTHONPATH=. python -u configs/sweep_completion.py <sims> <max_steps>
"""

import sys

import jax

from jaxarimaa import train
from jaxarimaa.config import (Config, FeaturesConfig, MCTSConfig, NetConfig,
                              SelfPlayConfig, TrainConfig)

SIMS, T = int(sys.argv[1]), int(sys.argv[2])
print(f"=== SWEEP sims={SIMS} max_steps={T} ===", flush=True)

cfg = Config(
    net=NetConfig(channels=256, blocks=15),
    mcts=MCTSConfig(num_simulations=SIMS, max_num_considered_actions=16),
    selfplay=SelfPlayConfig(
        batch_size=512 * len(jax.devices()), max_steps=T,
        resign_threshold=0.90, full_search_prob=0.25, fast_sims=8,
        greedy_after_turns=15),
    train=TrainConfig(
        train_batch_size=1024, iterations=3, train_steps_per_iter=8,
        replay_capacity=262144, warmup_steps=50, lr=5e-4,
        eval_max_steps=384,
        compile_cache_dir="gs://arimaa-tpu-2026-artifacts/compile-cache"),
    features=FeaturesConfig(
        bf16=True, fast_search=True, resign=True, playout_cap=True,
        symmetry_aug=True, adjudicate_truncation=True, moves_left_head=True,
        planes_frozen=True, planes_trap=True, planes_step_in_turn=True,
        planes_moved=True),
)

train.train(cfg, out_path=f"/tmp/sweep_{SIMS}_{T}.pkl", eval_every=3,
            init_params="results/jaxarimaa/pretrained_c256.pkl")
