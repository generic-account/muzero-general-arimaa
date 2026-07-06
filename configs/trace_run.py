"""Capture an XLA profiler trace of the current production self-play/train shape.

Runs 4 iterations of the stage-A configuration (no checkpointing/eval noise) and
captures a full trace of iteration 2 to /tmp/xla-trace2 — used to ground the next
round of performance work. Usage (TPU VM):
    PYTHONPATH=. python -u configs/trace_run.py
"""

import dataclasses

from jaxarimaa import train
from jaxarimaa.config import (Config, FeaturesConfig, MCTSConfig, NetConfig,
                              SelfPlayConfig, TrainConfig)

import jax

cfg = Config(
    net=NetConfig(channels=256, blocks=15),
    mcts=MCTSConfig(num_simulations=32, max_num_considered_actions=16),
    selfplay=SelfPlayConfig(batch_size=1024 * len(jax.devices()), max_steps=256,
                            resign_threshold=0.95),
    train=TrainConfig(train_batch_size=1024, iterations=4, train_steps_per_iter=16,
                      replay_capacity=262144, warmup_steps=8,
                      compile_cache_dir="gs://arimaa-tpu-2026-artifacts/compile-cache"),
    features=FeaturesConfig(bf16=True, fast_search=True, resign=True, playout_cap=True,
                            symmetry_aug=True, planes_frozen=True, planes_trap=True,
                            planes_step_in_turn=True, planes_moved=True),
)

train.train(cfg, out_path="/tmp/trace_out.pkl", eval_every=0,
            profile_dir="/tmp/xla-trace2")
