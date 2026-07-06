"""Stage A of the staged training plan: small-and-fast bootstrap run.

Config chosen from the TPU measurements (docs/JAX_REBASE_SCOPE.md §13-§14 + the
fast_search A/B): C256x15 + fast_search is FASTER than the config that already
demonstrated learning (729 vs 311 env-steps/s) with 4x the capacity, at 23% MFU.
Stage B (when the eval curve flattens): C512x15 + fast_search (49% MFU).

Run on a TPU VM (usually via infra/run_supervised.sh semantics):
    PYTHONPATH=. python -u configs/stage_a.py <run-name> [iterations]

Metrics: stdout + TensorBoard events (tensorboardX on TPU VMs) under
results/<run>/tb — rsync to GCS and view locally. Learning signals to watch:
  eval/win_rate (vs random, long games), arena/cand_win_rate (self-improvement),
  selfplay/value_target_absmean (game decisiveness), loss/policy.
"""

import sys

import jax

from jaxarimaa import train
from jaxarimaa.config import (Config, FeaturesConfig, MCTSConfig, NetConfig,
                              SelfPlayConfig, TrainConfig)

RUN = sys.argv[1] if len(sys.argv) > 1 else "stage_a"
ITERS = int(sys.argv[2]) if len(sys.argv) > 2 else 400
BUCKET = "gs://arimaa-tpu-2026-artifacts"

# Scale the game batch with the slice: 1024 games PER CHIP keeps per-shard shapes
# (and HBM footprint) identical to the validated single-chip run.
PER_CHIP_GAMES = 512
N_CHIPS = len(jax.devices())

cfg = Config(
    net=NetConfig(channels=256, blocks=15),
    mcts=MCTSConfig(num_simulations=128, max_num_considered_actions=16),
    selfplay=SelfPlayConfig(
        batch_size=PER_CHIP_GAMES * N_CHIPS, max_steps=512,
        resign_threshold=0.95,          # conservative early; value head must earn it
        full_search_prob=0.25, fast_sims=16,
        greedy_after_turns=15,          # decisive play after the opening (optima)
    ),
    train=TrainConfig(
        train_batch_size=1024, iterations=ITERS, train_steps_per_iter=16,
        replay_capacity=262144, warmup_steps=100, lr=2e-3,
        ckpt_interval=5, ckpt_max_keep=3,
        # Local Orbax dir: async sharded saves straight to gs:// time out on
        # multi-device meshes (orbax/gcsfs signaling); the stage runner rsyncs
        # this dir to GCS for durability and pulls it down on fresh VMs.
        ckpt_dir=f"results/jaxarimaa/{RUN}_ckpt",
        compile_cache_dir=f"{BUCKET}/compile-cache",
        arena_interval=10, arena_games=64, arena_threshold=0.55,
        eval_max_steps=384,             # long enough for eval games to finish
    ),
    features=FeaturesConfig(
        bf16=True, fast_search=True, resign=True, playout_cap=True,
        symmetry_aug=True, arena_gating=True, adjudicate_truncation=True,
        moves_left_head=True,
        planes_frozen=True, planes_trap=True, planes_step_in_turn=True,
        planes_moved=True,
    ),
)

train.train(cfg, out_path=f"results/jaxarimaa/{RUN}.pkl", eval_every=8,
            logdir=f"results/jaxarimaa/{RUN}_tb")
