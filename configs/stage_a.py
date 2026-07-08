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

# Post-cold-start regime: with a pretrained/grounded value head, deep search is
# no longer needed to GROUND value (only to improve the policy, which Gumbel does
# at low sims) — so sims drop 128->64 (~4x less tree-walk, the round-tail win),
# the value target is pure game-outcome (search-root blend removed), resign is
# less conservative (value is trustworthy), and the replay ratio rises (healthy
# targets -> learn more per game). Set INIT_PARAMS to the pretrained checkpoint.
INIT_PARAMS = None  # e.g. "results/jaxarimaa/pretrained.pkl"

cfg = Config(
    net=NetConfig(channels=256, blocks=15),
    mcts=MCTSConfig(num_simulations=32, max_num_considered_actions=16),
    selfplay=SelfPlayConfig(
        batch_size=PER_CHIP_GAMES * N_CHIPS, max_steps=384,
        resign_threshold=0.90,          # value is pretrained-grounded -> adjudicate sooner
        full_search_prob=0.25, fast_sims=8,
        greedy_after_turns=15,          # decisive play after the opening (optima)
    ),
    train=TrainConfig(
        # Loop-gain control (T512 stream probe + spotcheck50 post-mortem): the
        # warm-start collapse was mild static drift (~-60 Elo/300 steps from
        # noisy-Q search targets on deep positions) AMPLIFIED by an unstable
        # data feedback loop. 262144 rows was exactly ONE iteration of rollout
        # on v5e-4 (the 4-chip scale-up silently cut buffer depth 4x) -> 1M
        # rows restores ~4-iter depth; steps 32->16 halves updates per rollout.
        train_batch_size=1024, iterations=ITERS, train_steps_per_iter=16,
        replay_capacity=1048576, warmup_steps=100,
        lr=3e-4,  # warm-start LR, lowered again with the loop-gain fix (static
                  # policy drift per step scales with LR)
        max_steps_tiers=(256, 384, 512), completion_target=0.65,
        # Value-stream protection (probe-verified: value gradients through the
        # shared trunk were the warm-start poison; policy targets exonerated):
        value_loss_weight=0.25,   # damp trunk churn while value calibrates
        value_tail_weight=0.0,    # REAL outcomes only for the value loss
        corpus_mix=0.2,           # sharp-anchored batches (proven ~450 Elo shield)
        corpus_path="results/archive_ds_sharp/year*.npz",
        # Trust ratchet: anneal the three protections above toward full AlphaZero
        # (value_w 0.25->1.0, tail_w 0->0.25, mix 0.2->0) one stage per healthy
        # arena round; retreat on an Elo regression. Earliest full anneal:
        # anneal_stages * arena_interval = 100 iters (of ITERS).
        anneal_stages=10,
        anneal_value_loss_weight=1.0, anneal_value_tail_weight=0.25,
        anneal_corpus_mix=0.0,
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
            logdir=f"results/jaxarimaa/{RUN}_tb", init_params=INIT_PARAMS)
