"""Spot-check: the VALIDATED warm-start recipe + trust ratchet, full loop.

The loss-stream probe proved the value stream (not policy targets) poisons the
pretrained trunk, and the fixed recipe (value_loss_weight=0.25 +
value_tail_weight=0.0 + corpus_mix=0.2) scores 0.500 vs the pretrained anchor
on identical data/steps (zero damage). That probe reused STATIC targets, so it
proves no-harm only — this run is the CLIMB gate before the big run: the full
loop (fresh self-play every iteration) from the pretrained init, mirroring
stage_a exactly except iterations=50 and arena every 5 for slope resolution.
At anneal_stages=10 that exercises the ENTIRE ratchet trajectory live.

Success criteria (all four):
  - arena elo/estimate vs the pretrained anchor CLIMBING (positive slope)
  - ratchet reaching a high stage with <=1-2 retreats (no backoff storm)
  - loss/value_real_mse stable or falling (not collapsing to ~0 = overfit)
  - selfplay/completion holding in the 0.65 band

Run on the TPU VM:
    PYTHONPATH=. python -u configs/spotcheck_warmstart.py <run-name>
"""
import sys

import jax

from jaxarimaa import train
from jaxarimaa.config import (Config, FeaturesConfig, MCTSConfig, NetConfig,
                              SelfPlayConfig, TrainConfig)

RUN = sys.argv[1] if len(sys.argv) > 1 else "spotcheck50"

cfg = Config(
    net=NetConfig(channels=256, blocks=15),
    mcts=MCTSConfig(num_simulations=32, max_num_considered_actions=16),
    selfplay=SelfPlayConfig(
        batch_size=512 * len(jax.devices()), max_steps=384,
        resign_threshold=0.90, full_search_prob=0.25, fast_sims=8,
        greedy_after_turns=15),
    train=TrainConfig(
        train_batch_size=1024, iterations=50, train_steps_per_iter=32,
        replay_capacity=262144, warmup_steps=100, lr=5e-4,
        eval_max_steps=384, arena_interval=5, arena_games=64,
        arena_threshold=0.55,
        max_steps_tiers=(256, 384, 512), completion_target=0.65,
        value_loss_weight=0.25, value_tail_weight=0.0,
        corpus_mix=0.2, corpus_path="results/archive_ds_sharp/year*.npz",
        anneal_stages=10,
        anneal_value_loss_weight=1.0, anneal_value_tail_weight=0.25,
        anneal_corpus_mix=0.0,
        ckpt_interval=5, ckpt_max_keep=3,
        ckpt_dir=f"results/jaxarimaa/{RUN}_ckpt",
        compile_cache_dir="gs://arimaa-tpu-2026-artifacts/compile-cache"),
    features=FeaturesConfig(
        bf16=True, fast_search=True, resign=True, playout_cap=True,
        symmetry_aug=True, arena_gating=True, adjudicate_truncation=True,
        moves_left_head=True, planes_frozen=True, planes_trap=True,
        planes_step_in_turn=True, planes_moved=True),
)

train.train(cfg, out_path=f"results/jaxarimaa/{RUN}.pkl", eval_every=10,
            logdir=f"results/jaxarimaa/{RUN}_tb",
            init_params="results/jaxarimaa/pretrained_c256.pkl")
