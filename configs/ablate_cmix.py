"""Corpus-mix ablation arm: pretrained init + sweep-chosen config (n=32, T=384).

Usage (TPU VM): PYTHONPATH=. python -u configs/ablate_cmix.py <mix> <iters>
Arena anchor starts at the pretrained init, so elo/estimate directly measures
improvement OVER the pretrained model.
"""
import sys

import jax

from jaxarimaa import train
from jaxarimaa.config import (Config, FeaturesConfig, MCTSConfig, NetConfig,
                              SelfPlayConfig, TrainConfig)

MIX = float(sys.argv[1])
ITERS = int(sys.argv[2]) if len(sys.argv) > 2 else 50
print(f"=== ABLATION corpus_mix={MIX} iters={ITERS} ===", flush=True)

cfg = Config(
    net=NetConfig(channels=256, blocks=15),
    mcts=MCTSConfig(num_simulations=32, max_num_considered_actions=16),
    selfplay=SelfPlayConfig(
        batch_size=512 * len(jax.devices()), max_steps=384,
        resign_threshold=0.90, full_search_prob=0.25, fast_sims=8,
        greedy_after_turns=15),
    train=TrainConfig(
        train_batch_size=1024, iterations=ITERS, train_steps_per_iter=32,
        replay_capacity=262144, warmup_steps=50, lr=5e-4,
        eval_max_steps=384, arena_interval=10, arena_games=64,
        arena_threshold=0.55,
        max_steps_tiers=(256, 384, 512), completion_target=0.65,
        corpus_mix=MIX, corpus_path="results/archive_ds_sharp/year*.npz",
        compile_cache_dir="gs://arimaa-tpu-2026-artifacts/compile-cache"),
    features=FeaturesConfig(
        bf16=True, fast_search=True, resign=True, playout_cap=True,
        symmetry_aug=True, arena_gating=True, adjudicate_truncation=True,
        moves_left_head=True, planes_frozen=True, planes_trap=True,
        planes_step_in_turn=True, planes_moved=True),
)

train.train(cfg, out_path=f"results/jaxarimaa/cmix{MIX}.pkl", eval_every=10,
            init_params="results/jaxarimaa/pretrained_c256.pkl")
