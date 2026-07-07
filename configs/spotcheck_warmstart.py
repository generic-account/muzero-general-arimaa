"""Spot-check: warm-start recipe fix (low LR + conservative resign + corpus mix).

The corpus-mix ablation showed BOTH arms degrade vs the pretrained anchor
(pure self-play -720 Elo, 20% mix -246): cold-Adam at 5e-4 + resign@0.90 with a
confident pretrained value head self-poisons. This arm tests the fix:
lr 1e-4, warmup 300, resign 0.97, mix 0.2, arena every 5 for slope resolution.
Success = elo-vs-anchor >= ~0 and trending up.
"""
import jax

from jaxarimaa import train
from jaxarimaa.config import (Config, FeaturesConfig, MCTSConfig, NetConfig,
                              SelfPlayConfig, TrainConfig)

cfg = Config(
    net=NetConfig(channels=256, blocks=15),
    mcts=MCTSConfig(num_simulations=32, max_num_considered_actions=16),
    selfplay=SelfPlayConfig(
        batch_size=512 * len(jax.devices()), max_steps=384,
        resign_threshold=0.97, full_search_prob=0.25, fast_sims=8,
        greedy_after_turns=15),
    train=TrainConfig(
        train_batch_size=1024, iterations=30, train_steps_per_iter=32,
        replay_capacity=262144, warmup_steps=300, lr=1e-4,
        eval_max_steps=384, arena_interval=5, arena_games=64,
        arena_threshold=0.55,
        max_steps_tiers=(256, 384, 512), completion_target=0.65,
        corpus_mix=0.2, corpus_path="results/archive_ds_sharp/year*.npz",
        compile_cache_dir="gs://arimaa-tpu-2026-artifacts/compile-cache"),
    features=FeaturesConfig(
        bf16=True, fast_search=True, resign=True, playout_cap=True,
        symmetry_aug=True, arena_gating=True, adjudicate_truncation=True,
        moves_left_head=True, planes_frozen=True, planes_trap=True,
        planes_step_in_turn=True, planes_moved=True),
)

train.train(cfg, out_path="results/jaxarimaa/spotcheck.pkl", eval_every=10,
            init_params="results/jaxarimaa/pretrained_c256.pkl")
