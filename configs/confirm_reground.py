"""Confirmation run 3: value re-grounded on the SELF-PLAY distribution.

init = pretrained policy + value/aux heads re-fit on sharp-annotated
self-play positions (tools/dump_selfplay_positions.py -> annotate_sharp ->
pretrain --freeze-trunk-policy). Guardrails (KL leash, ratchet, protections)
unchanged from confirm_kl. Success = arena vs own init ~>=0 from the FIRST
reading and climbing (benefits from the start, not recovery-from-damage).

Root cause (T512 stream probe + spotcheck50): mild static policy drift from
noisy-Q search targets on deep positions (~-60 Elo/300 steps, every stream
else clean) amplified into -411 Elo in 4 iters by an unstable feedback loop —
the replay buffer held exactly ONE iteration of data on v5e-4 (4-chip scale-up
silently cut depth 4x) at 32 train steps/iter.

Fix under test (identical to spotcheck50 otherwise, incl. all v2/v3
protections and the trust ratchet): replay 262144 -> 1048576 rows (~4-iter
depth), train_steps_per_iter 32 -> 16, lr 5e-4 -> 3e-4.

Success (green light for the big run):
  - no cliff: arena elo vs pretrained anchor >= ~-50 through iter 10
  - stable-to-climbing after; ratchet advancing without a backoff storm

Run on the TPU VM:  PYTHONPATH=. python -u configs/confirm_reground.py <run> [iters]
"""
import sys

import jax

from jaxarimaa import train
from jaxarimaa.config import (Config, FeaturesConfig, MCTSConfig, NetConfig,
                              SelfPlayConfig, TrainConfig)

RUN = sys.argv[1] if len(sys.argv) > 1 else "reground"
ITERS = int(sys.argv[2]) if len(sys.argv) > 2 else 15

cfg = Config(
    net=NetConfig(channels=256, blocks=15),
    mcts=MCTSConfig(num_simulations=32, max_num_considered_actions=16),
    selfplay=SelfPlayConfig(
        batch_size=512 * len(jax.devices()), max_steps=384,
        resign_threshold=0.90, full_search_prob=0.25, fast_sims=8,
        greedy_after_turns=15),
    train=TrainConfig(
        train_batch_size=1024, iterations=ITERS, train_steps_per_iter=16,
        replay_capacity=1048576, warmup_steps=100, lr=3e-4,
        # arena_games=128 (256 games/round): learner and anchor are near-clones
        # (KL ~0.07), so game outcomes correlate and 128-game rounds were far
        # noisier than binomial — doubling tightens the slope estimate.
        eval_max_steps=384, arena_interval=5, arena_games=128,
        arena_threshold=0.58, ref_interval=4,
        max_steps_tiers=(256, 384, 512), completion_target=0.65,
        value_loss_weight=0.25, value_tail_weight=0.0,
        # mix sweep: 75:25 is NET POSITIVE (0.527); 50:50 already collapses
        # (0.418); the old 0.2 was deep in the damage zone
        corpus_mix=0.75,
        corpus_path="results/archive_ds_sharp/year*.npz",
        kl_prior_weight=1.0, anneal_kl_prior_weight=0.0,
        anneal_stages=40,  # 4x smaller steps; full anneal ~iter 400+ (horizon scaled to run length; stage-1 window coincided with the rehearsal decline)
        anneal_value_loss_weight=1.0, anneal_value_tail_weight=0.25,
        anneal_corpus_mix=0.70,  # FLOOR: live bleed began as mix crossed ~0.65 (matches the static sweep); weaning below this waits for demonstrated self-improvement
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
            init_params="results/jaxarimaa/regrounded_c256.pkl")
