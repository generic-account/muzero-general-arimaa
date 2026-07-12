"""Stage-2 pilot: from-scratch AZ loop with the optima/KataGo mechanism set.

Everything the Stage-1 forensics taught, inverted into a loop DESIGNED to
improve rather than to not-decline:
  - from scratch (no imitation warm-start, no KL leash, no corpus mixing —
    the triple prior-tether is gone; the imitation net survives only as the
    planted rung, i.e. the absolute measuring stick)
  - real improvement operator: n=128/m=32 full searches on 25% of moves
  - deblunder: value targets are outcome Q-mixed past exploration blunders
  - dense Arimaa aux targets (trap flow, capture-soon, material trajectory)
  - pruned policy targets (no Q-imputed mass on unvisited actions)
  - certification: self-play data only ever from arena-promoted nets
  - prior flattening (temp 1.2), no resign, step-cap games are real draws

Purpose: the from-scratch Elo-vs-games curve on our TPU stack — the number
that decides how the real budget gets committed.

Run: bash infra/run_stage.sh <run> configs/stage2_pilot.py [iters]
"""
import sys

import jax

from jaxarimaa import train
from jaxarimaa.config import (Config, FeaturesConfig, MCTSConfig, NetConfig,
                              SelfPlayConfig, TrainConfig)

RUN = sys.argv[1] if len(sys.argv) > 1 else "s2pilot"
ITERS = int(sys.argv[2]) if len(sys.argv) > 2 else 200
BUCKET = "gs://arimaa-tpu-2026-artifacts"

PER_CHIP_GAMES = 512
N_CHIPS = len(jax.devices())

cfg = Config(
    net=NetConfig(channels=128, blocks=10),   # optima-scale; grow later
    mcts=MCTSConfig(num_simulations=128, max_num_considered_actions=32),
    selfplay=SelfPlayConfig(
        batch_size=PER_CHIP_GAMES * N_CHIPS, max_steps=384,
        full_search_prob=0.25, fast_sims=16,
        greedy_after_turns=0,               # exploration > decisiveness here
    ),
    train=TrainConfig(
        train_batch_size=1024, iterations=ITERS, train_steps_per_iter=64,
        replay_capacity=1048576, warmup_steps=500,
        lr=2e-3,                            # from-scratch LR (no prior to protect)
        max_steps_tiers=(256, 384, 512), completion_target=0.65,
        value_loss_weight=1.0, value_tail_weight=1.0,
        dense_aux_weight=0.3, dense_aux_k=32,
        surprise_weight=0.5,
        prior_temp=1.2,
        deblunder_threshold=0.15, deblunder_width=0.15,
        kl_prior_weight=0.0, corpus_mix=0.0,   # NO tethers
        anneal_stages=0,                       # nothing to anneal
        ckpt_interval=5, ckpt_max_keep=3,
        ckpt_dir=f"results/jaxarimaa/{RUN}_ckpt",
        compile_cache_dir=f"{BUCKET}/compile-cache",
        arena_interval=10, arena_games=128, arena_threshold=0.55,
        ref_interval=4,                     # rung = planted imitation net:
        eval_max_steps=384,                 # absolute progress vs ~2250 scale
        eval_num_sims=32, eval_num_considered=16,
    ),
    features=FeaturesConfig(
        bf16=True, fast_search=True, playout_cap=True, symmetry_aug=True,
        arena_gating=True, moves_left_head=True,
        planes_frozen=True, planes_trap=True, planes_step_in_turn=True,
        planes_moved=True,
        # Stage-2 mechanism set:
        deblunder=True, dense_aux=True, prune_policy_targets=True,
        certification=True, truncation_draw=True,
        # explicitly OFF: resign (optima uses none; our resign scars agree),
        # adjudicate_truncation (superseded by truncation_draw)
    ),
)

train.train(cfg, out_path=f"results/jaxarimaa/{RUN}.pkl", eval_every=8,
            logdir=f"results/jaxarimaa/{RUN}_tb",
            profile_dir=f"/tmp/xla_{RUN}")  # iteration-2 trace -> optimization pass
