"""Warm-start x stage-2 validation, take 3: the WARM-START PROFILE.

ws2/ws3 post-mortems: the from-scratch-tuned stage-2 loop is actively
destructive for a strong prior — pruned policy targets delete prior mass on
unvisited actions and prior_temp>1 flattens a sharp prior; collapse to the
clamp in <=4 iterations, dense-head graft or not. This profile keeps the
stage-2 value machinery (deblunder, truncation draws, certification,
probation, dense aux zero-grafted) but uses Gumbel action_weights targets,
prior_temp=1.0, lr=1e-4.

Tested so far: warm-start + tethers (stage 1: fails, equilibrates below the
prior) and from-scratch + stage-2 mechanisms (s2pilot: climbs). This run
tests the big run's ACTUAL configuration: the C256x15 imitation net
(regrounded_plus9, ~2250 absolute) warm-started into the UNTETHERED stage-2
loop — no KL leash, no corpus mixing; certification + probation; deblunder;
dense aux (head fresh-grafted); truncation draws; prior flattening.

Free instrumentation from warm-starting: `anchor` (certified generator) and
`rung` (Elo-0 reference) both default to warm_init, so rung readings are a
direct signed delta vs the prior from the first cycle — the stage-1-redux
detector.

Pre-registered verdict (~40 iters, ~$5):
  * HEALTHY: rung stays >= -40 band / recovers and climbs; value_real_mse
    falls; some organic gate passes -> commit the fresh budget to this shape.
  * STAGE-1 REDUX: monotone rung decline below -80 with flat losses -> the
    untethered warm start also fails; fall back to value-damping variant or
    long from-scratch.

Run: bash infra/run_stage.sh <run> configs/warmstart2.py [iters]
"""
import sys

import jax

from jaxarimaa import train
from jaxarimaa.config import (Config, FeaturesConfig, MCTSConfig, NetConfig,
                              SelfPlayConfig, TrainConfig)

RUN = sys.argv[1] if len(sys.argv) > 1 else "ws4"
ITERS = int(sys.argv[2]) if len(sys.argv) > 2 else 40
BUCKET = "gs://arimaa-tpu-2026-artifacts"

PER_CHIP_GAMES = 512
N_CHIPS = len(jax.devices())

INIT_PARAMS = "results/jaxarimaa/regrounded_plus9_c256.pkl"

cfg = Config(
    net=NetConfig(channels=256, blocks=15),   # the imitation net's arch
    mcts=MCTSConfig(num_simulations=128, max_num_considered_actions=32),
    selfplay=SelfPlayConfig(
        batch_size=PER_CHIP_GAMES * N_CHIPS, max_steps=384,
        full_search_prob=0.25, fast_sims=16,
        greedy_after_turns=0,
    ),
    train=TrainConfig(
        train_batch_size=1024, iterations=ITERS, train_steps_per_iter=64,
        replay_capacity=1048576, warmup_steps=100,
        lr=1e-4,                            # warm profile: extra caution untethered
        max_steps_tiers=(256, 384, 512), completion_target=0.65,
        value_loss_weight=1.0, value_tail_weight=1.0,
        dense_aux_weight=0.3, dense_aux_k=32,
        surprise_weight=0.5,
        prior_temp=1.0,   # warm profile: flattening a sharp prior = active degradation (ws3)
        deblunder_threshold=0.15, deblunder_width=0.15,
        kl_prior_weight=0.0, corpus_mix=0.0,   # UNTETHERED
        anneal_stages=0,
        probation_after=3,
        ckpt_interval=5, ckpt_max_keep=3,
        ckpt_dir=f"results/jaxarimaa/{RUN}_ckpt",
        compile_cache_dir=f"{BUCKET}/compile-cache",
        arena_interval=5, arena_games=128, arena_threshold=0.55,
        ref_interval=2,                     # rung (== the prior) every 10 iters
        eval_max_steps=384,
        eval_num_sims=32, eval_num_considered=16,
    ),
    features=FeaturesConfig(
        bf16=True, fast_search=True, playout_cap=True, symmetry_aug=True,
        arena_gating=True, moves_left_head=True,
        planes_frozen=True, planes_trap=True, planes_step_in_turn=True,
        planes_moved=True,
        deblunder=True, dense_aux=True,
        # prune_policy_targets OFF for warm starts (ws3 post-mortem): pruning
        # deletes prior mass on the 1361 unvisited actions every update ->
        # catastrophic forgetting (-720 in 4 iters). Gumbel action_weights
        # impute prior+Q mass for unvisited actions - the designed target
        # when the value head is sound (which a regrounded warm start is).
        prune_policy_targets=False,
        certification=True, truncation_draw=True,
    ),
)

train.train(cfg, out_path=f"results/jaxarimaa/{RUN}.pkl", eval_every=8,
            logdir=f"results/jaxarimaa/{RUN}_tb", init_params=INIT_PARAMS)
