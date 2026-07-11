"""Sims ladder arm: TRAIN at n=128/m=32, EVAL pinned at 32/16.

Fixed-anchor equilibrium sits ~-110+-25 regardless of refresh cycling — the
improvement operator looks target-quality-bound, and n=32 was chosen on a
PLAY-strength equivalence, never validated for TARGET quality. This arm
raises the full-search budget (fast moves stay 8 sims; only ~25% of steps
pay). Question: does the equilibrium band move UP?
"""

import sys

import jax

from jaxarimaa import train
from jaxarimaa.config import (Config, FeaturesConfig, MCTSConfig, NetConfig,
                              SelfPlayConfig, TrainConfig)

RUN = sys.argv[1] if len(sys.argv) > 1 else "sims128"
ITERS = int(sys.argv[2]) if len(sys.argv) > 2 else 400
BUCKET = "gs://arimaa-tpu-2026-artifacts"

# Scale the game batch with the slice: 1024 games PER CHIP keeps per-shard shapes
# (and HBM footprint) identical to the validated single-chip run.
PER_CHIP_GAMES = 512
N_CHIPS = len(jax.devices())

# Post-cold-start regime (n=32 sims: measured identical strength to n=64 at
# 1.75x the throughput). INIT_PARAMS must be the RE-GROUNDED checkpoint —
# pretrained policy + value/aux heads re-fit on sharp-annotated SELF-PLAY
# positions (gs://arimaa-tpu-2026-artifacts/regrounded_c256.pkl). Warm-starting
# from the plain pretrained pkl re-opens the OOD-value drift the re-grounding
# closed. The KL trust region anchors to whatever INIT_PARAMS loads.
INIT_PARAMS = f"results/jaxarimaa/{RUN}_init.pkl"  # per-run init convention: staged from $BUCKET/runs/<run>/init.pkl (segment 1b: seg-1a iter-155 weights with value/aux heads re-fit on their OWN self-play distribution)

cfg = Config(
    net=NetConfig(channels=256, blocks=15),
    mcts=MCTSConfig(num_simulations=128, max_num_considered_actions=32),
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
        # steps 16->64: the dose grid cleared up to ~110 iters' dose (flat
        # 0.43-0.52 across 27x) — 16 was calibrated on the old unprotected
        # recipe. 64 = 4x data utilization + 4x gradient throughput for ~+1s
        # on a ~104s iteration. Generation size: launch-time A/B (trace shows
        # the search scaffolding is dispatch-bound, so lane cuts may not pay).
        train_batch_size=1024, iterations=ITERS, train_steps_per_iter=64,
        replay_capacity=1048576, warmup_steps=100,
        lr=3e-4,  # warm-start LR, lowered again with the loop-gain fix (static
                  # policy drift per step scales with LR)
        max_steps_tiers=(256, 384, 512), completion_target=0.65,
        # Value-stream protection (probe-verified: value gradients through the
        # shared trunk were the warm-start poison; policy targets exonerated):
        value_loss_weight=0.25,   # damp trunk churn while value calibrates
        value_tail_weight=0.0,    # REAL outcomes only for the value loss
        # mix sweep: 75:25 is NET POSITIVE (0.527); 50:50 already collapses
        # (0.418); the old 0.2 was deep in the damage zone
        corpus_mix=0.75,
        corpus_path="results/archive_ds_sharp/year*.npz",
        # Trust ratchet: anneal the protections toward full AlphaZero
        # (value_w 0.25->1.0, tail_w 0->0.25, mix 0.75->0.25, kl 1.0->0) one
        # stage per healthy arena round; retreat on regression. Earliest full
        # anneal: anneal_stages * arena_interval iters.
        kl_prior_weight=1.0,       # trust region — held to the FIXED prior below
        anneal_kl_prior_weight=0.0,
        # Phase A hypothesis: the rolling anchor (leash to each launch's init)
        # stair-stepped the decline. Hold to the strong prior permanently.
        kl_anchor_path="results/jaxarimaa/regrounded_plus9_c256.pkl",
        anneal_stages=40,  # 4x smaller steps; full anneal ~iter 400+ (horizon scaled to run length; stage-1 window coincided with the rehearsal decline)
        anneal_value_loss_weight=1.0, anneal_value_tail_weight=0.25,
        anneal_corpus_mix=0.70,  # FLOOR: live bleed began as mix crossed ~0.65 (matches the static sweep); weaning below this waits for demonstrated self-improvement
        ckpt_interval=5, ckpt_max_keep=3,
        # Local Orbax dir: async sharded saves straight to gs:// time out on
        # multi-device meshes (orbax/gcsfs signaling); the stage runner rsyncs
        # this dir to GCS for durability and pulls it down on fresh VMs.
        ckpt_dir=f"results/jaxarimaa/{RUN}_ckpt",
        compile_cache_dir=f"{BUCKET}/compile-cache",
        # 128/color = 256 games: near-clone games correlate, 64-game rounds
        # were noisier than binomial and the ratchet bands assume sigma~30.
        # threshold 0.58 (~2.6 sigma): cuts false promotions ~25x — the chained
        # elo/estimate inflates by ~+35 per false promotion and never reverts.
        # ref_interval=4: every 4th arena also plays the FROZEN reference rung
        # -> elo/vs_ref, the UNBIASED headline metric for run health.
        arena_interval=5, arena_games=128, arena_threshold=0.58,
        ref_interval=2,
        # eval pinned at the historical shape: readings measure the NET, and
        # stay comparable with every plus9-scale number in the ledger
        eval_num_sims=32, eval_num_considered=16,
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
