"""Self-play throughput benchmark — the canonical perf-engineering metric.

Methodology (docs in memory: xla-trace-s2pilot):
  * metric: env-steps/s PER CHIP at a pinned shape (default = the s2pilot
    production shape). Secondary: $/1M env-steps at the spot v5e price.
  * protocol: 1 untimed generate() (compile + warmup), then N timed calls,
    report each + median (generation is deterministic-shape, so variance is
    scheduler noise, typically <1%).
  * every optimization round must (a) keep the difftest + fast_search
    equivalence + selfplay determinism suites green (bit-exact play), then
    (b) move this number.

Usage (on a TPU VM, from the repo root):
  python tools/bench_selfplay.py                       # pilot shape
  python tools/bench_selfplay.py --profile /tmp/xla_bench   # + XLA trace
  python tools/bench_selfplay.py --reps 5 --batch 2048 --T 512
"""
import argparse
import dataclasses
import time

import jax

from jaxarimaa import distributed, selfplay, trainer
from jaxarimaa.config import (Config, FeaturesConfig, MCTSConfig, NetConfig,
                              SelfPlayConfig, TrainConfig)

SPOT_V5E4_USD_PER_H = 1.89  # us-east1 spot, 4 chips


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--batch", type=int, default=0, help="0 = 512 per chip")
    p.add_argument("--T", type=int, default=512)
    p.add_argument("--sims", type=int, default=128)
    p.add_argument("--m", type=int, default=32)
    p.add_argument("--full-prob", type=float, default=0.25)
    p.add_argument("--fast-sims", type=int, default=16)
    p.add_argument("--channels", type=int, default=128)
    p.add_argument("--blocks", type=int, default=10)
    p.add_argument("--reps", type=int, default=3)
    p.add_argument("--profile", default=None, help="XLA trace dir for rep 1")
    p.add_argument("--compact", action="store_true",
                   help="enable features.compact_search (v3 tree)")
    args = p.parse_args()

    n_chips = len(jax.devices())
    batch = args.batch or 512 * n_chips
    cfg = Config(
        net=NetConfig(channels=args.channels, blocks=args.blocks),
        mcts=MCTSConfig(num_simulations=args.sims,
                        max_num_considered_actions=args.m),
        selfplay=SelfPlayConfig(batch_size=batch, max_steps=args.T,
                                full_search_prob=args.full_prob,
                                fast_sims=args.fast_sims,
                                greedy_after_turns=0),
        train=TrainConfig(prior_temp=1.2),
        features=FeaturesConfig(
            bf16=True, fast_search=True, playout_cap=True, symmetry_aug=True,
            moves_left_head=True, dense_aux=True,
            planes_frozen=True, planes_trap=True, planes_step_in_turn=True,
            planes_moved=True,
            deblunder=True, prune_policy_targets=True, truncation_draw=True,
            compact_search=args.compact,
        ),
    )
    feats = cfg.features
    tc = cfg.train
    knobs = selfplay.SPKnobs(
        resign_thresh=0.0, full_prob=args.full_prob, fast_sims=args.fast_sims,
        greedy_after=0, dense_k=tc.dense_aux_k, surprise_w=tc.surprise_weight,
        prior_temp=tc.prior_temp, deblunder_threshold=tc.deblunder_threshold,
        deblunder_width=tc.deblunder_width)

    mesh = distributed.make_mesh()
    model = trainer.make_model(cfg)
    state = trainer.create_train_state(cfg, jax.random.PRNGKey(0))
    gen = selfplay.make_generate(mesh, model, batch, args.T,
                                 (args.sims, args.m), feats, knobs)

    print(f"bench shape: batch={batch} T={args.T} n={args.sims}/m={args.m} "
          f"full_prob={args.full_prob}/fast{args.fast_sims} "
          f"C{args.channels}x{args.blocks} chips={n_chips}")
    t0 = time.time()
    recs, _ = gen(state.params, jax.random.PRNGKey(1))
    jax.block_until_ready(recs)
    print(f"compile+first: {time.time() - t0:.1f}s")

    times = []
    for r in range(args.reps):
        if args.profile and r == 0:
            jax.profiler.start_trace(args.profile)
        t0 = time.time()
        recs, _ = gen(state.params, jax.random.PRNGKey(2 + r))
        jax.block_until_ready(recs)
        dt = time.time() - t0
        if args.profile and r == 0:
            jax.profiler.stop_trace()
        env_steps = batch * args.T
        eps = env_steps / dt
        times.append(dt)
        print(f"rep {r}: {dt:.1f}s  {eps:,.0f} env-steps/s "
              f"({eps / n_chips:,.0f}/chip)")
    med = sorted(times)[len(times) // 2]
    eps = batch * args.T / med
    usd_per_m = SPOT_V5E4_USD_PER_H / 3600.0 / eps * 1e6
    print(f"MEDIAN: {eps:,.0f} env-steps/s | {eps / n_chips:,.0f}/chip | "
          f"${usd_per_m:.3f}/1M env-steps")


if __name__ == "__main__":
    main()
