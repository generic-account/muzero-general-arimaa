"""End-to-end AlphaZero training loop for jaxarimaa.

Per iteration: run vectorized self-play (data-parallel across the device mesh),
add samples to the replay buffer, take several sharded gradient steps, and
periodically checkpoint + evaluate vs a random opponent.

Run a CPU smoke test:
    python -m jaxarimaa.train --tiny
"""

import argparse
import math
import os
import time

import jax
import jax.numpy as jnp

from . import (checkpoint, checkpointing, distributed, env as jenv, evaluate,
               metrics, perf, selfplay, trainer)
from .config import Config, tiny_config, tiny_transformer_config


def train(cfg: Config, out_path="results/jaxarimaa/model.pkl", eval_every=1,
          verbose=True, logdir=None, use_wandb=False, profile_dir=None,
          init_params=None):
    tc = cfg.train
    distributed.enable_compilation_cache(tc.compile_cache_dir)  # before any jit
    distributed.init_distributed(tc.multihost)   # no-op single-host
    mesh = distributed.make_mesh()
    if verbose:
        print(f"devices: {len(jax.devices())} | hosts: {jax.process_count()} | "
              f"mesh: {mesh.shape['data']} | backbone={cfg.net.backbone} "
              f"ch={cfg.net.channels} blocks={cfg.net.blocks}")

    logger = metrics.Logger(logdir=logdir, use_wandb=use_wandb, config=cfg.to_dict())
    key = jax.random.PRNGKey(tc.seed)
    key, kinit = jax.random.split(key)
    state = trainer.create_train_state(cfg, kinit)
    # Warm-start from a pretrained (imitation/distillation) checkpoint — the
    # post-cold-start replacement for random init. Orbax resume (below) still
    # takes precedence, so a preempted run continues from its own progress.
    if init_params:
        pre, _ = checkpoint.load(init_params)
        state = state.replace(params=pre)
        if verbose:
            print(f"warm-started params from {init_params}")
    state = distributed.replicate_tree(mesh, state)

    # Preemption-safe checkpointing: restore full state (params+opt+step) if present.
    ckpt_mgr, start_it = None, 0
    if tc.ckpt_interval:
        ckpt_dir = tc.ckpt_dir or os.path.join(os.path.dirname(out_path) or ".",
                                               "checkpoints")
        ckpt_mgr = checkpointing.CheckpointManager(ckpt_dir, tc.ckpt_interval,
                                                   tc.ckpt_max_keep)
        state, start_it = ckpt_mgr.maybe_restore(state)
        if start_it and verbose:
            print(f"resumed from checkpoint at iteration {start_it}")
        key = jax.random.fold_in(key, start_it)  # don't replay self-play after resume

    from .replay import DeviceReplay
    buf = DeviceReplay(mesh, tc.replay_capacity)
    model = trainer.make_model(cfg)
    feats = cfg.features
    active = [k for k, v in vars(feats).items() if v]
    if verbose:
        print(f"features: {active or 'baseline (none)'}")
    mcts = (cfg.mcts.num_simulations, cfg.mcts.max_num_considered_actions)
    sp_knobs = selfplay.SPKnobs(
        resign_thresh=cfg.selfplay.resign_threshold,
        full_prob=cfg.selfplay.full_search_prob,
        fast_sims=cfg.selfplay.fast_sims,
        greedy_after=cfg.selfplay.greedy_after_turns)
    # Adaptive game length: if tiers are configured, hop between them to keep the
    # game-completion fraction in a target band as the bot's play-length drifts
    # (each tier is a separate compile, cached after first use).
    tiers = list(tc.max_steps_tiers or (cfg.selfplay.max_steps,))
    if cfg.selfplay.max_steps not in tiers:
        tiers.append(cfg.selfplay.max_steps)
    tiers.sort()
    tier_ix = tiers.index(cfg.selfplay.max_steps)
    gen_cache = {}

    def get_generate(T):
        if T not in gen_cache:
            gen_cache[T] = selfplay.make_generate(mesh, model,
                                                  cfg.selfplay.batch_size, T,
                                                  mcts, feats, sp_knobs)
        return gen_cache[T]

    generate = get_generate(tiers[tier_ix])
    # Arena as an ELO METRIC (not a data gate): self-play ALWAYS uses the learner —
    # gating self-play data on a champion starved the learner of on-policy data when
    # arena samples were noisy (observed in the first long run). Instead we keep a
    # frozen ANCHOR; every arena_interval we play learner-vs-anchor, count unfinished
    # games as draws, convert the score to an Elo delta, and re-freeze the anchor
    # when the learner clearly passes it — producing a chained elo/estimate curve.
    anchor = state.params if feats.arena_gating else None
    anchor_elo = 0.0

    # Global per-iteration work (all devices/hosts): games and env-steps generated.
    games_per_iter = cfg.selfplay.batch_size
    samples_per_train = tc.train_batch_size * tc.train_steps_per_iter
    games_total = 0

    # Hardware-utilization instruments: measured-FLOPs MFU + optional XLA trace.
    obs_shape = jenv.observe(jenv.init_state(jax.random.PRNGKey(0)), feats).shape
    meter = perf.MFUMeter(model, state.params, obs_shape, cfg)
    profiler = perf.IterationProfiler(profile_dir)
    if verbose and meter.peak:
        print(f"perf: fwd={meter.fwd_flops_selfplay_batch/1e9:.2f} GFLOP/batch-fwd | "
              f"peak {meter.peak * meter.n_dev / 1e12:.0f} TFLOP/s over {meter.n_dev} dev")

    for it in range(start_it, tc.iterations):
        profiler.maybe_start(it)
        # --- self-play (each device plays distinct games; output sharded) ---
        t0 = time.time()
        key, ksp = jax.random.split(key)
        recs, completed_frac = generate(state.params, ksp)  # learner's params
        jax.block_until_ready(recs)                # settle async dispatch before timing
        sp_t = time.time() - t0
        cur_T = tiers[tier_ix]
        if len(tiers) > 1:  # completion-band controller (hysteresis both ways)
            if completed_frac < tc.completion_target and tier_ix < len(tiers) - 1:
                tier_ix += 1
                print(f"[adapt] completion {completed_frac:.2f} < "
                      f"{tc.completion_target:.2f}: max_steps {cur_T} -> {tiers[tier_ix]}")
                generate = get_generate(tiers[tier_ix])
            elif completed_frac > 0.95 and tier_ix > 0:
                tier_ix -= 1
                print(f"[adapt] completion {completed_frac:.2f} > 0.95: "
                      f"max_steps {cur_T} -> {tiers[tier_ix]}")
                generate = get_generate(tiers[tier_ix])
        flat = selfplay.flatten_samples(recs)
        buf.add(flat)
        # value-target magnitude: rises toward 1 as games actually finish (health signal)
        vt_absmean = float(jnp.mean(jnp.abs(flat["value_target"])))

        # --- training steps (sample straight from the on-device sharded buffer) ---
        t1 = time.time()
        last = {}
        if buf.size >= tc.min_replay_size:
            for _ in range(tc.train_steps_per_iter):
                if feats.symmetry_aug:
                    key, ksmp, kaug = jax.random.split(key, 3)
                else:
                    key, ksmp = jax.random.split(key, 2)  # no extra draw when disabled
                    kaug = ksmp  # unused by train_step when symmetry is off
                batch = buf.sample(ksmp, tc.train_batch_size)
                state, last = trainer.train_step(
                    state, batch, tc.value_loss_weight, kaug, feats.symmetry_aug,
                    (tc.moves_left_weight, tc.deep_supervision_weight, tc.mtp_weight))
            jax.block_until_ready(state.params)
        tr_t = time.time() - t1
        games_total += games_per_iter

        profiler.maybe_stop(it)
        env_steps_per_iter = games_per_iter * cur_T
        m = {
            "throughput/games_per_s": games_per_iter / max(sp_t, 1e-9),
            "throughput/env_steps_per_s": env_steps_per_iter / max(sp_t, 1e-9),
            "selfplay/completion": completed_frac,
            "selfplay/max_steps": cur_T,
            "throughput/train_samples_per_s": (samples_per_train / max(tr_t, 1e-9)) if last else 0.0,
            "throughput/selfplay_s": sp_t,
            "throughput/train_s": tr_t,
            "counters/games_total": games_total,
            "counters/buffer_size": buf.total_size,
            "selfplay/value_target_absmean": vt_absmean,
        }
        m.update(meter.metrics(sp_t, tr_t, trained=bool(last),
                               t_scale=cur_T / cfg.selfplay.max_steps))
        if last:
            m["loss/total"] = float(last["loss"])
            m["loss/policy"] = float(last["policy_loss"])
            m["loss/value"] = float(last["value_loss"])
        logger.write(it, m)

        if verbose:
            loss = float(last["loss"]) if last else float("nan")
            mfu = f" mfu={m['perf/mfu']*100:.1f}%" if "perf/mfu" in m else ""
            print(f"[iter {it:03d}] games={games_total:>7d} buf={buf.total_size:>7d} "
                  f"loss={loss:.3f} | {m['throughput/games_per_s']:.1f} games/s "
                  f"{m['throughput/env_steps_per_s']:.0f} env-steps/s "
                  f"{m['perf/achieved_tflops']:.2f} TFLOP/s{mfu} | "
                  f"sp {sp_t:.1f}s tr {tr_t:.1f}s")

        if eval_every and (it + 1) % eval_every == 0:
            key, ke = jax.random.split(key)
            w, l, u = evaluate.play_vs_random(
                model, state.params, ke, our_color=0,
                n_games=min(cfg.selfplay.batch_size, 512),
                max_steps=tc.eval_max_steps or cfg.selfplay.max_steps,
                num_sims=min(cfg.mcts.num_simulations, 32),  # vs random: 32 plenty
                max_considered=cfg.mcts.max_num_considered_actions, features=feats,
                fast=feats.fast_search)
            w, l, u = int(w), int(l), int(u)
            decided = w + l
            logger.write(it, {
                "eval/wins": w, "eval/losses": l, "eval/unfinished": u,
                "eval/win_rate": (w / decided) if decided else 0.0,
            })
            if verbose:
                print(f"          eval vs random (gold): W{w} L{l} unfinished{u}")

        if feats.arena_gating and (it + 1) % tc.arena_interval == 0:
            key, ka1, ka2 = jax.random.split(key, 3)
            ns, nc = cfg.mcts.num_simulations, cfg.mcts.max_num_considered_actions
            ms, g = (tc.eval_max_steps or cfg.selfplay.max_steps), tc.arena_games
            a1, b1, u1 = evaluate.play_match(model, state.params, anchor, ka1, 0,
                                             g, ms, ns, nc, feats,
                                             feats.fast_search)  # learner = gold
            a2, b2, u2 = evaluate.play_match(model, anchor, state.params, ka2, 0,
                                             g, ms, ns, nc, feats,
                                             feats.fast_search)  # learner = silver
            wins = int(a1) + int(b2)
            losses = int(b1) + int(a2)
            draws = int(u1) + int(u2)  # unfinished games count as draws
            total = wins + losses + draws
            score = (wins + 0.5 * draws) / max(total, 1)
            sc = min(max(score, 0.01), 0.99)
            elo_est = anchor_elo + 400.0 * math.log10(sc / (1.0 - sc))
            promoted = score > tc.arena_threshold
            if promoted:  # learner clearly past the anchor: re-freeze the chain here
                anchor = state.params
                anchor_elo = elo_est
            logger.write(it, {"arena/score": score, "arena/decided": wins + losses,
                              "arena/promoted": float(promoted),
                              "elo/estimate": elo_est, "elo/anchor": anchor_elo})
            if verbose:
                print(f"          arena: score {score:.2f} (W{wins} L{losses} D{draws})"
                      f" -> elo~{elo_est:+.0f}{' [anchor re-frozen]' if promoted else ''}")

        if ckpt_mgr:
            ckpt_mgr.save(it, state)  # periodic; Orbax gates by save-interval
            if (it + 1) % tc.ckpt_interval == 0:
                # Also refresh the small portable weights pickle so current
                # strength can be evaluated (e.g. on the AEI ladder) mid-run.
                os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
                checkpoint.save(out_path, state.params,
                                {"config": cfg.to_dict(), "steps": it + 1})

    if ckpt_mgr:
        ckpt_mgr.close()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    checkpoint.save(out_path, state.params,
                    {"config": cfg.to_dict(), "steps": tc.iterations})
    logger.close()
    if verbose:
        print(f"saved checkpoint -> {out_path}")
    return state


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiny", action="store_true", help="CPU smoke-test config")
    ap.add_argument("--transformer", action="store_true",
                    help="use the transformer backbone (with --tiny)")
    ap.add_argument("--out", default="results/jaxarimaa/model.pkl")
    ap.add_argument("--logdir", default=None,
                    help="TensorBoard logdir (local path or gs://bucket/run)")
    ap.add_argument("--wandb", action="store_true", help="stream metrics to W&B")
    ap.add_argument("--multihost", action="store_true",
                    help="call jax.distributed.initialize() (multi-host slices)")
    ap.add_argument("--ckpt-interval", type=int, default=0,
                    help="iters between preemption-safe Orbax checkpoints (0=off)")
    ap.add_argument("--ckpt-dir", default=None,
                    help="durable checkpoint dir (gs://... for spot); default local")
    ap.add_argument("--compile-cache", default=None,
                    help="persistent XLA compilation cache dir (gs://... or local)")
    ap.add_argument("--profile-dir", default=None,
                    help="capture an XLA trace of one iteration to this dir")
    args = ap.parse_args()
    if args.transformer:
        cfg = tiny_transformer_config()
    elif args.tiny:
        cfg = tiny_config()
    else:
        cfg = Config()
    import dataclasses
    overrides = {}
    if args.multihost:
        overrides["multihost"] = True
    if args.ckpt_interval:
        overrides["ckpt_interval"] = args.ckpt_interval
    if args.ckpt_dir:
        overrides["ckpt_dir"] = args.ckpt_dir
    if args.compile_cache:
        overrides["compile_cache_dir"] = args.compile_cache
    if overrides:
        cfg = dataclasses.replace(cfg, train=dataclasses.replace(cfg.train, **overrides))
    train(cfg, out_path=args.out, logdir=args.logdir, use_wandb=args.wandb,
          profile_dir=args.profile_dir)


if __name__ == "__main__":
    main()
