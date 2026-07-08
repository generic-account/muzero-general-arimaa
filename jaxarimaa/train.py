"""End-to-end AlphaZero training loop for jaxarimaa.

Per iteration: run vectorized self-play (data-parallel across the device mesh),
add samples to the replay buffer, take several sharded gradient steps, and
periodically checkpoint + evaluate vs a random opponent.

Run a CPU smoke test:
    python -m jaxarimaa.train --tiny
"""

import argparse
import json
import math
import os
import time

import jax
import jax.numpy as jnp

from . import (anneal, checkpoint, checkpointing, distributed, env as jenv,
               evaluate, metrics, perf, selfplay, trainer)
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
    # Frozen copy of the warm-start params, captured BEFORE Orbax resume so it
    # is identical across preemptions: used for the KL trust region and as the
    # default Elo reference rung.
    warm_init = state.params
    kl_anchor = warm_init if (init_params and tc.kl_prior_weight > 0) else None

    # Preemption-safe checkpointing: restore full state (params+opt+step) if present.
    ckpt_mgr, start_it, anneal_sidecar, anchor_pkl = None, 0, None, None
    if tc.ckpt_interval:
        ckpt_dir = tc.ckpt_dir or os.path.join(os.path.dirname(out_path) or ".",
                                               "checkpoints")
        ckpt_mgr = checkpointing.CheckpointManager(ckpt_dir, tc.ckpt_interval,
                                                   tc.ckpt_max_keep)
        if "://" not in str(ckpt_dir):  # remote dirs: no sidecars (state resets)
            anneal_sidecar = os.path.join(ckpt_dir, "anneal.json")
            anchor_pkl = os.path.join(ckpt_dir, "anchor.pkl")
        state, start_it = ckpt_mgr.maybe_restore(state)
        if start_it and verbose:
            print(f"resumed from checkpoint at iteration {start_it}")
        key = jax.random.fold_in(key, start_it)  # don't replay self-play after resume

    from .replay import DeviceReplay
    buf = DeviceReplay(mesh, tc.replay_capacity)
    corpus_sampler, corpus_rng = None, None
    if tc.corpus_mix > 0 and tc.corpus_path:
        from . import corpus as corpus_mod
        import numpy as _np
        import functools as _ft
        corpus_sampler = corpus_mod.CorpusSampler(tc.corpus_path, cfg.features)
        corpus_rng = _np.random.default_rng(tc.seed + 1)
        corpus_shard = _ft.partial(distributed.shard_batch, mesh)
        if verbose:
            print(f"corpus-mix: {tc.corpus_mix:.0%} of train steps from "
                  f"{corpus_sampler.n:,} expert positions")
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
    # The frozen anchor + its chained Elo survive preemption too (anchor.pkl in
    # the ckpt dir, rewritten at each promotion): without this, resume re-anchors
    # to the learner itself and a mid-regression preemption gets laundered.
    if anchor is not None and anchor_pkl and start_it and os.path.exists(anchor_pkl):
        a_params, a_meta = checkpoint.load(anchor_pkl)
        anchor = distributed.replicate_tree(mesh, a_params)
        anchor_elo = float(a_meta.get("elo", 0.0))
        if verbose:
            print(f"restored arena anchor (elo {anchor_elo:+.0f}) from {anchor_pkl}")

    # Fixed reference rung for UNBIASED Elo (see TrainConfig.ref_interval):
    # measurement decoupled from promotion gating. Survives preemption via
    # rung.pkl + sidecar fields.
    rung, rung_elo, arena_rounds = None, 0.0, 0
    rung_pkl = (os.path.join(ckpt_dir, "rung.pkl")
                if (anneal_sidecar and tc.ref_interval) else None)
    if feats.arena_gating and tc.ref_interval:
        rung = warm_init  # NOT the (possibly Orbax-restored) learner
        if rung_pkl and start_it and os.path.exists(rung_pkl):
            r_params, r_meta = checkpoint.load(rung_pkl)
            rung = distributed.replicate_tree(mesh, r_params)
            rung_elo = float(r_meta.get("elo", 0.0))
            if verbose:
                print(f"restored elo reference rung (elo {rung_elo:+.0f})")

    # Trust ratchet (anneal.TrustRatchet): walk the value-conservative warm-start
    # knobs toward full AlphaZero, gated on the anchor-Elo. Its FULL state
    # (stage + health baseline) survives preemption via a sidecar JSON next to
    # the Orbax checkpoints.
    side = {}
    if anneal_sidecar and start_it and os.path.exists(anneal_sidecar):
        with open(anneal_sidecar) as f:
            side = json.load(f)
        if verbose:
            print(f"restored anneal state {side} from {anneal_sidecar}")
    ratchet = anneal.TrustRatchet(tc, stage=int(side.get("stage", 0)),
                                  best=side.get("best"), ema=side.get("ema"))
    value_w, value_tail_w, cur_mix, kl_w = ratchet.knobs()
    if "tier_ix" in side:  # adaptive max_steps tier survives preemption too
        tier_ix = min(int(side["tier_ix"]), len(tiers) - 1)
        generate = get_generate(tiers[tier_ix])
    arena_rounds = int(side.get("arena_rounds", 0))

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
                if (corpus_sampler is not None
                        and corpus_rng.random() < cur_mix):
                    batch = corpus_sampler.sample(corpus_rng, tc.train_batch_size,
                                                  shard_fn=corpus_shard)
                else:
                    batch = buf.sample(ksmp, tc.train_batch_size)
                state, last = trainer.train_step(
                    state, batch, value_w, kaug, feats.symmetry_aug,
                    (tc.moves_left_weight, tc.deep_supervision_weight, tc.mtp_weight),
                    tc.policy_loss_weight, value_tail_w, kl_anchor, kl_w)
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
            if "value_real_mse" in last:  # undiluted value-head health signal
                m["loss/value_real_mse"] = float(last["value_real_mse"])
                m["loss/value_real_frac"] = float(last["value_real_frac"])
            if "kl_prior" in last:  # drift from the pretrained prior
                m["loss/kl_prior"] = float(last["kl_prior"])
        logger.write(it, m)

        if verbose:
            loss = float(last["loss"]) if last else float("nan")
            mfu = f" mfu={m['perf/mfu']*100:.1f}%" if "perf/mfu" in m else ""
            print(f"[iter {it:03d}] games={games_total:>7d} buf={buf.total_size:>7d} "
                  f"loss={loss:.3f} compl={completed_frac:.2f} T={cur_T} | "
                  f"{m['throughput/games_per_s']:.1f} games/s "
                  f"{m['throughput/env_steps_per_s']:.0f} env-steps/s "
                  f"{m['perf/achieved_tflops']:.2f} TFLOP/s{mfu} | "
                  f"sp {sp_t:.1f}s tr {tr_t:.1f}s")

        if eval_every and (it + 1) % eval_every == 0:
            key, ke = jax.random.split(key)
            # Health signal only (saturates near 100% quickly): small and cheap —
            # at 512 games x 32 sims this was ~7% of self-play compute amortized.
            w, l, u = evaluate.play_vs_random(
                model, state.params, ke, our_color=0,
                n_games=min(cfg.selfplay.batch_size, 128),
                max_steps=tc.eval_max_steps or cfg.selfplay.max_steps,
                num_sims=min(cfg.mcts.num_simulations, 16),
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
                if anchor_pkl:  # keep the chain durable across preemptions
                    checkpoint.save(anchor_pkl + ".tmp", anchor,
                                    {"elo": anchor_elo})
                    os.replace(anchor_pkl + ".tmp", anchor_pkl)
            logger.write(it, {"arena/score": score, "arena/decided": wins + losses,
                              "arena/promoted": float(promoted),
                              "elo/estimate": elo_est, "elo/anchor": anchor_elo})
            if verbose:
                print(f"          arena: score {score:.2f} (W{wins} L{losses} D{draws})"
                      f" -> elo~{elo_est:+.0f}{' [anchor re-frozen]' if promoted else ''}")
            if tc.anneal_stages:
                prev = ratchet.stage
                if ratchet.update(elo_est, score=score):
                    value_w, value_tail_w, cur_mix, kl_w = ratchet.knobs()
                    if verbose:
                        print(f"          [anneal] stage {prev}->{ratchet.stage}"
                              f"/{tc.anneal_stages}: value_w={value_w:.3f} "
                              f"tail_w={value_tail_w:.3f} corpus_mix={cur_mix:.3f} "
                              f"kl_w={kl_w:.3f}")
                logger.write(it, {"anneal/stage": ratchet.stage,
                                  "anneal/value_weight": value_w,
                                  "anneal/value_tail_weight": value_tail_w,
                                  "anneal/corpus_mix": cur_mix,
                                  "anneal/kl_weight": kl_w})
            arena_rounds += 1
            if rung is not None and arena_rounds % tc.ref_interval == 0:
                # Unbiased Elo: fresh match vs the frozen rung — nothing gates
                # on this reading, so it has no promotion-selection bias.
                key, kr1, kr2 = jax.random.split(key, 3)
                ra, rb, ru = evaluate.play_match(model, state.params, rung, kr1,
                                                 0, g, ms, ns, nc, feats,
                                                 feats.fast_search)
                ra2, rb2, ru2 = evaluate.play_match(model, rung, state.params,
                                                    kr2, 0, g, ms, ns, nc,
                                                    feats, feats.fast_search)
                rw, rl = int(ra) + int(rb2), int(rb) + int(ra2)
                rd = int(ru) + int(ru2)
                rs = (rw + 0.5 * rd) / max(rw + rl + rd, 1)
                rsc = min(max(rs, 0.01), 0.99)
                elo_ref = rung_elo + 400.0 * math.log10(rsc / (1.0 - rsc))
                logger.write(it, {"elo/vs_ref": elo_ref, "arena/ref_score": rs})
                if verbose:
                    print(f"          ref: score {rs:.2f} -> elo_ref {elo_ref:+.0f}")
                if rs > 0.95:
                    # Rung saturated (Elo resolution dies near 1.0): freeze a
                    # new rung; calibrate the gap with a DEDICATED fresh match
                    # (the re-rung decision selected on rs, this sample doesn't).
                    new_rung = state.params
                    key, kc1, kc2 = jax.random.split(key, 3)
                    ca, cb, cu = evaluate.play_match(model, new_rung, rung, kc1,
                                                     0, g, ms, ns, nc, feats,
                                                     feats.fast_search)
                    ca2, cb2, cu2 = evaluate.play_match(model, rung, new_rung,
                                                        kc2, 0, g, ms, ns, nc,
                                                        feats, feats.fast_search)
                    cw, cl = int(ca) + int(cb2), int(cb) + int(ca2)
                    cd = int(cu) + int(cu2)
                    cs = min(max((cw + 0.5 * cd) / max(cw + cl + cd, 1),
                                 0.01), 0.99)
                    rung_elo += 400.0 * math.log10(cs / (1.0 - cs))
                    rung = new_rung
                    if rung_pkl:
                        checkpoint.save(rung_pkl + ".tmp", rung,
                                        {"elo": rung_elo})
                        os.replace(rung_pkl + ".tmp", rung_pkl)
                    if verbose:
                        print(f"          [rung] new reference frozen at "
                              f"elo {rung_elo:+.0f}")

        if ckpt_mgr:
            saved = ckpt_mgr.save(it, state)  # periodic; Orbax gates by save-interval
            if saved and anneal_sidecar:  # persist the ratchet state in lockstep
                tmp = anneal_sidecar + ".tmp"  # atomic: a preemption (or the
                with open(tmp, "w") as f:      # GCS mirror) must never see a
                    json.dump({"it": it, "stage": ratchet.stage,   # torn file
                               "best": ratchet.best, "ema": ratchet.ema,
                               "tier_ix": tier_ix,
                               "arena_rounds": arena_rounds}, f)
                os.replace(tmp, anneal_sidecar)
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
