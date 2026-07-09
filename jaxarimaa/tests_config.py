"""Config lint: the big-run config must stay inside MEASURED-safe territory.

Every assertion here encodes an experimental result (mix sweep, ratchet
noise calibration, warm-start investigation). If a config edit trips one,
either the edit is a mistake or new evidence justifies changing the TEST.

Run:  PYTHONPATH=. python jaxarimaa/tests_config.py
"""
import ast
import re
import sys
import types

import jax


def load_cfg(path):
    """Extract the Config(...) from a config script without running train()."""
    src = open(path).read()
    tree = ast.parse(src)
    # drop the trailing train.train(...) call
    tree.body = [n for n in tree.body
                 if not (isinstance(n, ast.Expr)
                         and isinstance(n.value, ast.Call)
                         and getattr(getattr(n.value.func, "value", None),
                                     "id", "") == "train")]
    mod = types.ModuleType("cfgmod")
    mod.__dict__["__name__"] = "cfgmod"
    sys.argv = [path]
    exec(compile(tree, path, "exec"), mod.__dict__)
    return mod


def check(path):
    m = load_cfg(path)
    cfg, tc = m.cfg, m.cfg.train

    # -- warm-start init: must be the re-grounded checkpoint, and KL must anchor it
    init = getattr(m, "INIT_PARAMS", None) or "regrounded"
    assert ("regrounded" in str(init)) or ("_init.pkl" in str(init)), \
        f"{path}: INIT_PARAMS must be a re-grounded/per-run staged ckpt (got {init})"
    assert tc.kl_prior_weight > 0, f"{path}: KL trust region is OFF"

    # -- corpus mix: measured positive at 0.75, collapsed by 0.50; live bleed
    #    began as mix crossed ~0.65 -> start and anneal floor stay above it
    assert tc.corpus_mix >= 0.70, f"{path}: corpus_mix {tc.corpus_mix} below safe zone"
    assert tc.anneal_corpus_mix >= 0.65, \
        f"{path}: mix anneal floor {tc.anneal_corpus_mix} enters the collapse zone"
    assert tc.corpus_path, f"{path}: corpus_path unset (mix would silently no-op)"

    # -- value-stream protections (probe-verified)
    assert tc.value_loss_weight <= 0.5 and tc.value_tail_weight == 0.0, \
        f"{path}: value protections weakened"

    # -- ratchet noise calibration: hold/backoff bands assume sigma~30 Elo,
    #    which needs >=128 games/color at these thresholds; slow-bleed guard on
    assert tc.arena_games >= 128, f"{path}: arena_games {tc.arena_games} too noisy"
    assert tc.anneal_score_floor > 0, f"{path}: slow-bleed guard disabled"
    assert tc.arena_threshold >= 0.58, \
        f"{path}: promotion threshold {tc.arena_threshold} admits noise promotions"
    assert tc.ref_interval > 0, f"{path}: unbiased elo/vs_ref disabled"

    # -- loop-gain control (4-chip buffer-depth fix)
    ndev = len(jax.devices())
    rows_per_iter = round(0.25 * 512) * cfg.selfplay.batch_size  # T=512 tier
    assert tc.replay_capacity >= 3 * rows_per_iter, \
        (f"{path}: replay {tc.replay_capacity} is <3 iters deep "
         f"({rows_per_iter}/iter at {ndev} chips) — the silent-scaling trap")
    # dose grid (2026-07-08): flat 0.43-0.52 through 1728 steps (~110 iters'
    # dose); 192 keeps a 3x margin below the highest fully-clean point
    assert tc.train_steps_per_iter <= 192, f"{path}: dose beyond grid-cleared envelope"

    # -- preemption safety
    assert tc.ckpt_interval and tc.ckpt_dir and "://" not in tc.ckpt_dir, \
        f"{path}: local Orbax ckpts required (sidecars + run_stage mirror)"
    assert cfg.features.arena_gating, f"{path}: ratchet needs arena_gating"
    print(f"ok {path}")


if __name__ == "__main__":
    for p in ("configs/stage_a.py", "configs/stage_a1b.py", "configs/confirm_reground.py"):
        check(p)
    print("config lint passed")
