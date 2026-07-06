"""Hardware-utilization tooling: measured-FLOPs MFU + XLA profiler traces.

Two instruments, both cheap:

1. **MFU** — at startup we get the network's *actual* forward FLOPs from XLA
   (`.lower().compile().cost_analysis()`), build a per-iteration FLOP estimate
   (self-play: (sims+1) batched forwards per move-step; training: ~3x forward per
   sample for fwd+bwd), and each iteration divide by measured wall-clock and the
   devices' peak to get achieved TFLOP/s and MFU. Logged as `perf/*` metrics.
   The estimate covers matmul work only (env-step / tree ops excluded), so MFU
   here is a *lower bound* on useful-work fraction — exactly the number that
   tells us whether scaling is justified.

2. **Profiler** — `--profile-dir` captures an XLA trace of one full iteration
   (self-play + train) after compilation has settled. View in TensorBoard's
   profile plugin or Perfetto to see op-level device occupancy and gaps.
"""

import jax
import jax.numpy as jnp

# Peak bf16 FLOP/s per chip (dense). Matched by substring against device_kind.
PEAK_FLOPS = {
    "v5 lite": 197e12,   # v5e ("TPU v5 lite" / "TPU v5e")
    "v5e": 197e12,
    "v5p": 459e12,
    "v5": 459e12,        # after v5e/v5 lite checks
    "v4": 275e12,
    "v6e": 918e12,       # Trillium
    "v3": 123e12,
    "v2": 45e12,
}


def device_peak_flops():
    """Peak FLOP/s of one device, or None if unknown (e.g. CPU)."""
    kind = jax.devices()[0].device_kind.lower()
    for key, peak in PEAK_FLOPS.items():
        if key in kind:
            return peak
    return None


def forward_flops(model, params, obs_shape, batch):
    """XLA-measured FLOPs of one batched forward pass (batch x obs_shape)."""
    dummy = jnp.zeros((batch,) + obs_shape, jnp.float32)
    f = jax.jit(jax.vmap(lambda o: model.apply(params, o)))
    cost = f.lower(dummy).compile().cost_analysis()
    if isinstance(cost, list):  # older jax returns a list per computation
        cost = cost[0]
    return float(cost.get("flops", 0.0)) if cost else 0.0


class MFUMeter:
    """Per-iteration MFU/TFLOPs from measured forward FLOPs + config constants."""

    def __init__(self, model, params, obs_shape, cfg):
        sp, tr, mc = cfg.selfplay, cfg.train, cfg.mcts
        fwd_sp = forward_flops(model, params, obs_shape, sp.batch_size)
        fwd_tr = forward_flops(model, params, obs_shape, tr.train_batch_size)
        # Self-play: per move-step, one root eval + num_simulations recurrent evals.
        # With playout_cap, only `full_search_prob` of steps use full sims; the rest
        # use fast_sims — account for the mixture or MFU is overstated ~2x.
        if cfg.features.playout_cap:
            per_step = (sp.full_search_prob * (mc.num_simulations + 1)
                        + (1 - sp.full_search_prob) * (sp.fast_sims + 1))
        else:
            per_step = mc.num_simulations + 1
        self.selfplay_flops = fwd_sp * per_step * sp.max_steps
        # Training: fwd+bwd ~= 3x forward, per gradient step.
        self.train_flops = fwd_tr * 3.0 * tr.train_steps_per_iter
        self.peak = device_peak_flops()
        self.n_dev = len(jax.devices())
        self.fwd_flops_selfplay_batch = fwd_sp

    def metrics(self, sp_seconds, tr_seconds, trained: bool):
        total = self.selfplay_flops + (self.train_flops if trained else 0.0)
        wall = max(sp_seconds + tr_seconds, 1e-9)
        achieved = total / wall
        m = {
            "perf/achieved_tflops": achieved / 1e12,
            "perf/selfplay_tflops": self.selfplay_flops / max(sp_seconds, 1e-9) / 1e12,
        }
        if self.peak:
            m["perf/mfu"] = achieved / (self.peak * self.n_dev)
        return m


class IterationProfiler:
    """Capture an XLA trace of exactly one iteration (after warmup) to `logdir`."""

    def __init__(self, logdir, capture_iter=2):
        self.logdir = logdir
        self.capture_iter = capture_iter
        self._active = False

    def maybe_start(self, it):
        if self.logdir and it == self.capture_iter:
            jax.profiler.start_trace(self.logdir)
            self._active = True

    def maybe_stop(self, it):
        if self._active and it == self.capture_iter:
            jax.profiler.stop_trace()
            self._active = False
            print(f"[perf] XLA trace for iter {it} written to {self.logdir}")
