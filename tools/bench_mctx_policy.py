"""Standalone mctx benchmark: `gumbel_muzero_policy` vs the batched policy.

Deliberately depends on mctx + jax only, so it can be attached to an upstream
issue/PR and run by anyone. Two knobs matter for interpreting the result:

  * `--rec-cost` sets how expensive `recurrent_fn` is. Cheap means tree ops
    dominate, which is the regime sensitive to tree-side changes. Expensive
    means the network dominates, which is the regime that shows the headline
    call-count win.
  * `--variants` selects which module provides `batched_gumbel_muzero_policy`,
    so two implementations can be compared in one process.

Protocol: one untimed call (compile + warmup), then `--reps` timed calls,
reporting the median. Also reports the number of `recurrent_fn` invocations,
which is the mechanism being exploited.

Usage:
  python tools/bench_mctx_policy.py --n 800 --m 32 --batch 256 --rec-cost cheap
"""
import argparse
import importlib
import time

import jax
import jax.numpy as jnp
import mctx


def make_recurrent_fn(num_actions, cost):
  """A deterministic recurrent_fn with a tunable amount of arithmetic."""
  layers = {"cheap": 0, "medium": 4, "expensive": 16}[cost]

  def recurrent_fn(params, rng_key, action, embedding):
    del rng_key
    x = embedding + jnp.sin(action.astype(embedding.dtype))[:, None]
    for w in params:  # a small MLP stack when `layers` > 0
      x = jnp.tanh(x @ w)
    total = jnp.sum(x, axis=-1)
    return mctx.RecurrentFnOutput(
        reward=jnp.tanh(total),
        discount=jnp.ones_like(total),
        prior_logits=jnp.stack([total] * num_actions, axis=-1),
        value=jnp.tanh(jnp.mean(x, axis=-1)),
    ), x

  return recurrent_fn, layers


def main():
  p = argparse.ArgumentParser()
  p.add_argument("--n", type=int, default=800, help="num_simulations")
  p.add_argument("--m", type=int, default=32, help="max_num_considered_actions")
  p.add_argument("--batch", type=int, default=256)
  p.add_argument("--num-actions", type=int, default=1393)
  p.add_argument("--embed", type=int, default=64)
  p.add_argument("--reps", type=int, default=3)
  p.add_argument("--rec-cost", default="cheap",
                 choices=["cheap", "medium", "expensive"])
  p.add_argument("--variants", nargs="+",
                 default=["mctx._src.policies_batched"],
                 help="modules exporting batched_gumbel_muzero_policy")
  p.add_argument("--skip-baseline", action="store_true")
  args = p.parse_args()

  key = jax.random.PRNGKey(0)
  k1, k2, k3, k4 = jax.random.split(key, 4)
  root = mctx.RootFnOutput(
      prior_logits=jax.random.normal(k1, (args.batch, args.num_actions)),
      value=jax.random.normal(k2, (args.batch,)),
      embedding=jax.random.normal(k3, (args.batch, args.embed)))
  rec, layers = make_recurrent_fn(args.num_actions, args.rec_cost)
  params = [jax.random.normal(jax.random.fold_in(k4, i),
                              (args.embed, args.embed)) / args.embed**0.5
            for i in range(layers)]

  print(f"shape: batch={args.batch} num_actions={args.num_actions} "
        f"n={args.n} m={args.m} embed={args.embed} "
        f"rec_cost={args.rec_cost}({layers} layers) devices={jax.device_count()}")

  # The mechanism: how many recurrent_fn calls each implementation makes.
  from mctx._src import policies_batched
  rounds = policies_batched._rounds_from_schedule(  # pylint: disable=protected-access
      min(args.m, args.num_actions), args.n)
  print(f"recurrent_fn calls: sequential={args.n}  batched={len(rounds)}  "
        f"({args.n / len(rounds):.1f}x fewer)\n")

  def timeit(label, fn):
    out = fn()
    jax.block_until_ready(out.action)
    t = []
    for _ in range(args.reps):
      t0 = time.perf_counter()
      out = fn()
      jax.block_until_ready(out.action)
      t.append(time.perf_counter() - t0)
    med = sorted(t)[len(t) // 2]
    print(f"  {label:<44} {med * 1e3:9.1f} ms   "
          f"(spread {max(t) / min(t) - 1:.1%})")
    return med

  common = dict(params=params, rng_key=jax.random.PRNGKey(7), root=root,
                recurrent_fn=rec, num_simulations=args.n,
                max_num_considered_actions=args.m)
  results = {}
  if not args.skip_baseline:
    f = jax.jit(lambda: mctx.gumbel_muzero_policy(**common))
    results["gumbel_muzero_policy"] = timeit("gumbel_muzero_policy", f)
  for mod_name in args.variants:
    mod = importlib.import_module(mod_name)
    f = jax.jit(lambda m=mod: m.batched_gumbel_muzero_policy(**common))
    results[mod_name] = timeit(f"batched [{mod_name.split('.')[-1]}]", f)

  base = results.get("gumbel_muzero_policy")
  if base:
    print()
    for k, v in results.items():
      if k != "gumbel_muzero_policy":
        print(f"  {k.split('.')[-1]:<30} {base / v:6.2f}x vs gumbel_muzero_policy")


if __name__ == "__main__":
  main()
