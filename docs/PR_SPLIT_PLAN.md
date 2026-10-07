# Splitting the work into publishable PRs

Everything is currently on `master` of the private arimaa repo. This is the plan
for carving out the parts that are useful to other people, in dependency order.

## Status

| piece | where it lives | state |
|---|---|---|
| mctx batched policy | `generic-account/mctx` branch `batched-gumbel-policy` | **pushed**, fork PR #1 open, 3 code commits + notes |
| arimaa JAX env | `jaxarimaa/{env,types,constants}.py` here | not yet carved out |
| GatherExpander finding | `jaxarimaa/env.py` + memory notes | not yet written up |
| negative results | `docs/` + memory | not yet written up |

## 1. mctx — `batched_gumbel_muzero_policy`

Furthest along. See `UPSTREAM_NOTES.md` in the mctx fork for dev setup, the
settled design decisions, and the remaining checklist. Headline: **3.70x faster
than `gumbel_muzero_policy`** on a v5e-4, measured.

Sequencing there: sign the Google CLA, open an *issue* before a PR (the
near-tie divergence is a maintainer judgement call), and consider landing the
`mask_invalid_actions` move as a separate small refactor PR first.

## 2. The Arimaa JAX environment — the strongest standalone contribution

`jaxarimaa/{env,types,constants}.py` is ~710 lines and depends on **nothing else
in the project** (`constants` imports nothing; `types` imports only `constants`;
`env` imports only those two). It is a complete, vectorized, jittable Arimaa
rules engine: 1393-action table, legality mask, step function with trap
resolution and freezing, Zobrist repetition handling, and an observation
encoder.

Two possible homes:

- **A PR to [pgx](https://github.com/sotetsuk/pgx)** (JAX board-game envs).
  Arimaa is not in their set. This is the natural fit and reaches the right
  audience. Needs adapting to their `Env` interface and their test conventions.
- **Its own small package** (`jaxarimaa-env`). Lower friction, no interface
  negotiation, but much less reach.

Either way the carve-out is the same work: lift those three files plus
`difftest.py` (the oracle differential test, 239 lines — strong evidence of
correctness and the thing that makes the contribution credible), write a README
with the action encoding, and drop the project-specific `FeaturesConfig` plumbing
in favour of plain arguments.

**Check first:** whether pgx's chess/shogi/go implementations already hit the
GatherExpander pathology below. If they do, that finding is worth more to them
than the Arimaa env is.

## 3. The GatherExpander finding — small, high-value, not Arimaa-specific

On TPU, `grid[clip(vy), clip(vx)]` at a table of **constant** coordinates is
rewritten by XLA's GatherExpander into a **serial loop with one iteration per
coordinate**. In our case that was 1393 iterations, and it was ~40% of all
device time — invisible to scope-level profiling, which attributed it to
"gather".

The fix is a constant one-hot matmul on the (otherwise idle) MXU, exact for
bools and small ints. In this repo that is `_gather_mm` in `jaxarimaa/env.py`,
and it produced **+73%** end-to-end.

Two writeups worth doing:

- **An XLA issue.** Silent loop-expansion of a tileable gather is a compiler
  pathology; a minimal repro would be short.
- **A pgx PR**, if their table-driven legality masks have the same shape. Verify
  before writing.

This is the most broadly useful thing the project produced and the least
entangled with everything else.

## 4. Negative results — a writeup, not code

Each of these cost real money to establish and would save someone else the same:

- **Certification deadlock.** Arena-gated self-play stalls when the search
  improvement step is smaller than the promotion margin — every loss looks
  healthy throughout. A 3-strike probation escape produced a +167 Elo jump in
  one cycle. (`jaxarimaa/train.py`, `probation_after`.)
- **The equal-strength value wall.** Training a *calibrated* value head on
  self-play outcomes destroys play, isolated with a frozen-trunk probe: trunk
  and policy verified bitwise unchanged, play still collapsed. The transmission
  path is search's Q-ranking, not the policy.
- **The frozen-component probe itself** is a reusable debugging method for
  AlphaZero-style systems: freeze trunk + policy, verify params are identical,
  and see whether play still degrades. If it does, the fault is in the target
  path.
- **Measured dead ends:** compact search (built, proven exact, 12% *slower*);
  batch-size scaling flat from 512 to 1024 games/chip, which refuted the
  dispatch-overhead hypothesis.

## Suggested order

1. **GatherExpander writeup** — smallest, most transferable, no dependencies.
2. **mctx issue** — opens the maintainer conversation early, since that is the
   long pole.
3. **Arimaa env** — the biggest carve-out; decide pgx vs standalone first.
4. **Negative results** — a blog post or a long issue; no deadline.
