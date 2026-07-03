# Project Scope: JAX / mctx max-throughput Arimaa AlphaZero

Status: proposed scope (2026-06-30). Supersedes the muzero-general training core; **keeps**
`games/arimaa.py` (as a reference oracle) and the entire AEI benchmark harness.

## 1. Goal & guiding principles

Build a **maximum-throughput, JAX-native AlphaZero** for Arimaa on top of DeepMind's
**mctx** (batched on-device MCTS), with a **vectorized JAX game engine** so thousands of
self-play games run in parallel on one accelerator.

Three non-negotiable design properties:

1. **Max GPU/TPU throughput** — vectorized env (`jax.vmap`/`jit`) + batched mctx search +
   `pmap` multi-device self-play & training. This is the whole reason for the rebase.
2. **optima insights are pluggable** — richer input planes, SE blocks, moves-left/aux heads,
   playout-cap randomization, arena gating, symmetry augmentation, transposition-friendly
   search. Each is an opt-in module, not a fork.
3. **Swappable backbones** — the net is `backbone(obs) -> features`, then shared heads. A
   registry maps a name (`resnet`, `se_resnet`, `transformer`, ...) to a module so we can A/B
   architectures with a config flag.

AZ-first (we have a cheap exact simulator, so ground-truth search is our strength), but the
mctx wiring stays **MuZero-capable** so the learned-model bet remains testable in the same
codebase.

## 2. Reference stack (all Apache-2.0 / MIT — commercial-friendly)

| Component | Repo | Role | License |
|---|---|---|---|
| **mctx** | google-deepmind/mctx | Batched MCTS: `muzero_policy`, `gumbel_muzero_policy`, low-level `search` | Apache-2.0 |
| **pgx** | sotetsuk/pgx | Vectorized-env *interface* + canonical AlphaZero example (Haiku+optax+mctx). Arimaa NOT included → we write it. | Apache-2.0 |
| **a0-jax** | NTT123/a0-jax | Self-contained AZ-in-JAX (mctx), pluggable net via `--agent-class`. Cleaner minimal template than pgx's in-tree example. | MIT |
| **muax** | (ref) | MuZero-in-JAX reference for the learned-model mode | ref |

mctx contract we must satisfy:
- `RootFnOutput(prior_logits, value, embedding)` — embedding = our env state (AZ) or latent (MuZero).
- `recurrent_fn(params, rng, action, embedding) -> (RecurrentFnOutput(reward, discount, prior_logits, value), new_embedding)`.
  - **AZ mode:** `recurrent_fn` steps the *real* vectorized env and runs the net on the new obs.
  - **MuZero mode:** `recurrent_fn` calls a learned dynamics net.
- The pgx AZ example uses `gumbel_muzero_policy` with a real-env `recurrent_fn` — that *is*
  AlphaZero, and gives Gumbel's policy-improvement guarantees at low simulation counts.

## 3. Components to build

1. **Arimaa JAX env** (pgx-style) — *the crux and the main risk*. Fixed-shape jittable arrays:
   - State: `board` (piece planes or an int8 8×8), `current_player`, `steps_left`,
     `step_within_turn`, `already_moved`/push-pull lock mask, `legal_action_mask`, `rewards`,
     `terminated`, repetition hashes.
   - `init`, `step(state, action)`, `observe(state)` — all branchless (`lax.cond`/`where`,
     no data-dependent Python control flow, no dynamic shapes).
   - Reuse our **per-step action factorization** (1393 = 1392 step-specs + END_TURN): one
     action = one step keeps env transitions simple and jittable. (Compound-move encoding à la
     optima's 2261 is a later experiment, not v1.)
   - Setup phase handled as `Place` actions + a phase plane (learned, like optima).
2. **Backbone-agnostic network** (Flax or Haiku): `obs -> features`, shared policy head
   (→ 1393 logits) + value head (tanh or categorical) + optional **moves-left** / margin heads.
   Registry-based backbone selection. Optional **SE** blocks. Feature-plane set is config-driven.
3. **Search wiring** (`search.py`): builds `root_fn`/`recurrent_fn` for AZ and MuZero modes;
   wraps `gumbel_muzero_policy`; legal-action masking; playout-cap randomization hook.
4. **Self-play** (`selfplay.py`): `pmap` over devices, `vmap` over a large batch of games,
   `lax.scan` over move steps with `auto_reset`; collects (obs, policy_target, value_target).
5. **Training** (`train.py`): value targets by discounted backtrack; loss = policy CE + value
   + aux; optax (SGD+Nesterov or AdamW) with warmup; `pmap` gradient averaging; checkpointing.
6. **Arena / gating** (optima insight): candidate must beat champion over N games @ high visits
   to be promoted; optional pool of net sizes.
7. **AEI bridge** (`aei_jax_engine.py`): load params, run JAX inference to pick moves, speak AEI
   — so the new bot drops straight into the **existing harness** and plays sharp/occam/faerie.
   (Mirror the current `arimaa_aei_engine.py`; swap the policy backend to the JAX net.)

## 4. How current work is preserved (rebase risk is bounded)

- **`games/arimaa.py` becomes the reference oracle.** We validate the JAX env by *differential
  testing*: run both engines on the same random games and assert identical legal-move sets and
  outcomes. Our proven Python engine is exactly the ground truth we need — it is NOT thrown away.
- **AEI harness + built opponents** (sharp, occam, faerie, simple) transfer unchanged.
- **The optima analysis** (`memory/optima-vs-us.md`) is the feature backlog for Phase 4.

## 5. Key decisions to lock before coding

- **NN library: Flax (linen or nnx) vs Haiku.** pgx's example uses Haiku (copy-paste head start),
  but Haiku is in maintenance mode; **Flax is more actively developed and better for transformer
  backbones** (the stated goal). Recommendation: **Flax**, accepting we adapt rather than copy the
  pgx example.
- **Base template: fork a0-jax (minimal, self-contained, MIT) vs pgx example vs from-scratch.**
  Recommendation: study both, **structure like a0-jax** (self-contained, pluggable `--agent-class`)
  but write our own env/loop so we own the throughput + gating extensions.
- **Value head: scalar tanh vs categorical (KataGo/our current support_size).** Start scalar; make
  it swappable.
- **Action factorization: per-step (1393, v1) vs compound (optima-style).** Per-step for v1.

## 6. Risks & hard parts (honest)

1. **Vectorized JAX Arimaa engine is genuinely hard** — push/pull/freeze/trap/repetition and the
   multi-step turn, all branchless and fixed-shape. This is the bulk of the novel work and the top
   risk. Mitigation: differential test against `games/arimaa.py` from day one.
2. **Policy head is 1393-wide** — fine, but confirm memory/throughput at scale.
3. **Compute reality:** mctx+pgx-class stacks can do *millions* of self-play games on TPU/multi-GPU,
   but reaching ~3000 Elo is still **thousands of accelerator-hours**. A single consumer GPU yields a
   working, improving bot — not a superhuman one. Serious runs want a cloud TPU v3/v4 pod or multi-GPU.
4. **Debugging JAX** (jit/shape/tracer errors) has a learning curve vs eager PyTorch.

## 7. Roadmap (v2 — updated after the initial build)

### 7.0 Status — DONE and validated
Phases 0–3 and the backbone work are built: `jaxarimaa/` env (`observe`/`legal_action_mask`
/`step`/`init_state`) **differential-tested vs the oracle over 1.4M+ transitions, 0
mismatches**; ResNet(+SE) and Transformer backbones behind a registry; mctx AlphaZero
search; vectorized self-play with bootstrapped value targets; jitted train step with a
data-parallel mesh; replay/checkpoint/eval; AEI inference bridge (`--policy jax`) in the
opponent harness; a quality cleanup pass. Remaining: the strength climb — the two tracks
below **plus real accelerator compute**. Tracks are independent; do Tier 1 of each first.

### 7.A Distributed-training track
**Tier 1 — the blocking gap (verified):**
- **Shard self-play across the mesh.** `selfplay.generate` currently runs **replicated**
  (output sharding `P()`): every device plays the *same* games, so more chips give **zero
  self-play speedup**. Fix: make the per-game batch a *sharded input* — split keys outside
  and pass a `[B]` key array with `data` sharding (or wrap the rollout in `shard_map` over
  the `data` axis) so each device plays a distinct subset. Highest-value distributed change;
  without it the slice roadmap is moot for the expensive half.

**Tier 2 — scale enablers (before medium slices):**
- **On-device sharded replay** (a sharded `jax.Array` ring) replacing the host numpy buffer —
  avoids a host-memory gather bottleneck.
- **Slice-aware LR/batch**: global batch = per-device × chips; apply the linear-scaling rule
  (parameterize `trainer.make_optimizer`'s peak LR by global batch; warmup already present).
- **Replay-ratio / staleness** knob (train steps per self-play sample) once throughput jumps.

**Tier 3 — multi-host / pod:**
- `jax.distributed.initialize()` at process start; mesh from `jax.devices()`; fold
  `process_index` into PRNG keys so hosts don't duplicate games. Keep arena/gating (7.B) as
  the periodic global barrier, out of the hot loop.

### 7.B "Free wins" from optima (cheap, proven; see optima-vs-us memory)
**Tier 1 — cheapest, highest-ROI, low-risk:**
- **Richer input planes.** The net is currently blind to freezing/traps/turn-progress. Add:
  **frozen mask** (already computable via `env._frozen_grid`), **trap squares** (static),
  **step-within-turn** (= `4 - steps_left`). Bump `OBS_SHAPE`/`observe`; the legacy-13 planes
  stay byte-identical (re-run difftest on those; hand-verify the new ones). The "already-moved
  this turn / push-pull lock" plane needs a small new `State` field + `step` bookkeeping.
- **Left-right symmetry augmentation** (free 2× data). Arimaa is symmetric under `x→7-x`;
  precompute the induced **action-index permutation** (fixed table in `constants.py`) and
  mirror (board, policy-target) in the trainer. Pure data win, no model change.

**Tier 2 — moderate, proven:**
- **Arena gating.** Keep champion params; periodically play candidate-vs-champion (reuse
  `evaluate.play_vs_random` against the champion) and promote past a win-rate threshold.
  Regression-proofs training; doubles as the distributed barrier.
- **Moves-left auxiliary head** (predict remaining plies; target from game length) for
  decisive finishing — small `network.py` + target change.

**Tier 3 — bigger / uncertain:**
- **Playout-cap randomization** (throughput). Wrinkle: `num_simulations` is a static jit arg →
  needs two search variants or a padded scheme; defer until profiled.
- **Transposition-aware search** — optima's real structural edge, but mctx builds a tree not a
  DAG; a custom search. High effort; the one "not free" win.
- **Learned setup** (add PLACE actions; enables a meaningful setup/phase plane) and **KataGo
  deblunder** value-target correction.

### 7.C The differentiated bet (unchanged)
Keep MuZero's learned model + per-step factorization as an *efficiency* experiment, measured
head-to-head — not assumed. The mctx wiring already keeps both AZ and MuZero modes reachable.

**Suggested first sprint:** 7.A-Tier1 (shard self-play) + 7.B-Tier1 (input planes + symmetry)
— makes the slice actually scale *and* feeds the net the signal it's missing: the highest
strength-per-effort combination before committing to a large run.

## 8. Distributed self-play & training — scaling to large TPU slices

### 9.1 What mctx provides (and doesn't)

mctx has **no distribution layer of its own**. It is a batched, `jit`/`vmap`/`pmap`/`shard_map`-
compatible search primitive: give it a batch of root states and it searches them in parallel on
whatever devices the surrounding JAX program is using. **All** distribution comes from JAX (SPMD
sharding + multi-host), not from mctx or pgx. So the scaling story is a JAX story, and it applies
uniformly to search, self-play, and training.

### 9.2 The right architecture for us: synchronous, all-on-device SPMD

Because our **environment is vectorized JAX (on-device)**, we do **not** need the CPU actor-learner
/ centralized-inference cluster (Ray, SEED RL, the original AlphaZero/AlphaStar setup) that
PyTorch/TF engines like optima require. Instead:

- The **entire loop — self-play rollouts (env.step + mctx search) and the gradient step — is one
  SPMD program**, replicated across every chip.
- **Parallel axis = data parallelism.** Self-play games are independent, and our net is small
  (SE-ResNet / modest transformer), so we shard the *batch of games* across devices and
  all-reduce (`pmean`) gradients in the training step. **No model/tensor/pipeline parallelism
  needed** — pure data parallelism scales to thousands of chips.
- This is exactly how pgx's AlphaZero example scales: self-play batch **1024 on 1×A100 → 8192 on
  8×A100 via `pmap`**, near-linear. The same code runs on TPU pods unchanged under SPMD.

### 9.3 Single-host vs multi-host (the one real code change for pod scale)

- **Single host (≤ 8 chips):** `pmap` (or `jit` + `NamedSharding`) over local devices — trivial.
- **Multi-host (TPU slices ≥ 1 host):** call `jax.distributed.initialize()` once; then a global
  `jax.Array` spans all processes and a `Mesh` + `NamedSharding`/`PartitionSpec` lets you program
  the whole slice "as one big device." Every host runs the identical SPMD program; collectives
  (grad all-reduce, replay gather) cross hosts over ICI automatically.
- **API choice:** prefer **`jit` + `shard_map`/`NamedSharding`** over `pmap` for the multi-host
  path — it's the modern, more composable, better-performing route and debuggable eagerly. (`pmap`
  works but is legacy and awkward to compose.) Build the mesh as data-parallel (`Mesh(devices,
  ('data',))`) from day one so single-host → pod is a mesh-size change, not a rewrite.

### 9.4 TPU slice targets for the roadmap

Data-parallel scaling maps directly onto slice sizes (chips ≈ replicas; each host ≈ 4–8 chips):

| Tier | Example slice | Chips / hosts | Role |
|---|---|---|---|
| Dev | 1×GPU or `v5e-8` / `v4-8` | ~8 / 1 host | Correctness, smoke trains, `pmap` only |
| Small | `v5e-32` / `v4-32` | 32 / ~4 hosts | First multi-host; validate `jax.distributed` + mesh |
| Medium | `v5e-256` | 256 / ~32–64 hosts | Real self-play throughput (v5e maxes at 256/slice) |
| Large | `v5p-256 → v5p-512+` | 256→512+ / 32+ hosts | Serious training runs |
| Pod | `v5p` (up to 8960) / `v4` (4096) | thousands | Ceiling; where ~3000-Elo-scale compute lives |

Design so **slice size is a config knob**: `global_batch = per_device_batch × num_chips`, mesh
axis sized to the slice. Nothing in the model or env changes between `v5e-8` and a `v5p` pod.

### 9.5 Scaling caveats to design for now

- **Global batch grows with chips** → tune LR warmup / scaling (linear-scaling rule) or self-play
  will destabilize at large slices. Make batch and LR schedule slice-aware.
- **Replay buffer at pod scale:** an all-on-device replay (sharded `jax.Array` ring buffer) avoids
  host-memory bottlenecks; if it outgrows HBM, spill to host RAM per host (keep it host-local, not
  a global gather). Decide this before medium tier.
- **Arena/gating sync** is a periodic global barrier — cheap relative to self-play, but keep it out
  of the hot loop.
- **Determinism/RNG:** fold `process_index`/device id into PRNG keys so hosts don't duplicate
  self-play games.
- **Cost reality:** near-linear *throughput* scaling ≠ linear *strength* gains; larger slices buy
  wall-clock, and pod-scale runs are expensive. Scale the slice to the experiment, not by default.

## 9. Out of scope (v1)

Compound-move action encoding; learned-dynamics MuZero mode (kept *possible* but not built first);
endgame DBs / opening books; distributed multi-node beyond single-host multi-device.

## 10. Modular Track-B plan (ablation-ready) — plan of record

Status update: Track A (distributed) + Orbax checkpointing + metrics are DONE. This
section is the modular design for the model-quality improvements. **Principle:** one
`FeaturesConfig` of independent boolean toggles; baseline = all off (= current behavior).
Each feature is individually on/off so we can run single-feature ablations and ship only
what earns its keep. Toggles thread to exactly one owning module each.

### Backends (decided)
- **ResNet+SE = default main run** (proven for board-game AZ; matches optima). **Transformer
  = ablation** via the existing registry (`--transformer`). Strength ordering is unknown until
  a fixed-compute A/B; conv is the safe prior. Batching is already optimal (vmap→XLA batched
  ops, params broadcast once, no BatchNorm) — no rework needed there.

### `FeaturesConfig` toggles (proposed)
- `planes_frozen`, `planes_trap`, `planes_step_in_turn`, `planes_moved` — input planes.
- `symmetry_aug` — left-right (x→7-x) data augmentation.
- `moves_left_head` — auxiliary head.
- `arena_gating` — champion/candidate promotion.
- `playout_cap` — playout-cap randomization.
- `bf16` — mixed-precision compute (efficiency).
OBS_SHAPE is DERIVED from the enabled plane toggles; the net's first Conv infers input
channels, so the backbone auto-adapts — only `observe()` + the buffer's obs shape depend on it.

### Tier 1 — cheapest, highest ROI (do first)
- **Input planes** [`planes_*`] — owner: `constants.py` (plane registry + derived OBS_SHAPE) +
  `env.observe`. Add: frozen mask (via `_frozen_grid`), trap squares (static), step-within-turn
  (= 4−steps_left). `planes_moved` (push-pull lock / already-moved) needs a small new `State`
  field + `step` bookkeeping. Re-run difftest on the legacy-13 planes (must stay byte-identical);
  hand-verify new planes. Net blind to freezing/traps today → likely real strength.
- **Left-right symmetry aug** [`symmetry_aug`] — owner: `constants.py` (precomputed action-index
  permutation for x→7-x) + `trainer` (mirror obs + permute policy target). Free 2× data, no model
  change.
- **bf16 mixed precision** [`bf16`] — owner: `network.py`/`trainer` (compute dtype; keep master
  params fp32). Biggest raw-compute win on TPU/GPU; helps every config.

### Tier 2 — moderate, proven
- **Arena gating** [`arena_gating`] — owner: `train.py` + reuse `evaluate` vs a held champion;
  promote past a win-rate threshold. Regression-proofs training; is the natural multi-host barrier.
- **Moves-left head** [`moves_left_head`] — owner: `network.py` (extra head) + `selfplay`
  (length target) + `trainer` (extra loss term, gated). Decisive finishing.

### Tier 3 — bigger / uncertain
- **Playout-cap randomization** [`playout_cap`] — owner: `selfplay`. Wrinkle: `num_simulations`
  is a static jit arg → two compiled search variants (full/fast) selected per move. Throughput win.
- **Transposition-aware search** — optima's real structural edge; mctx is a tree not a DAG →
  custom search. High effort; the one "not free" win. Defer.
- **Learned setup** (PLACE actions + phase plane) and **KataGo deblunder** value correction. Defer.

### Ablation protocol
Baseline (all off) → enable one toggle at a time → measure win-rate vs a fixed opponent
(random, then `occam`/`sharp`) at equal compute. Keep the toggles that move the needle; the
`FeaturesConfig` in each checkpoint's metadata records exactly what was on.

**Suggested first build:** `FeaturesConfig` scaffold + Tier-1 (`planes_*`, `symmetry_aug`, `bf16`),
since those are cheap, high-ROI, and immediately ablatable.

## 11. Experiments & open designs (scoping) — extends §10

Status: Tier-1 planes + symmetry + bf16 + planes_moved + arena_gating are BUILT. This
section scopes the remaining architectural experiments and open design questions. Each
lists: what, design options, cost, and a recommendation. Nothing here is built yet.

### 11.1 moves_left_head  [toggle: moves_left_head] — LOW effort
Auxiliary head predicting remaining plies to game end (softmax over K bins). Target = actual
plies-to-end, already computable in self-play's reverse value scan. Loss += weight*CE (gated).
- **Minimal (recommended first):** train the head only — a proven auxiliary-task regularizer;
  no search change. Owner: network.py (head) + selfplay (target) + trainer (loss term).
- **Full:** feed moves-left into MCTS selection to prefer faster wins / slower losses (KataGo).
  Requires threading it into the search value — more integration; do only if the head helps.

### 11.2 early_exit  [toggle: early_exit] — LOW-MED effort — NEEDS 1 CLARIFICATION
Ambiguous; two useful readings (defaulting to the first):
- **(A) Game-level adjudication / resign (likely intent):** end a self-play game early once the
  value is confidently decided (|v|>thresh for N consecutive steps) → stop burning steps on
  decided games (KataGo). Owner: selfplay (adjudicate a lane: set terminal + predicted winner).
  Risk: false adjudication biases targets → keep a fraction of games to completion for
  calibration. Pairs naturally with moves_left_head.
- **(B) Network early-exit (adaptive compute):** exit the ResNet early on "easy" positions via
  intermediate heads. Real architectural idea but uncommon in AZ; higher complexity, uncertain.
- Recommendation: confirm which; default to (A) — it's a compute win with known practice.

### 11.3 learned-model mode (MuZero)  [mode: learned_model] — HIGH effort (its own phase)
The differentiated bet. Replace the exact-simulator recurrent_fn with a learned model:
representation h(obs)->s0, dynamics g(s,a)->(s',reward), prediction f(s)->(policy,value); mctx
recurrent_fn calls g+f instead of env.step. Training gains unroll losses (policy/value/reward
over K steps + n-step returns) — the machinery we left behind in muzero-general. Not a small
toggle: a parallel training pipeline. mctx already supports it (muzero_policy). Frame as a
dedicated phase; measure head-to-head vs the AZ line at equal compute. Honest prior: on Arimaa's
cheap exact simulator this is a *liability* for tactics; its only edge is inference efficiency
(amortized encoder). Do AZ well first.

### 11.4 Arena candidate pool  [param: arena_pool_size] — LOW-MED effort, feasible
Generalize arena from 1 candidate vs 1 champion to K candidates (recent/best snapshots) run as a
**gauntlet vs the champion** (K matches, NOT K^2 round-robin — keeps it cheap). Promote the best
that clears the threshold. Since arena is periodic, K*games is affordable. Impl: keep a ring of K
param snapshots (host or device); loop `play_match` over them (params are large pytrees, so loop
rather than vmap to avoid K x memory). Efficient enough. Recommendation: build as `arena_pool_size`
(default 1 = current behavior); gauntlet, not round-robin.

### 11.5 Learned setup  [toggle: learned_setup] — MED-HIGH, needs curriculum (11.6)
Today setup is random. Options:
- **(A) PLACE actions in the main action space** (optima-style, ~16 place actions): env gains a
  setup phase; net's policy head grows. Changes N_ACTIONS + env; biggest surgery.
- **(B) Separate setup policy/head** invoked only in the setup phase; movement policy unchanged.
  Cleaner separation, but two policies to train.
- Recommendation: (A) for a single unified net, BUT allocate the setup actions from the start
  (fixed action space) and *gate* whether the setup policy is used vs forced-random — this makes
  it curriculum-toggleable (11.6) without re-init. Depends on 11.6.

### 11.6 Curriculum / toggling features DURING training — MED, high leverage (design)
Goal: flip features mid-run (e.g., random setup -> learned setup after fundamentals; enable planes
partway). Key constraint: **behavior toggles are hot-swappable; shape-changing toggles are not**,
because params are init'd for fixed input/output shapes.
- **Behavior toggles** (symmetry_aug, arena_gating, playout_cap, early_exit, moves_left loss
  weight): trivially schedulable per-iteration.
- **Shape-changing toggles** (input planes, learned_setup action space): cannot flip mid-run
  naively. **Design fix — "allocate-but-gate":** build the net ONCE with the SUPERSET shape (all
  planes present; full action space incl. PLACE), but zero/mask the not-yet-enabled parts. Flipping
  a feature on mid-run then just starts feeding real signal into inputs/outputs the net already has
  weights for — no re-init. Planes: always computed, zeroed until enabled. Learned setup: PLACE
  actions always in the space, masked to forced-random until enabled.
- **Mechanism:** a `FeatureSchedule` mapping iteration -> active FeaturesConfig (or per-feature
  start-iters); train loop resolves the active config each iter. Net built from the schedule's
  superset. Recommendation: adopt allocate-but-gate so the schedule never changes tensor shapes.

### 11.7 Transposition-aware search — EVALUATE BEFORE BUILDING
- **Why it might matter a lot for US specifically:** per-step action factorization means a 4-step
  turn has many step-orderings reaching the SAME position -> the search tree re-explores identical
  states heavily *within a turn*. Transposition redundancy is plausibly higher for us than for
  optima's compound-move encoding.
- **Why it might NOT be worth it:** mctx builds a TREE, not a DAG (no node merging); Gumbel-MuZero
  already uses FEWER sims (sequential halving) than deep PUCT, shrinking the absolute waste; and a
  mutable position->eval cache is infeasible inside a jitted, batched on-device search.
- **CHEAP MEASUREMENT FIRST (do this, ~an afternoon):** after a search, mctx's tree is a pytree of
  node embeddings (our States). Hash each expanded node (board+player+steps_left, e.g. Zobrist or
  bytes) and count unique vs total expansions -> the *transposition rate* = the ceiling on savings.
  If high (say >50%) AND we're compute-bound, a custom transposition search may justify its cost;
  else drop it. This measurement is a gate, not a commitment.
- **If pursued:** a custom search (not mctx) or an mctx fork with node dedup — HIGH effort, fights
  the batched-JAX paradigm. The one "not free" win. Decide by the measurement.

### Suggested experiment order
1. moves_left_head (train-only) — cheap, proven.
2. arena_pool_size — cheap generalization of what's built.
3. Transposition MEASUREMENT experiment — cheap, decides 11.7 up front.
4. early_exit(A) — after confirming intent; compute win.
5. Curriculum mechanism (11.6, allocate-but-gate) — unlocks learned_setup cleanly.
6. learned_setup — depends on 5.
7. learned-model mode — its own phase, after the AZ line is strong.

## 12. Deeper design notes (throughput, LLM transfer, early-exit, allocate-but-gate)

Answers to a round of design questions, integrated with §10–§11. Research-backed where noted.

### 12.1 early_exit = network adaptive-depth (interpretation B)
- **Core problem (verified):** early-exit saves compute only when different examples exit at
  different depths — but our batched, jitted, on-device MCTS self-play runs the whole batch
  through the net in lockstep. Heterogeneous exit depths break SIMD; the literature's fix is
  "dynamic rebatching" (route exiters out, regroup continuers — e.g. DREX), which is complex and
  infeasible inside jitted mctx. **=> true early-exit yields ~no throughput win in batched self-play.**
- **Two regimes:** batched self-play (early-exit unhelpful) vs single-game inference (the AEI /
  tournament bot plays one position at a time — there early-exit DOES cut per-move latency).
- **Batch-friendly cousin (recommended):** deep-supervision auxiliary heads at intermediate depths
  — trained (denser supervision, richer representation, MTP-flavored) but ALWAYS run full depth at
  inference. Captures much of the representational benefit, no batching problem. Reserve true
  early-exit for a separate latency-mode of the tournament bot.
- **Transformer context:** Lc0's transformer (BT4/"Chessformer") beats its best conv net by ~270
  Elo with fewer params — real evidence transformers win at board scale (64 tokens). If we pursue
  the transformer backbone, borrow **Smolgen** (dynamic attention-bias generation — the trick that
  made attention effective for chess) and 2D/RoPE positional encodings.

### 12.2 Multi-TPU throughput for SMALL nets — pipeline/tensor parallelism NOT needed
- Pipeline/tensor parallelism exist to fit models too big for one chip. Our nets are tiny (even a
  Lc0-scale transformer at ~200M fits on one accelerator). Splitting them across chips adds
  cross-chip communication for zero benefit. **Data parallelism (more concurrent games) is the
  right and sufficient axis** (already built via shard_map).
- Real throughput levers instead: (1) bigger data-parallel batch (more games); (2) bf16; (3)
  counter-intuitively, a BIGGER net — tiny nets UNDER-utilize the MXU (memory/overhead-bound); a
  larger net does more FLOPs per byte -> better hardware efficiency AND strength; (4) minimize host
  sync / donate buffers. If we're ever "compute-bound" with a tiny net, we're really
  MXU-underutilized -> scale the net or the batch, don't add model parallelism.

### 12.3 LLM ideas worth transferring
- **MTP-analog (most transferable, batch-friendly, train-time):** auxiliary heads predicting future
  policy/value/state at multiple horizons from the same hidden state -> denser supervision + richer
  multi-step representations (DeepSeek-V3 MTP shows consistent gains). Converges with moves_left_head
  and the deep-supervision idea (12.1) and MuZero unroll. Concretely: predict policy/value at t+1,t+2
  or the full 4-step turn. Cheap experiment.
- **Smolgen + 2D/RoPE positions** for the transformer backbone (from Lc0) — proven board-specific.
- **Muon / modern optimizers** — cheap orthogonal experiment for matmul-heavy nets.
- **Distillation** — train a small fast net from a strong big one for the latency-sensitive
  tournament bot (deployment).
- **MoE** — conditional compute; same SIMD/routing batching problem as early-exit; not worth at this scale.
- Speculative decoding has no clean analog (MCTS isn't autoregressive); the learned-model/MuZero
  unroll is the nearest thing.

### 12.4 allocate-but-gate: bloat concern resolved by WHERE the feature lives
- **Edge features are cheap to carry:** input planes add channels only to the FIRST conv (a few
  input channels); action-space features (learned-setup PLACE actions) add only to the FINAL policy
  Dense (1393 -> ~1409). The trunk (blocks x channels) is unchanged. So allocate-but-gate is nearly
  free for planes + action space — the bloat worry is largely unfounded for exactly the features we
  want to curriculum-toggle.
- **Trunk/architecture features are NOT cheap to carry:** extra depth/width, experts, per-layer
  early-exit heads. Don't allocate these speculatively.
- **Endorsed workflow (matches the user's instinct):** Phase 1 (exploration) — allocate the superset
  of EDGE features, run curriculum + ablation to find which features/curricula help (cheap, since
  edges are nearly free). Phase 2 (production) — allocate only the winners; decide TRUNK features by
  separate ablation runs and commit before the big run. So: yes, explore-with-superset then trim;
  the trim mainly matters for trunk features.

## 13. TPU training cost estimate (background-agent research, 2026)

Full write-up delivered separately; key numbers (v5e-equivalent chip-hours; error bars ~5-10x):

| Phase | Slice | chip-hours | $ (spot -> on-demand) |
|---|---|---|---|
| Exploratory / ablation (30-50 short runs) | v5e-8/run | ~700-2,400 | ~$250-$900 -> ~$860-$2,900 |
| Full run — low | v5e-256 | ~2,100-3,500 | ~$0.8k-$1.3k -> ~$2.5k-$4.3k |
| Full run — central | v5e-256 / v5p-128 | ~6,600-11,100 | ~$2.4k-$4.0k -> ~$8k-$13k |
| Full run — high | v5p-256+ | ~44k-74k | ~$16k-$27k -> ~$53k-$89k |

Pricing used (per chip-hour): v5e $1.20 on-demand / ~$0.36 spot; v5p $4.20 / ~$1.26.
Peak bf16 ~197 TFLOP/s (v5e), ~459 (v5p). Small nets get only ~10-25% MFU (memory/overhead
bound) — factored in, and why v5p's raw FLOP edge doesn't translate to proportional savings.

- **Cost driver = self-play MCTS forwards** (num_simulations x plies/game x games), not gradients.
- **Reference anchors:** optima/rusty_zero (~3000 Elo) < 5,000 GPU-h with a small SE-ResNet +
  playout-cap; KataGo ~12,000 GPU-h; AlphaZero-chess ~20,000 TPU-h (brute-force ceiling). v5e @
  15-25% MFU ~= a V100, so optima's <5k GPU-h maps to ~3-5k v5e chip-h — between "low" and
  "central". **Planning figure: budget ~10,000 v5e chip-h (~$3.6k spot / ~$12k on-demand), with
  headroom to 30k if the strength climb is slow.**
- **Run everything on SPOT** (~70% off) — we have Orbax preemption-safe checkpointing; a $12k
  on-demand full run becomes ~$3.6k. Ablations belong on v5e-8; full run on v5e-256 (best $/FLOP
  for small nets); v5p only if wall-clock > $ or the net grows enough to raise MFU.
- **Dominant uncertainty = games-to-~3000-Elo (~5x).** De-risk cheaply BEFORE the full run:
  (a) `jax.jit(fwd).lower().compile().cost_analysis()` for real net FLOPs; (b) one-chip
  games/sec + MFU measurement (metrics harness already logs throughput); (c) a short
  games-to-strength curve on v5e-8. Together these shrink the ~10x estimate to ~2x.

## 14. Roofline / MFU analysis (measured) — and net-sizing guidance

Measured arithmetic intensity via XLA `cost_analysis` (self-play batch 256, bf16):

| Net | params | AI (FLOP/byte) |
|---|---|---|
| C=64,  8 blocks  | 1.0M  | 228 |
| C=128, 15 blocks | 4.9M  | 373 |
| C=256, 20 blocks | 24.7M | 740 |
| C=128, **batch=1** | 4.9M | **17** (memory-bound) |

Roofline ridge points (peak bf16 / HBM BW): v5e 240, v4 224, v5p 166, A100 153, H100 296,
L4 403, T4 203 FLOP/byte.

**Key finding:** at self-play batch, C=128 (AI 373) is ABOVE the ridge on every TPU and on
A100/H100 -> **compute-bound, not memory/bandwidth-bound.** So low MFU is NOT a bandwidth
problem; it's (1) **systolic-array fill** — conv N-dim = channels, so C must be a MULTIPLE OF
128 (the 128x128 MXU width); C=64 half-fills the array, C=128 fills one tile, C=256 two; and
(2) **Amdahl drag** from non-MXU work (tiny policy/value/SE heads, LayerNorm, mctx tree
gather/scatter, env step) that is fixed per simulation and doesn't touch the MXU.

Consequences:
- **Bigger net = higher MFU** (trunk matmul dominates the fixed per-sim overhead). C=64 is the
  WORST case (half-filled arrays + overhead-dominated). Default net set to **C=128, 10 blocks**.
- **Batch is the array-feeder:** M-dim = 64*B (spatial*games). B=1 -> AI 17 (memory-bound); big
  self-play batch -> AI 370+ (compute-bound). "Run forwards in parallel" == large game batch.
- **Hardware $/useful-FLOP:** for a C=128-256 net, **v5e is the sweet spot** (fills its 128-MXU,
  cheapest); v5p only pays off at C>=256 or when wall-clock > $. Cheap GPUs (L4/T4) are
  bandwidth-starved (ridge > our AI) -> memory-bound -> poor for this workload; A100/H100 fill
  well but pricey. TPU v5e likely wins for a properly-sized net.
- **Revised MFU:** the earlier "10-25%" was pessimistic for a properly-sized net; a C=128 net on
  v5e (compute-bound, full MXU) should reach ~35-55% MFU -> full-run cost lands at the cheaper
  end of §13. Confirm with a one-chip profile (FLOPs/wall-clock / peak).

### Amdahl mitigation
- Main lever: **size the net so trunk matmul >> fixed per-sim overhead** (C=128-256, more
  blocks). Overhead (env step, tree ops) is ~constant per simulation regardless of net size.
- **Large game batch** amortizes fixed per-step cost and fills M.
- Let **XLA fuse** (MXU overlaps VPU/HBM within fused graphs automatically) — do NOT hand-roll
  streams; TPU has no CUDA-style multi-stream, and self-play vs training both hit the MXU so they
  can't overlap on shared devices (an actor-learner split could, but adds complexity we don't need
  at small scale). Keep ops fusable: leading batch dim, avoid awkward reshapes/layouts.
- Fewer sequential sims (Gumbel low-sim + playout-cap, both built) -> less accumulated per-sim
  overhead.

### Getting the most out of a v3/v4 (or any) pod
- v3/v4 use 128x128 MXUs -> a C=128 net fills the width (one tile); C=256 = two. So C>=128 is a
  prerequisite for good pod MFU.
- To feed EVERY chip you need total game batch >> num_chips (M=64*per_chip_batch >= 128 needs
  per_chip_batch >= 2, but hundreds/chip to amortize overhead). Self-play scales games cheaply,
  so this is feasible — but a huge pod + small net is underutilized.
- Self-play is a SEQUENTIAL scan over moves (each depends on prior) -> parallelism is only across
  the game-batch axis, not time; and fixed per-move overhead is replicated per chip and doesn't
  shrink with pod size. So a big pod hits diminishing returns unless the net is big AND the batch
  is enormous. **Right-size the slice to (net size x achievable batch); don't rent a full v4-4096
  pod for a small net.** v3 (cheap, smaller per-chip FLOPs) can be economical for a small net if
  available. Scale the slice only while a one-chip/one-slice MFU measurement stays high.

## 15. Current status & next steps (the build -> train inflection)

### Where we are
The engine is **feature-complete for the exploratory phase and has NEVER trained a real model.**
Built + validated (CPU / simulated-multi-device only):
- Vectorized JAX env (observe/legal_action_mask/step/init_state), differential-tested vs the
  oracle over 1.4M+ transitions, 0 mismatches.
- mctx Gumbel-AlphaZero search; sharded self-play; on-device HBM replay; data-parallel training;
  multi-host init (guarded); Orbax preemption-safe checkpoint+resume; metrics/monitoring.
- Backbones: ResNet+SE, Transformer (+Smolgen, +2D-RoPE), registry. Default net C=128/10 blocks.
- Toggles (all ablatable, baseline byte-identical): planes {frozen,trap,step_in_turn,moved},
  symmetry_aug, bf16, arena_gating, resign, playout_cap, moves_left_head, deep_supervision, mtp,
  smolgen, rope; optimizer {adamw,adam,sgd,lion}.
- AEI inference bridge into the opponent harness (bot_sharp/occam/faerie built).
- Cost estimate (§13) + roofline/MFU analysis (§14).

### The inflection: STOP adding features, START measuring/training
We have never confirmed the approach *learns* (2-iter CPU smoke tests show only that loss moves).
Continuing to add features (learned setup, transposition search) before any real training is
premature — we'd be optimizing unmeasured things. Highest-value next steps, in order:

1. **Get on real hardware** (one GPU or `v5e-1`): confirm the JAX pipeline runs on GPU/TPU, and
   **measure real MFU + games/sec** (cost_analysis FLOPs / wall-clock / peak). De-risks §13/§14.
2. **Games-to-strength curve** (the #1 unknown): train a small real config, plot win-rate vs
   random -> occam -> sharp against cumulative self-play games. This is the "does it actually
   learn / climb?" gate — the single most important experiment we have not run.
3. **Ablation campaign** on `v5e-8`: flip one toggle at a time (baseline all-off), measure
   win-rate at equal compute; keep the winners. This is the whole point of the modular toggles.
4. **Scale** the winners to a competitive run (`v5e-256`, spot) per §13.

### Remaining BUILD items (slot in around the empirical work)
- **arena_pool_size** (tunable arena candidates; gauntlet vs champion) — small, do when useful.
- **Transposition MEASUREMENT** (§11.7): hash mctx tree node embeddings, count unique/total ->
  decides whether a transposition-aware search is worth it. Cheap gate; run alongside step 1-2.
- **Features/inputs "finally" bucket:** learned_setup (PLACE actions) + curriculum
  (FeatureSchedule, allocate-but-gate). Do AFTER we know the base approach learns and want more
  strength; the superset net must include the aux heads + setup action space.
- Tier-3 stretch: transposition-aware search (only if the measurement says yes), learned-model
  MuZero mode (its own phase), KataGo deblunder.

### Evaluation / opponents
- Strong live opponent (optima/rusty_zero) is blocked on weights -> email the author. Meanwhile
  Janzert/OpFor (AEI-native, buildable) would add a Challenge-level rung to the ladder.
- Multi-host launch recipe (ops, environment-specific) needed only when scaling past one host.

## 16. Scaling further: actor-learner decoupled async training (scoping)

Spot-readiness code wins (DONE): **persistent XLA compilation cache**
(`distributed.enable_compilation_cache`, `--compile-cache <dir|gs://>`; verified a fresh
process gets a cache HIT → fast restart) and **durable checkpoint dir**
(`TrainConfig.ckpt_dir` / `--ckpt-dir gs://...`; URIs pass through, only local paths abspath'd).
With Orbax resume + these two, a single spot run restarts cheaply. What follows scopes the
*next* scale tier.

### The two architectures (Podracer framing)
Our current design is **Anakin** (DeepMind "Podracer" taxonomy): everything on-device, ONE
synchronous SPMD program — self-play + train in the same jitted loop, gradient all-reduce each
step, single mesh over ICI. Simple, high-utilization, but a single synchronous job: any spot
preemption stalls/fails the whole thing (Orbax restarts it).

**Actor-learner decoupled async = Sebulba**: separate roles.
- **Actors** (many, independently preemptible): each runs self-play with a recent policy, pushes
  trajectories to a shared replay; periodically pulls fresh params. Reuse our env/net/search/mctx
  and `selfplay.make_generate` unchanged — only the loop wrapper changes.
- **Learner** (one, or a small sync group): pulls batches from shared replay, does gradient
  steps, publishes params periodically.

### Why it fits US (and matches LLM RL)
- **Staleness-tolerant, like async LLM RL.** Actors train on *slightly stale* params (off-policy).
  AZ is unusually robust to this: the training target is the SEARCH-IMPROVED policy (MCTS corrects
  whatever policy it's given) + game-outcome values — we're already quasi-off-policy via the replay
  window. So bounded staleness costs little, exactly the trade modern async RLHF/GRPO systems make.
- **Resilience:** a preempted actor just stops contributing; the learner and other actors continue.
  No synchronous barrier -> no all-or-nothing preemption. This is the real answer to "run across
  several small spot slices for one run" (which plain synchronous SPMD can't do — separate slices
  aren't one ICI mesh).
- **Specialization / diversity:** actors can run different temperatures, opening books, opponents,
  or even net variants -> more diverse self-play data. Heterogeneous hardware too (cheap inference
  slices for actors, one beefy learner).

### Costs / risks
- Real infrastructure: a shared replay SERVICE (DeepMind **Reverb** is the standard; or a
  GCS/file trajectory queue) + a param store (GCS + version polling) + actor/learner orchestration
  + networking. Much more than our single-program design.
- Staleness must be BOUNDED (actors refresh params every K games; optionally weight/window by
  recency) or learning slows/destabilizes. Add staleness metrics.
- Async nondeterminism -> harder reproducibility/debugging.

### Migration path (when we outgrow one slice / Multislice)
Everything below the loop reuses unchanged (env, net, backbones, search, mctx, trainer, Orbax):
1. Stand up a shared replay (Reverb or a simple sharded GCS queue) + a param store (GCS, versioned).
2. Actor process = loop{ pull latest params; `make_generate`; push flattened samples }.
3. Learner process = loop{ sample shared replay; `train_step`; every K steps publish params }.
4. Bound staleness (param refresh cadence); log actor-vs-learner param lag.

### Recommendation (sequencing)
- **Not now.** Actor-learner is premature until (a) the synchronous approach is CONFIRMED to learn
  (§15 games-to-strength) and (b) we hit the ceiling of one slice / Multislice. Building a
  replay+param service before knowing the recipe learns would be wasted.
- **For the ablation phase + first competitive run**, synchronous Anakin on spot (compile cache +
  GCS Orbax + auto-relaunch) and independent per-config runs on separate spot slices are enough.
- **Adopt Sebulba/actor-learner** as the deliberate scale-up once we want a single very large run
  with spot resilience across many slices — it's the proven large-scale-AZ / LLM-RL pattern, and
  AZ's staleness-tolerance makes it a good fit. Reference: Hessel et al. 2021, "Podracer
  architectures for scalable RL" (Anakin vs Sebulba); DeepMind Reverb for the replay service.
