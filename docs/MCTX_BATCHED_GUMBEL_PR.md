# Upstreaming `batched_gumbel_muzero_policy` to mctx — design spec

Status: **spec / pre-PR**. Target: `google-deepmind/mctx` @ `main`.
Source: **`jaxarimaa/mctx_batched.py`** (551 lines, pure jax + mctx, zero project
imports — programmatically asserted) + `jaxarimaa/tests_fast_search_v2.py`.
Project glue lives separately in `jaxarimaa/fast_search.py` (106 lines).
Extraction is **done**: the module is lift-and-drop ready.

---

## 0. Repo reconnaissance (2026-08, verified)

| Fact | Value |
|---|---|
| mctx stars / open issues | 2,649 / 7 |
| Last push | 2026-07-09 (`main` @ `450fbf7656b88dd1d8ca5b2db3a2f9464cb322f2`) |
| **Newest *installable* version** | **0.0.71** — what we pin. The "Release v0.1.9" commit (2026-06-12) never published: no `v0.1.9` git tag, PyPI's latest is 0.0.71, and a "Fix the pypi release hook" commit followed three days later. There is no newer release to port to. |
| Archived | No |
| Semantic drift in `_src/{search,action_selection,qtransforms,tree}.py` since our pinned 0.0.71 | **None.** Every diff is a `# pyrefly: ignore[...]` type-suppression comment. `seq_halving.py` and `base.py` byte-identical. |
| Every symbol we depend on present on `main` | Yes (`score_considered`, `get_sequence_of_considered_visits`, `instantiate_tree_from_root`, `gumbel_muzero_interior_action_selection`, `qtransform_completed_by_mix_value`, `Tree`, `masked_argmax`, `GumbelMuZeroExtraData`, `PolicyOutput`) |
| Public API (`mctx/__init__.py`) main vs pinned | **Byte-identical** |

**The search algorithm has not changed in ~2 years.** A PR written against our pinned
copy applies to `main` essentially unmodified.

### ⚠️ Prior art collision: open PR #116

`retretor`, "Avoid full-buffer copies in the backward pass of search" (opened
2026-07-16, open + mergeable, 1 file, +80/−24). It independently invents
**exactly** our `_backward_batched` technique:

> records the leaf-to-root path in a compact O(num_nodes) loop carry, then applies
> all tree updates with a single scatter after the loop. Unused path slots use index
> `num_nodes` and are dropped via scatter `mode="drop"`.

That is our implementation down to the sentinel index and the `mode="drop"` trick.
Their motivation is XLA:GPU copying whole `[num_nodes, num_actions]` buffers per
loop iteration; measured on RTX 5070 @ 1729 actions:

| sims | before | after | speedup |
|---|---|---|---|
| 16 | 29.3 ms | 27.0 ms | 1.08× |
| 32 | 72.3 ms | 48.5 ms | 1.49× |
| 64 | 394.7 ms | 120.8 ms | 3.27× |

Also: compile time 60 s → 22 s at 64 sims.

**Implications, all of them good except one:**
1. Convergent design by an independent party is strong evidence the technique is right.
2. Maintainers are entertaining performance PRs in exactly this file — de-risks ours.
3. **Drop the backward-pass rewrite from our PR.** It is their contribution. Ours
   should be written to *compose* with #116, not compete with it.
4. The two wins are **orthogonal and multiplicative**: #116 removes per-simulation
   buffer copies (tree-op axis); we remove sequential `recurrent_fn` calls
   (call-count axis). Their 394 ms → 121 ms at 64 sims is *before* any reduction in
   the number of simulation steps.

---

## 1. The algorithm as mctx implements it

Sequential Halving with Gumbel at the root (Danihelka et al., *Policy improvement by
planning with Gumbel*): sample `g(a)` once, score actions by
`g(a) + logits(a) + σ(q̂(a))`, visit the considered set, halve, repeat.

mctx flattens this into a per-simulation schedule. `seq_halving.get_sequence_of_considered_visits(m, n)`
returns, for each simulation index, the visit count an action must currently have to
be eligible. For `m=16, n=32`:

```
[0]*16, [1]*8, [2]*4, [3]*4        # 32 entries
```

`gumbel_muzero_root_action_selection` then does, per simulation:

```python
considered_visit = table[num_considered, simulation_index]
to_argmax = score_considered(considered_visit, gumbel, prior_logits,
                             completed_qvalues, visit_counts)   # -inf where visits != cv
return masked_argmax(to_argmax, tree.root_invalid_actions)
```

and `search.search` runs `num_simulations` iterations of `fori_loop`, each doing:
one vmapped `simulate` descent (`while_loop`), one `recurrent_fn` call at batch `B`,
`expand`'s scatters, one vmapped `backward` (`while_loop`).

**The cost that matters:** `n` sequential `recurrent_fn` calls at batch `B`. For
MuZero that is `n` dynamics-network evaluations; for AlphaZero-style use with a real
simulator it is `n` env steps. Either way it is the dominant term, and it is
serialized.

## 2. The regrouping

Run-length-encode the schedule into `(considered_visit, width)` rounds. Consecutive
simulations sharing a `cv` visit **distinct root actions** (each pick consumes an
action by incrementing its count past `cv`), so they are independent.

Measured with `_rounds_from_schedule`:

| n | m | sequential calls | rounds | reduction | order-independent rounds | halving rounds |
|---|---|---|---|---|---|---|
| 32 | 16 | 32 | 4 | **8.0×** | 2 | 2 |
| 50 | 16 | 50 | 12 | 4.2× | 9 | 3 |
| 128 | 32 | 128 | 27 | 4.7× | 23 | 4 |
| 200 | 32 | 200 | 46 | 4.3× | 42 | 4 |
| 800 | 32 | 800 | 155 | **5.2×** | 151 | 4 |
| 800 | 64 | 800 | 141 | 5.7× | 136 | 5 |
| 1600 | 64 | 1600 | 279 | 5.7× | 274 | 5 |

Per round: select all `w` actions at once (`top_k` instead of `w` sequential
argmaxes), descend all `w` subtrees in one nested-`vmap` `[K, B]` lockstep
`while_loop`, make **one** `recurrent_fn` call at `[B*w]`, do one scatter per tree
array, and one lockstep backup.

## 3. Why it is correct

- **Disjointness.** Each lane forces a distinct root action, so the `w` subtrees are
  disjoint below the root. Descent is read-only on the round-start tree snapshot, so
  lanes cannot observe each other.
- **Node allocation.** New indices come from the simulation counter
  (`sim_offset + arange(w) + 1`), distinct per lane by construction — matching
  mctx's "node first expanded on simulation *i* gets index *i*".
- **Non-root backups.** On-schedule each non-root node has exactly one visitor in a
  round, so mctx's expressions apply verbatim → bitwise identical.
- **Root backup.** mctx applies `w` sequential incremental means. The count-weighted
  mean is associative, so the closed form `(v·n + Σ_k leaf_k) / (n + w)` is
  mathematically identical; only fp summation order differs. Root *selection* reads
  `raw_values` and children stats, never `node_values`, so visit counts are unaffected.
- **Visit counts** are integer scatter-adds — exact.

## 4. Where it differs from mctx (disclose prominently)

**(a) Static round widths vs per-row schedule shrinking.** mctx computes
`num_considered = min(m, num_valid_actions)` *per row* and indexes a different table
row. Our rounds are static-width; rows with fewer valid actions re-visit their best
valid action for surplus slots (mctx itself re-expands at `max_depth` similarly).
Affects only rows near terminal states.

**(b) Within-round Q drift on halving rounds.** mctx recomputes `completed_qvalues`
after *every* simulation. In a halving round (`2w` eligible, `w` slots) the
transform's global terms — `visit_scale = maxvisit_init + max(visits)`, the rescale
min/max, the mixed value — move between picks, so mctx's sequential argmax can select
a different `w`-subset than our round-start `top_w`. Near-ties only.

**The quantitative mitigation is the strongest argument in the PR.** The number of
halving rounds is `log2(m) − 1` — **constant in the simulation budget** — while total
rounds grow with `n`. Two round classes are provably order-independent:

- **Round 0**: every action is unvisited, so `_complete_qvalues` fills the whole array
  with the mixed value, `_rescale_qvalues` maps a constant array to **0**, and
  `completed_qvalues ≡ 0`. Scores reduce to `gumbel + normalized_logits` — static.
- **Extra-visit rounds** (`width == previous width`): every eligible candidate is
  visited, so the *set* cannot differ regardless of order.

At `n=800, m=32`: **151 of 155 rounds are exact**; divergence is confined to 4 rounds
(≈3%), and the fraction → 0 as budget grows. With a drift-free qtransform
(`value_scale=0`) the entire search matches mctx's visit counts exactly — asserted by
the test suite on every case.

**Interpretive argument worth making (verify against the paper text first):** the
paper's phase formulation halves on a *single consistent snapshot* at phase
boundaries, which is what round-start `top_w` does. mctx's per-pick recomputation is
an implementation choice of the flat-loop encoding. If that reading holds, ours is
arguably the more faithful instantiation rather than an approximation — but do not
assert this in the PR without quoting the paper.

## 5. Costs and open design work

1. **Compile time / HLO size.** Our round loop is a Python `for`, so the round body
   **unrolls** — 27 bodies at `(128, 32)`, 155 at `(800, 32)`. This is the one place
   we likely regress where #116 improves (they report 60 s → 22 s).
   **Fix, and it is a genuine design improvement over the current code:** consecutive
   rounds frequently share a width (`(800,32)` begins `(0,32),(1,32),(2,32),(3,32),(4,32),(5,16)…`).
   Group same-width runs into a `lax.scan` over a stacked `cv` vector with one traced
   body — collapsing ~155 unrolled bodies into ~5 (one per distinct width). Do this
   *before* submitting.
2. **Peak memory.** A `[B*w]` `recurrent_fn` call uses `w×` the activation memory of
   mctx's `[B]` call (`w = m` on round 0). For large MuZero dynamics networks this can
   OOM where mctx would not. Needs a `max_lanes_per_call` knob that chunks a round.
3. **API surface.** mctx's `gumbel_muzero_policy` accepts `loop_fn` (for
   `hk.fori_loop`). We have no `fori_loop`, so either accept-and-ignore with a
   docstring note or reject it explicitly.
4. **`gumbel_scale=0.0`** (used for perfect-information eval) should be tested — it
   makes many scores exactly tied, which is precisely the regime where (b) bites.
5. **Extraction hygiene — COMPLETE (commit `a01f659`).** `mctx_batched.py` now holds
   only the v2 path and is verified free of project imports. `pack_states`/`unpack_states`
   and `run_search` moved to `fast_search.py`. The v3 `compact_*` path (built, proven
   exact, measured **12% slower**) was deleted from the live tree — recoverable at
   commit `e78e180` if bigger nets ever change the tradeoff.

   On `_backward_batched` vs #116: keep ours, but as a *documented generalization*.
   Theirs is single-leaf, where each path node has exactly one visitor, so plain
   `.set()` scatters reproduce sequential semantics. Round batching needs K leaves
   sharing the root, which forces scatter-**add** visit counts and the associative
   closed form for the root value. Round batching also affords a tighter footprint
   than #116 can reach: a leaf expanded in round *r* is at depth ≤ *r*+1, so the path
   scan is bounded by a static `num_hops` rather than `[num_nodes]` scratch. Naming
   now mirrors #116 (`path_parents`, `path_actions`, …) so a reviewer reads it as
   their function extended.

---

## 6a. Component audit — what the PR literally contains

`jaxarimaa/mctx_batched.py`, 551 lines including docstrings:

| component | lines | classification |
|---|---|---|
| `_rounds_from_schedule` | 19 | **Novel. The load-bearing insight** — RLE of mctx's own visit schedule into `(cv, width)` rounds. |
| `_mask_invalid_actions` | 9 | **Duplicate.** Logic identical to `mctx._src.policies._mask_invalid_actions` (ours is a reindent minus their `chex.assert_equal_shape`). On upstream: import theirs, delete ours. |
| `_write_nodes_batched` | 41 | Novel. `expand` + `update_tree_node` tails for a whole round: K per-array scatters → one. |
| `_backward_batched` | 105 | **K-leaf generalization of PR #116's technique** (see below). |
| `_completed_q_and_score_subset` | 59 | Novel optimization: reproduces the default qtransform's global terms from the `considered` columns, so per-round root scoring is O(m) not O(num_actions). Now directly gated (§6b). |
| `_make_forced_simulate` | 55 | Novel. `simulate` with a forced depth-0 action, doubly-vmapped `[K, B]` lockstep. |
| `batched_gumbel_muzero_policy` | 162 | Public entry point. Signature parity with `gumbel_muzero_policy`: **10 of 11 params**, only `loop_fn` absent (no `fori_loop` to parameterize). |

**mctx surface consumed:** `Tree` (+ its `ROOT_INDEX`/`UNVISITED`/`NO_PARENT`),
`score_considered`, `get_sequence_of_considered_visits`,
`instantiate_tree_from_root`, `gumbel_muzero_interior_action_selection`,
`masked_argmax`, `GumbelMuZeroExtraData`, `qtransform_completed_by_mix_value`,
`PolicyOutput`. All in `_src` — which is *why upstreaming is the right home*: as
an external package we must reach into mctx internals (fragile across releases);
inside mctx these are ordinary internal calls.

## 6b. DECISION: independent additive PR — do **not** base on #116

The audit settles this with a fact I had assumed the other way earlier. Our only
**call** into `search.py` is `instantiate_tree_from_root` (one line). The
`expand`/`backward`/`simulate` mentions in this module are *docstrings* naming
what each function mirrors — not invocations; the round-batched loop never calls
`search.search`, `search.expand`, `search.backward`, or `search.simulate`.

PR #116's diff is two hunks, both inside the `backward` region (`@@ -246,6` adds
`_BackwardState`; `@@ -259,40` rewrites `backward`). It does not touch
`instantiate_tree_from_root` or anything else we consume.

⇒ **Zero shared modified code. Zero merge conflict. Zero dependency.** Our PR is
purely additive: one new module.

Rejected alternatives, with reasons:
- *Branch off #116.* Couples our merge to an unmerged PR and forces rebases if
  theirs changes in review, for no technical benefit — we don't use their code.
- *Unify the two backwards into one function serving K=1 and K>1.* Their
  single-leaf `.set()` path is simpler and cheaper for K=1; generalizing would
  impose scatter-add and closed-form-mean overhead on the sequential path and
  balloon the review surface. If a maintainer wants unification later, it is a
  clean follow-up, not a precondition.

What we **do** owe #116: a citation in the PR body crediting the independently
derived path-record + single-scatter technique, and an explicit statement that
the two changes compose (they speed up the sequential loop's backward; we reduce
how many sequential steps exist at all).

## 6c. Pre-PR checklist

- [x] **`lax.scan` grouping of same-width rounds — DONE (commit `db4e2c1`).**
      Round 0 is peeled (it needs the full-width path and establishes the
      `considered` bookkeeping); the remaining rounds are grouped into
      consecutive same-width runs, each run scanned over a stacked `cv` vector
      with one traced body. The schedule's widths are non-increasing, so
      grouping-by-run == grouping-by-width. Each group uses ONE static hop bound
      taken from its deepest round; over-running the backward walk is exact
      (lanes deactivate at the root and their records are dropped by the masked
      scatter), it only costs a few extra `[B, K]` gathers on a group's earlier
      rounds. **Verified bitwise identical to the unrolled version** across 36
      arrays (action / action_weights / visit_counts / node_values /
      children_values / children_visits) × 3 `(n,m)` × 2 steering settings.

      Measured (CPU lowering, tiny net, so numbers isolate graph size):

      | n/m | rounds | StableHLO lines | optimized HLO | compile |
      |---|---|---|---|---|
      | 32/16 | 4 | 12,127 → 9,575 (1.3×) | 20,602 → 17,501 | 1.6 s → 1.3 s |
      | 128/32 | 27 | 72,570 → 15,543 (**4.7×**) | 125,469 → 27,566 | 8.4 s → 2.2 s (3.8×) |
      | 800/32 | 155 | 404,906 → 18,170 (**22.3×**) | 708,688 → 32,978 (21.5×) | 77.9 s → 2.7 s (**28.9×**) |

      The structural point matters more than any single row: graph size now grows
      **1.9×** from n=32 to n=800, versus **33.4×** unrolled. This converts the
      one axis where we would have regressed relative to #116 (which reports
      60 s → 22 s at 64 sims) into a decisive advantage, and it makes large
      simulation budgets practical at all.
- [ ] Chunking knob (`max_lanes_per_call`) — `[B*w]` uses `w×` the activation
      memory of mctx's `[B]` call; can OOM a large dynamics net.
- [ ] Import mctx's `_mask_invalid_actions`; delete our copy.
- [ ] Decide `loop_fn`: accept-and-ignore with a docstring note, or reject.
- [ ] Test `gumbel_scale=0.0` (perfect-information eval) — it makes scores
      exactly tied, the worst case for divergence class (b).
- [x] **Direct gate on `_completed_q_and_score_subset`** (commit `297a0a1`).
      The audit found this 59-line hand-derived algebra was **untested**: the
      drift-free exactness test passes `functools.partial(qtransform,
      value_scale=0.0)`, which fails the `is qtransform_completed_by_mix_value`
      identity check and therefore takes the general full-width branch, never
      the subset path. Under the default transform the subset path *is* live but
      divergence from mctx is expected anyway, so a bug there could have hidden
      as "drift". Added test compares the subset scores against
      `score_considered(vmap(qtransform)(...))` on a tree mctx itself built:
      **max|diff| = 0.0e+00 across cv ∈ {1,2,3,4}, `-inf` masks matching.** The
      algebra is exactly right — it just had no guard.
- [ ] Chex type annotations to match mctx house style.
- [ ] Standalone mctx-only benchmark (needed for the issue regardless).

## 6d. Recommended PR strategy

**Step 1 — Issue first, not a PR.** The change is large and raises a semantics
question (divergence class (b)) that only a maintainer can rule on. Open an issue
with: the round table from §2, the exactness argument from §3–4, a standalone
benchmark, and an explicit question — *"is an opt-in policy with documented
near-tie divergence acceptable, or must it be bit-exact to `gumbel_muzero_policy`?"*
Reference #116 and state that the two compose.

**Step 2 — Standalone benchmark** (needed for the issue anyway): mctx-only script,
synthetic `recurrent_fn` at a few cost levels, `gumbel_muzero_policy` vs ours across
the §2 grid, on CPU + whatever accelerator is available. Report call counts and wall
time separately so the mechanism is legible.

**Step 3 — PR**, if the maintainer signals yes: new module
`mctx/_src/policies_batched.py` exporting `batched_gumbel_muzero_policy`, additive
only, mirroring the `gumbel_muzero_policy` signature, plus the equivalence test suite
ported from `tests_fast_search_v2.py` (v1-vs-v2 exactness, drift-free-vs-mctx
exactness, validity on all rows including the off-schedule fallback).

**Fallback if rejected as too large:** contribute `_rounds_from_schedule` to
`seq_halving.py` as a documented utility (RLE of the existing schedule — trivially
reviewable, no semantics change) and publish the policy as a small companion
package. The utility is the load-bearing insight; the rest is mechanism.

---

## 7. Dev environment (set up 2026-08, verified)

| Item | Value |
|---|---|
| Fork | `generic-account/mctx` at `/Users/glichtstein/Documents/mctx` |
| Remotes | `origin` → fork (SSH), `upstream` → `google-deepmind/mctx` (HTTPS) |
| Sync | fork `main` == `upstream/main` == **`450fbf7`** — the exact SHA our code was validated against |
| Branch | `batched-gumbel-policy` |
| Venv | `/tmp/mctx-dev` (`pip install --editable ".[test]"` + pytest/pytest-xdist/flake8/pylint/pylint-exit). Recreate: `python3 -m venv /tmp/mctx-dev && /tmp/mctx-dev/bin/pip install -e ".[test]" pytest pytest-xdist flake8 pylint pylint-exit` |
| Baseline | **24/24 pytest**, **pylint 10.00/10**, **flake8 0 issues** on the pristine fork |

**Running their suite correctly:** `tree_test.py` opens
`../mctx/_src/tests/test_data/*.json` by *relative* path, so pytest must run from
a directory one level inside the repo root — that is why `test.sh` does
`mkdir _testing && cd _testing`. Running from anywhere else yields 4 spurious
`FileNotFoundError` failures.

### CI bar our module must clear (from `test.sh`)

1. `flake8 --select=E9,F63,F7,F82,E225,E251` — currently 0 issues repo-wide.
2. **`pylint --rcfile=.pylintrc -efail -wfail -cfail`** — fails on errors,
   warnings **and conventions**. Their own files score a clean **10.00/10**, so
   that is the bar, not "no errors".
3. `pytype -j auto --keep-going --disable import-error`.
4. `pytest --pyargs mctx`.
5. A clean-room `python -m build` + wheel install.

### House-style deltas our module needs (from `.pylintrc`)

- **`indent-string='  '` — mctx is TWO-SPACE indented; `mctx_batched.py` is
  four-space.** A full mechanical reindent is required, plus
  `indent-after-paren=4` for continuations.
- `max-line-length=80`.
- Docstrings on every public symbol (pylint conventions are fatal here).
- chex annotations + asserts to match density (their files: 8–12 `chex.assert*`,
  9–20 `chex.Array/Numeric/PRNGKey` annotations each; ours: **0 and 0**).

### Non-code blocker

`CONTRIBUTING.md` requires a signed **Google CLA** (<https://cla.developers.google.com/>)
before any PR can be merged. One-time, per-person, and needs to be done by the
contributor — not something that can be handled in-repo.
