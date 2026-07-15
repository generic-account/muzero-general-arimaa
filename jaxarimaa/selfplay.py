"""Vectorized self-play, sharded across the device mesh.

`_rollout` plays a per-device batch of games with lax.scan (search -> record ->
step -> auto-reset), then computes value targets by a reverse scan (game outcome
back-propagated, net-value bootstrap for truncated tails).

`make_generate` wraps `_rollout` in `shard_map` over the 'data' axis so EACH device
plays a DISTINCT subset of games (folding the device's global axis index — and the
host's process index — into the rng). On 1 device this is a no-op. This is what makes
self-play actually scale with the slice (otherwise every chip replays the same games).
"""

import typing

import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P

from . import constants as C
from . import env as jenv
from . import search


class SPKnobs(typing.NamedTuple):
    """Self-play scalar knobs. A NamedTuple (not a bare tuple) so call sites can't
    silently misorder them; hashable, so it bakes at trace time like a tuple."""
    resign_thresh: float = 0.9
    full_prob: float = 0.25       # playout-cap: fraction of moves that get full sims
    fast_sims: int = 8            # sims for the cheap (untrained, not stored) moves
    greedy_after: int = 0         # play argmax after this many completed turns (0 = off)
    # --- Stage-2 knobs (all inert at defaults) ---
    dense_k: int = 32             # horizon (steps) for capture-in-k / material-trajectory
    surprise_w: float = 0.0       # per-row loss weight 1 + s*KL(target||prior)
    prior_temp: float = 1.0       # >1 flattens priors fed to search (anti-sharpening)
    deblunder_threshold: float = 0.15
    deblunder_width: float = 0.15
    # --- Stage-2.2 knobs (inert at defaults; see FeaturesConfig gates) ---
    ml_steer: float = 0.0         # moves-left steering weight inside search
    resolve_steps: int = 0        # policy-only rollout budget for truncated games
    qmix_lambda: float = 1.0      # weight on OUTCOME in value target (1 = off)
    handicap_frac: float = 0.0    # fraction of games starting a piece down


def _rollout(model, params, rng, batch, max_steps, mcts, features, sp_knobs):
    """Self-play `batch` games for `max_steps`; return dict of [T, batch, ...].

    Feature-gated extras: playout-cap randomization (a static count of full-sim
    moves per game, only those stored for training; the rest use cheap `fast_sims`),
    resign/adjudication (end a decided game early and reset the lane), truncation
    adjudication (material_eval instead of net bootstrap for unfinished tails), and
    a greedy-after-N-turns switch for decisive play.
    """
    num_sims, max_considered = mcts
    (resign_thresh, full_prob, fast_sims, greedy_after,
     dense_k, surprise_w, prior_temp, db_thresh, db_width,
     ml_steer, resolve_steps, qmix_lambda, handicap_frac) = sp_knobs
    playout_cap = features is not None and features.playout_cap
    resign = features is not None and features.resign
    if features is not None and features.fast_search:
        from . import fast_search as search_impl
    else:
        search_impl = search
    rng, kinit = jax.random.split(rng)
    states = jax.vmap(jenv.init_state)(jax.random.split(kinit, batch))
    if (features is not None and features.handicap_games
            and handicap_frac > 0.0):
        # KataGo-style handicap: a fraction of games start one NON-RABBIT piece
        # down on a random side — decisive, honestly-labeled games at any
        # strength (the outcome-signal source that never runs dry).
        rng, kdo, kside, kpick = jax.random.split(rng, 4)
        do = jax.random.uniform(kdo, (batch,)) < handicap_frac
        side = jax.random.bernoulli(kside, 0.5, (batch,))       # True = silver
        b = states.board                                        # [B,8,8] int8
        cand = jnp.where(side[:, None, None],
                         (b >= 8) & (b <= 12),                  # silver non-rabbit
                         (b >= 2) & (b <= 6))                   # gold non-rabbit
        g = jax.random.gumbel(kpick, b.shape)
        flat = jnp.where(cand, g, -jnp.inf).reshape(batch, -1)
        pick = jnp.argmax(flat, axis=-1)
        removed = b.reshape(batch, -1).at[jnp.arange(batch), pick].set(0)
        removed = removed.reshape(b.shape)
        newb = jnp.where((do & jnp.any(cand, axis=(1, 2)))[:, None, None],
                         removed, b)
        # re-seed the repetition ring: init_state stored the un-handicapped hash
        h0 = jax.vmap(lambda bb: jenv.position_hash(bb, jnp.int8(0)))(newb)
        states = states.replace(
            board=newb, turn_start_board=newb,
            rep_hist=states.rep_hist.at[:, 0].set(h0))

    def _search(sims):
        def branch(operand):
            s, k = operand
            out = search_impl.run_search(model, params, k, s, sims, max_considered,
                                         features, prior_temp, ml_steer)
            root_cv = out.search_tree.children_values[:, 0]
            root_vis = out.search_tree.children_visits[:, 0] > 0
            # deblunder raw data: search Q of the played action vs the best
            # VISITED alternative (root-player perspective, like node_values)
            q_chosen = jnp.take_along_axis(root_cv, out.action[:, None], 1)[:, 0]
            q_best = jnp.max(jnp.where(root_vis, root_cv, -jnp.inf), axis=-1)
            if features is not None and features.prune_policy_targets:
                # Close the Q-imputation channel: target mass only on actions
                # the search actually visited, renormalized (KataGo-style
                # pruning adapted to Gumbel action_weights).
                w = jnp.where(root_vis, out.action_weights, 0.0)
                weights = w / jnp.maximum(w.sum(-1, keepdims=True), 1e-8)
            elif features is not None and features.visit_policy_targets:
                # optima/AZ-style target: normalized root visit counts. Unvisited
                # actions get ZERO mass — no Q-imputation channel, so an OOD
                # value head can only reorder the visited few, not reweight the
                # whole prior. (Gumbel action_weights stay in use for greedy
                # argmax play via `weights` either way.)
                v = out.search_tree.children_visits[:, 0].astype(jnp.float32)
                weights = v / jnp.maximum(v.sum(-1, keepdims=True), 1.0)
            else:
                weights = out.action_weights
            if surprise_w > 0.0:
                # KataGo policy-surprise weighting: rows where search genuinely
                # disagreed with the prior teach the most.
                pri = jax.nn.log_softmax(
                    out.search_tree.children_prior_logits[:, 0], axis=-1)
                wt = weights.astype(jnp.float32)
                kl = jnp.sum(jnp.where(wt > 0, wt * (jnp.log(wt + 1e-9) - pri), 0.0),
                             axis=-1)
                row_w = 1.0 + surprise_w * kl
            else:
                row_w = jnp.ones_like(q_chosen)
            return (out.action, weights, out.search_tree.node_values[:, 0],
                    q_chosen, q_best, row_w)
        return branch

    # Playout-cap randomization with a STATIC count: exactly `n_full` of the T
    # steps use the full-sim search (chosen at random positions); ONLY those steps
    # are stored for training. The old per-step bernoulli produced ~75% weight-0
    # buffer rows that diluted every training batch 4x.
    n_full = max(1, round(full_prob * max_steps)) if playout_cap else max_steps
    if playout_cap:
        rng, kperm = jax.random.split(rng)
        full_steps = jnp.zeros((max_steps,), bool).at[
            jax.random.permutation(kperm, max_steps)[:n_full]].set(True)
    else:
        full_steps = jnp.ones((max_steps,), bool)

    def body(carry, is_full):
        states, rng = carry
        if playout_cap:
            rng, ks, kr = jax.random.split(rng, 3)
            action, weights, root_v, q_chosen, q_best, row_w = jax.lax.cond(
                is_full, _search(num_sims), _search(fast_sims), (states, ks))
        else:
            rng, ks, kr = jax.random.split(rng, 3)
            action, weights, root_v, q_chosen, q_best, row_w = _search(num_sims)((states, ks))

        if greedy_after:
            # Decisive play after the opening (optima's temp->0 @ move 15): switch
            # from Gumbel's exploration pick to the argmax of the improved policy.
            # rep_ptr starts at 1 and increments per finished turn.
            greedy = states.rep_ptr - 1 >= greedy_after
            action = jnp.where(greedy, jnp.argmax(weights, -1).astype(action.dtype),
                               action)

        rec = {
            # bf16 storage: obs values (0/1 and quarters) and softmax policy targets
            # are bf16-exact/safe; halves the two largest scan/replay tensors.
            "obs": jax.vmap(lambda s: jenv.observe(s, features))(states).astype(jnp.bfloat16),
            "policy_target": weights.astype(jnp.bfloat16),
            "player": states.player,
        }
        if features is not None and features.deblunder:
            rec["q_chosen"] = q_chosen.astype(jnp.float32)
            rec["q_best"] = q_best.astype(jnp.float32)
        if features is not None and features.qmix_value:
            rec["root_v"] = root_v.astype(jnp.float32)  # mover-persp search value
        if surprise_w > 0.0:
            rec["weight"] = row_w.astype(jnp.float32)
        nstates = jax.vmap(jenv.step)(states, action)
        if features is not None and features.dense_aux:
            # Per-step capture events for the dense targets, GOLD's perspective:
            # +1 = a silver piece was captured (gold gains), -1 = gold captured.
            def side_count(b, lo, hi):
                return jnp.sum((b >= lo) & (b <= hi), axis=(-2, -1))
            g_lost = (side_count(states.board, 1, 6)
                      - side_count(nstates.board, 1, 6)) > 0
            s_lost = (side_count(states.board, 7, 12)
                      - side_count(nstates.board, 7, 12)) > 0
            traps = []
            for (tx, ty) in C.TRAPS_XY:
                prev_t = states.board[:, ty, tx]
                next_t = nstates.board[:, ty, tx]
                gold_cap = (prev_t >= 1) & (prev_t <= 6) & (next_t == 0) & g_lost
                silv_cap = (prev_t >= 7) & (next_t == 0) & s_lost
                traps.append(silv_cap.astype(jnp.int8) - gold_cap.astype(jnp.int8))
            rec["trap_cap"] = jnp.stack(traps, axis=-1)          # int8 [B, 4]
            rec["mat_delta"] = (s_lost.astype(jnp.int8)
                                - g_lost.astype(jnp.int8))       # int8 [B]
        if resign:
            # Resign only off FULL-search root values: fast (8-sim) estimates are
            # noisy and would adjudicate games spuriously, corrupting the stored
            # full rows' targets upstream of them.
            adj = is_full & (jnp.abs(root_v) > resign_thresh)
            belief_adj = adj & (~nstates.terminated)  # real terminal wins
            adj_winner = jnp.where(root_v > 0, states.player,
                                   1 - states.player).astype(jnp.int8)
            nstates = nstates.replace(
                terminated=nstates.terminated | adj,
                winner=jnp.where(belief_adj, adj_winner, nstates.winner))
            rec["belief_adj"] = belief_adj
        else:
            rec["belief_adj"] = jnp.zeros_like(nstates.terminated)
        rec["term"] = nstates.terminated
        rec["winner"] = nstates.winner
        fresh = jax.vmap(jenv.init_state)(jax.random.split(kr, batch))
        nstates = jenv.where_state(nstates.terminated, fresh, nstates)  # auto-reset
        return (nstates, rng), rec

    (final_states, _), recs = jax.lax.scan(body, (states, rng), full_steps)

    # Truncated-tail resolution (stage-2.2): finish cap-truncated games with
    # search-free policy-only play (~1 net eval/step, ~50x cheaper than a
    # searched step). A resolved game contributes a REAL outcome label; the
    # false-0 "draw" labels of unfinished games were measured to teach the
    # value head agnosticism (memory: warmstart-stage2).
    res_done = jnp.zeros((batch,), bool)
    res_out = jnp.zeros((batch,), jnp.float32)
    if (features is not None and features.rollout_resolve
            and resolve_steps > 0):
        def rbody(s, _):
            obs, legal = jax.vmap(
                lambda st: jenv.observe_and_mask(st, features))(s)
            logits, _, _ = jax.vmap(lambda o: model.apply(params, o))(obs)
            a = jnp.argmax(jnp.where(legal, logits, -jnp.inf), axis=-1)
            ns = jax.vmap(jenv.step)(s, a)
            ns = jenv.where_state(s.terminated, s, ns)
            return ns, None
        rstates, _ = jax.lax.scan(rbody, final_states, None,
                                  length=resolve_steps)
        res_done = rstates.terminated
        res_out = jnp.where(
            res_done,
            jnp.where(rstates.winner == final_states.player, 1.0, -1.0),
            0.0).astype(jnp.float32)

    # Value carried into truncated tails: material/advancement adjudication (a
    # grounded, discriminative signal — breaks the self-confirming near-zero
    # bootstrap loop) or, when the feature is off, the net's own value.
    if features is not None and features.truncation_draw:
        # optima-style: hitting the step cap scores as a REAL draw (0), giving
        # the value head a true (if bland) signal instead of a proxy. Resolved
        # games override the draw with their real rollout outcome.
        boot_val = res_out
        grounded0 = jnp.ones((batch,), bool)
    elif features is not None and features.adjudicate_truncation:
        boot_val = jnp.where(res_done, res_out,
                             jax.vmap(jenv.material_eval)(final_states))
        grounded0 = res_done
    else:
        fobs = jax.vmap(lambda s: jenv.observe(s, features))(final_states)
        _, net_boot, _ = jax.vmap(lambda o: model.apply(params, o))(fobs)
        boot_val = jnp.where(res_done, res_out, net_boot)
        grounded0 = res_done

    # Reverse scan producing, per step from the side-to-move perspective:
    #  value_target in [-1,1] (terminal -> outcome; else next value sign-flipped iff the
    #  mover changed; truncated tail -> bootstrap), and moves_left_target = normalized
    #  plies to game end (terminal -> 0; else next+1; capped; truncated tail -> capped).
    MLCAP = C.MOVES_LEFT_CAP

    deblunder = features is not None and features.deblunder

    def back(carry, step):
        next_player, v, ml, grounded, db_v, db_w = carry
        player, term, winner = step["player"], step["term"], step["winner"]
        outcome = jnp.where(winner == player, 1.0, -1.0)
        sign = jnp.where(player == next_player, 1.0, -1.0)
        v_t = jnp.where(term, outcome, sign * v).astype(jnp.float32)
        ml_t = jnp.where(term, 0.0, jnp.minimum(ml + 1.0, MLCAP))
        # Grounded = the value target descends from a REAL terminal. A resign
        # adjudication (belief_adj) ends the game but its "outcome" is the net's
        # OWN belief — marking it real would train the value head on itself at
        # full weight (the self-confirming poison value_tail_weight exists to
        # stop). It propagates as the target but stays value_real=0. Every
        # terminal RESETS the carry (games are lane-concatenated; a later game's
        # real terminal must not ground an earlier game's resign prefix).
        g_t = jnp.where(term, ~step["belief_adj"], grounded)
        if deblunder:
            # optima/KataGo Q-mix: positions BEFORE an exploration blunder take
            # value targets mixed toward the PRE-blunder search estimate (what
            # the outcome would have been under good play) instead of the noisy
            # realized outcome. db_v flips perspective like v; terminals reset
            # the carry (blunders don't cross game boundaries in a lane).
            db_v_here = (sign * db_v).astype(jnp.float32)
            db_w_here = jnp.where(term, 0.0, db_w)
            v_out = (1.0 - db_w_here) * v_t + db_w_here * db_v_here
            # does THIS step's mover blunder? (nearest-downstream blunder wins:
            # overwrite the carry for earlier steps)
            gap = step["q_best"] - step["q_chosen"]
            w_new = jnp.clip((gap - db_thresh) / jnp.maximum(db_width, 1e-6),
                             0.0, 1.0)
            is_bl = (w_new > 0.0) & (~term)
            db_v_next = jnp.where(is_bl, step["q_best"], db_v_here)
            db_w_next = jnp.where(is_bl, w_new, db_w_here)
            return ((player, v_t, ml_t, g_t, db_v_next, db_w_next),
                    (v_out, (ml_t / MLCAP).astype(jnp.float32), g_t))
        return ((player, v_t, ml_t, g_t, db_v, db_w),
                (v_t, (ml_t / MLCAP).astype(jnp.float32), g_t))

    scan_steps = {"player": recs["player"], "term": recs["term"],
                  "winner": recs["winner"], "belief_adj": recs["belief_adj"]}
    if deblunder:
        scan_steps["q_chosen"] = recs["q_chosen"]
        scan_steps["q_best"] = recs["q_best"]
    _, (value_target, moves_left_target, value_real) = jax.lax.scan(
        back,
        (final_states.player, boot_val.astype(jnp.float32),
         jnp.full(boot_val.shape, MLCAP, jnp.float32),
         grounded0,
         jnp.zeros((batch,), jnp.float32),   # db_v
         jnp.zeros((batch,), jnp.float32)),  # db_w
        scan_steps,
        reverse=True,
    )
    if (features is not None and features.qmix_value
            and qmix_lambda < 1.0):
        # TD-style variance reduction (stage-2.2): mix the noisy realized
        # outcome with the root search value at each position. At equal
        # strength, outcomes are near coin flips; root-Q is a far lower-
        # variance estimate. Terminal rows keep their exact outcome.
        mix = (qmix_lambda * value_target
               + (1.0 - qmix_lambda) * recs["root_v"])
        value_target = jnp.where(recs["term"], value_target, mix)

    # value_target is the standard MC game-outcome target (terminal -> +/-1,
    # non-terminal -> outcome from that mover's view, truncated tail -> adjudicated
    # bootstrap). The old search-root-value blend was a cold-start crutch (pure
    # outcome collapsed when games never finished); with a pretrained/grounded
    # value head and games that terminate, ground-truth outcome is the right
    # target and lets value learn past the teacher.

    # Only store aux targets when their head is enabled — otherwise they are dead
    # weight in the replay buffer (HBM). Baseline (heads off) carries neither.
    out = {
        "obs": recs["obs"],
        "policy_target": recs["policy_target"],
        "value_target": value_target,
        # 1.0 = value target descends from a REAL terminal; 0.0 = bootstrapped/
        # adjudicated tail. trainer scales the VALUE loss by this (see
        # value_tail_weight) — crude tail targets churned the trunk (probe).
        "value_real": value_real.astype(jnp.float32),
    }
    if surprise_w > 0.0:
        out["weight"] = recs["weight"]  # loss_fn's per-row weight hook
    if features is not None and features.dense_aux:
        # Dense targets via a second reverse scan (EWMA of future events, reset
        # at terminals; horizon set by dense_k -> gamma = 1 - 1/k):
        #   trap_own[4]  in [-1,1]: discounted future capture flow per trap
        #   cap_soon[2]  in [0,1]:  decayed will-lose-a-piece indicator (me/opp)
        #   mat_traj[1]  in [-1,1]: discounted future material swing
        # All emitted in the MOVER's perspective per row.
        gamma = 1.0 - 1.0 / float(dense_k)

        def dback(carry, step):
            own, cg, cs, mat = carry
            ev = step["trap_cap"].astype(jnp.float32)           # [B,4] gold persp
            g_lost = jnp.any(step["trap_cap"] == -1, axis=-1).astype(jnp.float32)
            s_lost = jnp.any(step["trap_cap"] == 1, axis=-1).astype(jnp.float32)
            md = step["mat_delta"].astype(jnp.float32)          # gold persp
            reset = step["term"][:, None].astype(jnp.float32)
            r1 = step["term"].astype(jnp.float32)
            own_t = ((1 - gamma) * ev + gamma * own) * (1 - reset) + (1 - gamma) * ev * reset
            cg_t = jnp.maximum(g_lost, gamma * cg * (1 - r1))
            cs_t = jnp.maximum(s_lost, gamma * cs * (1 - r1))
            mat_t = (1 - gamma) * md + gamma * mat * (1 - r1)
            pl = step["player"].astype(jnp.float32)             # 0 gold, 1 silver
            flip = 1.0 - 2.0 * pl                               # +1 gold, -1 silver
            dense_row = jnp.concatenate([
                own_t * flip[:, None],                          # mover persp traps
                jnp.where(pl > 0, cs_t, cg_t)[:, None],         # I lose a piece soon
                jnp.where(pl > 0, cg_t, cs_t)[:, None],         # opp loses soon
                (mat_t * flip)[:, None],                        # mover material traj
            ], axis=-1)
            return (own_t, cg_t, cs_t, mat_t), dense_row

        B4 = jnp.zeros((batch, 4), jnp.float32)
        B1 = jnp.zeros((batch,), jnp.float32)
        _, dense_target = jax.lax.scan(
            dback, (B4, B1, B1, B1),
            {"trap_cap": recs["trap_cap"], "mat_delta": recs["mat_delta"],
             "term": recs["term"], "player": recs["player"]},
            reverse=True)
        out["dense_target"] = dense_target.astype(jnp.float32)  # [T,B,7]
    if features is not None and features.moves_left_head:
        out["moves_left_target"] = moves_left_target
    if features is not None and features.mtp:
        # MTP target: the NEXT step's value; masked at the last step and at game
        # boundaries (where the next step belongs to a freshly-reset game).
        T = value_target.shape[0]
        out["mtp_value_target"] = jnp.concatenate([value_target[1:], value_target[-1:]], 0)
        not_last = (jnp.arange(T) < T - 1)[:, None]
        out["mtp_mask"] = (not_last & (~recs["term"])).astype(jnp.float32)
    # Completion telemetry: how many games reached a terminal this rollout.
    # Each lane leaves ~1 unfinished game at scan end, so the fraction of
    # started games that completed is terminals / (terminals + batch).
    terminals = jnp.sum(recs["term"].astype(jnp.int32)).reshape(1)

    if not playout_cap:
        return out, terminals  # every step is a full-search move; no filtering
    # Keep only the full-search timesteps (static count n_full): fast-move rows
    # aren't trained on. jnp.nonzero with static size keeps shapes fixed.
    idx = jnp.nonzero(full_steps, size=n_full)[0]
    return {k: v[idx] for k, v in out.items()}, terminals


def make_generate(mesh, model, batch_size, max_steps, mcts, features=None,
                  sp_knobs=SPKnobs()):
    """Build a jitted, sharded self-play function `(params, rng) -> recs [T,B,...]`.

    Compiled once and reused across iterations (model/sizes/features/knobs are static).
    `sp_knobs` is an SPKnobs (see above).
    """
    n = mesh.shape["data"]
    if batch_size % n:
        raise ValueError(f"batch_size {batch_size} must divide mesh size {n}")
    per = batch_size // n

    def per_shard(params, rng):
        r = jax.random.fold_in(rng, jax.lax.axis_index("data"))
        return _rollout(model, params, r, per, max_steps, mcts, features, sp_knobs)

    sharded = jax.shard_map(per_shard, mesh=mesh, in_specs=(P(), P()),
                            out_specs=(P(None, "data"), P("data")),
                            check_vma=False)
    jitted = jax.jit(sharded)

    def generate(params, rng):
        """-> (recs [T,B,...], completed_frac scalar in [0,1])."""
        recs, terms = jitted(params, jax.random.fold_in(rng, jax.process_index()))
        n_term = float(jnp.sum(terms))
        return recs, n_term / (n_term + batch_size)

    return generate


def flatten_samples(recs):
    """[T, B, ...] -> [B*T, ...], keeping the 'data'-sharded game axis leading and
    contiguous (swap T/B first) so downstream sharding stays clean."""
    def f(x):
        x = jnp.swapaxes(x, 0, 1)              # [B, T, ...]
        return x.reshape((-1,) + x.shape[2:])  # [B*T, ...]
    return {k: f(v) for k, v in recs.items()}
