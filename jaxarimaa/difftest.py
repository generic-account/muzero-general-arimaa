"""Differential-testing infrastructure: legacy engine (oracle) vs JAX engine.

The legacy Python engine ``games/arimaa.py`` is battle-tested and byte-for-byte
authoritative. We validate the vectorized JAX engine by driving BOTH from the same
positions and asserting they agree on: observation encoding, the legal-action mask,
and per-action state transitions.

Positions are sampled by playing random legal games with the legacy engine, so the
distribution is realistic (traps sprung, pieces frozen, pushes/pulls, near-goal).

Checks whose JAX side is still a stub report SKIPPED rather than failing, so this
harness is runnable now and grows coverage as env.py is filled in.
"""

import numpy as np

from games import arimaa as legacy
from . import constants as C
from . import env as jenv


# ---------------------------------------------------------------------------
# Bridge: legacy Board -> jaxarimaa cell-code array
# ---------------------------------------------------------------------------
def array_from_legacy_board(board) -> np.ndarray:
    """legacy Board -> int8 [8,8] cell-code array (0 empty; 1..6 gold; 7..12 silver)."""
    arr = np.zeros((C.BOARD, C.BOARD), np.int8)
    for y in range(C.BOARD):
        for x in range(C.BOARD):
            piece = board[(x, y)]
            if piece is not None:
                color, rank = legacy.parse_piece(piece)
                arr[y, x] = C.cell_code(color, rank)
    return arr


def state_of(s) -> "object":
    """Build a JAX State from a snapshot, propagating `terminated`."""
    return jenv.state_from_board(
        s["board"], s["player"], s["steps_left"], terminated=s.get("terminated", False)
    )


def snapshot(env) -> dict:
    """Capture the legacy env's current position + ground-truth observation."""
    return {
        "board": array_from_legacy_board(env.board),
        "player": int(env.board.state.player),
        "steps_left": int(env.board.state.left),
        "terminated": bool(env.board.state.end),
        "obs": np.asarray(env.get_observation()),
        "legal": sorted(env.legal_actions()),
    }


def random_positions(n_games=20, max_steps=400, seed=0) -> list:
    """Play random legal games with the legacy engine; snapshot every position."""
    snaps = []
    rng = np.random.RandomState(seed)
    for g in range(n_games):
        env = legacy.ArimaaEnv(seed=int(rng.randint(1 << 30)))
        env.reset()
        snaps.append(snapshot(env))
        for _ in range(max_steps):
            legal = env.legal_actions()
            if not legal:
                break
            a = int(rng.choice(legal))
            _, _, done = env.step(a)
            snaps.append(snapshot(env))
            if done:
                break
    return snaps


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------
def check_action_table_parity() -> tuple[bool, str]:
    """The JAX action tables must be a bijection with the legacy action indexing."""
    if C.N_ACTIONS != len(legacy.ACTION_LIST) + 1:
        return False, f"N_ACTIONS {C.N_ACTIONS} != legacy {len(legacy.ACTION_LIST)+1}"
    if C.END_TURN != legacy.ACTION_END_TURN:
        return False, f"END_TURN {C.END_TURN} != legacy {legacy.ACTION_END_TURN}"
    # spot-check a few specs round-trip to the tables
    t = C.ACTION_TABLES
    for i in (0, 1, 500, len(legacy.ACTION_LIST) - 1):
        spec = legacy.ACTION_LIST[i]
        if (int(t["frm_x"][i]), int(t["frm_y"][i])) != tuple(spec.old_pos):
            return False, f"action {i} from mismatch"
        if (int(t["to_x"][i]), int(t["to_y"][i])) != tuple(spec.new_pos):
            return False, f"action {i} to mismatch"
    return True, f"{C.N_ACTIONS} actions, END_TURN={C.END_TURN}"


def check_observation_parity(snaps) -> tuple[bool, str]:
    """JAX observe(state) must equal legacy get_observation() exactly."""
    bad = 0
    for s in snaps:
        st = state_of(s)
        got = np.asarray(jenv.observe(st))
        if not np.array_equal(got, s["obs"]):
            bad += 1
    ok = bad == 0
    return ok, f"{len(snaps)} positions, {bad} mismatches"


def check_legal_mask_parity(snaps) -> tuple[bool | None, str]:
    """JAX legal_action_mask must match legacy legal_actions() on all board-legal
    moves. The 3-fold-repetition restriction on END_TURN is stateful (game history)
    and intentionally NOT in the pure mask, so it is isolated here: END_TURN's
    *deterministic* condition (steps_left != 4) is checked exactly, and cases where
    the oracle additionally forbids END_TURN by repetition are reported separately.
    """
    try:
        st = state_of(snaps[0])
        jenv.legal_action_mask(st)
    except NotImplementedError:
        return None, "SKIPPED (env.legal_action_mask is a stub)"
    step_bad = end_bad = rep_blocked = 0
    for s in snaps:
        st = state_of(s)
        mask = np.asarray(jenv.legal_action_mask(st))
        legal = set(s["legal"])
        # step-actions [0, END_TURN): must match exactly
        jax_steps = set(np.nonzero(mask[:C.END_TURN])[0].tolist())
        if jax_steps != (legal & set(range(C.END_TURN))):
            step_bad += 1
        # END_TURN: compare against the deterministic condition
        det = (s["steps_left"] != 4)
        jax_end = bool(mask[C.END_TURN])
        oracle_end = C.END_TURN in legal
        if jax_end != det:
            end_bad += 1  # bug in the deterministic END_TURN logic
        elif det and not oracle_end:
            rep_blocked += 1  # oracle forbade END_TURN via repetition (expected)
    ok = step_bad == 0 and end_bad == 0
    return ok, (f"{len(snaps)} positions | step-action mismatches={step_bad} "
                f"end-det mismatches={end_bad} | repetition-blocked(info)={rep_blocked}")


def check_step_parity(snaps, max_snaps=400) -> tuple[bool | None, str]:
    """For each legal action, JAX step must reproduce the legacy transition: the
    board, and the turn/terminal bookkeeping. We drive legacy from a FRESH board
    (empty position_counts) so the stateful repetition rule never fires, making
    legacy's result purely board-based and therefore exactly comparable.

    Legacy overloads ``state.player`` to hold the winner on termination, so on
    terminal results we compare our ``winner`` to legacy ``state.player`` and skip
    the side-to-move comparison.
    """
    import jax
    try:
        st = state_of(snaps[0])
        jenv.step(st, 0)
    except NotImplementedError:
        return None, "SKIPPED (env.step is a stub)"
    jstep = jax.jit(jenv.step)
    board_bad = field_bad = term_bad = checked = terminals = 0
    for s in snaps[:max_snaps]:
        if s.get("terminated"):
            continue
        for a in s["legal"]:
            # END_TURN is included: a fresh legacy board has empty position_counts,
            # so its _finish_turn never triggers repetition -> purely board-based.
            env = legacy.ArimaaEnv(seed=0)
            env.board = _rebuild_legacy(s)
            env.position_counts = {}
            env.end_reason = None
            env.turn_progress = 0.0
            env.turn_steps_taken = 0
            env.step(a)
            leg_end = bool(env.board.state.end)
            leg_player = int(env.board.state.player)
            leg_left = int(env.board.state.left)
            want_board = array_from_legacy_board(env.board)

            ns = jstep(state_of(s), a)
            checked += 1
            if not np.array_equal(np.asarray(ns.board), want_board):
                board_bad += 1
            if bool(ns.terminated) != leg_end:
                term_bad += 1
            elif leg_end:
                terminals += 1
                if int(ns.winner) != leg_player:  # legacy player == winner on end
                    field_bad += 1
            else:
                if int(ns.player) != leg_player or int(ns.steps_left) != leg_left:
                    field_bad += 1
    ok = board_bad == 0 and field_bad == 0 and term_bad == 0
    return ok, (f"{checked} transitions ({terminals} terminal) | board={board_bad} "
                f"term-flag={term_bad} fields={field_bad}")


def _rebuild_legacy(snap):
    """Reconstruct a legacy Board from a snapshot's cell-code array + turn info."""
    board = legacy.Board()
    board.state.setup = False
    board.state.end = False
    board.state.player = snap["player"]
    board.state.left = snap["steps_left"]
    arr = snap["board"]
    for y in range(C.BOARD):
        for x in range(C.BOARD):
            code = int(arr[y, x])
            if code == 0:
                board[(x, y)] = None
            else:
                color, rank = C.decode_cell_code(code)
                board[(x, y)] = legacy.make_piece(color, rank)
    return board


def main():
    print("Sampling positions from the legacy oracle ...")
    snaps = random_positions(n_games=30, seed=0)
    print(f"  {len(snaps)} positions\n")
    checks = [
        ("action-table bijection", check_action_table_parity()),
        ("observation parity", check_observation_parity(snaps)),
        ("legal-mask parity", check_legal_mask_parity(snaps)),
        ("step-transition parity", check_step_parity(snaps)),
    ]
    allpass = True
    for name, (ok, msg) in checks:
        if ok is None:
            tag = "SKIP"
        elif ok:
            tag = "PASS"
        else:
            tag = "FAIL"; allpass = False
        print(f"[{tag}] {name}: {msg}")
    print("\n" + ("ALL RUNNABLE CHECKS PASSED" if allpass else "FAILURES PRESENT"))
    return 0 if allpass else 1


if __name__ == "__main__":
    raise SystemExit(main())
