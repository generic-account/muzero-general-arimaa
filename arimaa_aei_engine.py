#!/usr/bin/env python
"""AEI engine wrapper around our Arimaa implementation.

This exposes our bot to the Arimaa Engine Interface (AEI) so it can play against
other engines via AEI controllers (e.g. AEI's ``roundrobin`` / ``gameroom``).

Design: pyrimaa's ``Position`` is the authoritative game state -- it already
handles setup placement, push/pull step consumption, trap captures and legality.
We only use *our* engine (``games.arimaa``) to CHOOSE our move on our turn and to
render the chosen steps into AEI step notation. Setup placement is delegated to
pyrimaa's ``to_placing_move`` so it is always legal.

Policies:
  * ``random``  -- pick uniformly random legal steps (baseline / plumbing test)
  * ``greedy``  -- one-ply greedy on our material/progress evaluation
  * ``muzero``  -- (hook) load a MuZero checkpoint and run MCTS  [not yet wired]

Usage (as an AEI engine, driven by a controller over stdin/stdout):
    python arimaa_aei_engine.py --policy greedy

The heavy imports (torch/games) happen lazily so the AEI opening handshake is fast.
"""

import argparse
import random
import sys
from queue import Empty, Queue
from threading import Event, Thread


# ---------------------------------------------------------------------------
# AEI transport (mirrors third_party/AEI/pyrimaa/simple_engine.py)
# ---------------------------------------------------------------------------
class _ComThread(Thread):
    def __init__(self):
        Thread.__init__(self)
        self.stop = Event()
        self.messages = Queue()
        self.daemon = True

    def send(self, msg):
        sys.stdout.write(msg + "\n")
        sys.stdout.flush()

    def run(self):
        while not self.stop.is_set():
            try:
                msg = sys.stdin.readline()
            except AttributeError:
                return
            if msg == "":
                # EOF: controller closed the pipe
                self.messages.put("quit")
                return
            self.messages.put(msg.strip())


class AEIException(Exception):
    pass


# ---------------------------------------------------------------------------
# Move selection / rendering using our engine
# ---------------------------------------------------------------------------
def _alg(pos):
    """Our (x, y) square -> algebraic 'a8'..'h1' (y=0 is rank 8)."""
    x, y = pos
    return chr(97 + x) + str(8 - y)


def _dir(frm, to):
    """Direction char for an orthogonal one-square move, AEI convention."""
    fx, fy = frm
    tx, ty = to
    if tx == fx + 1:
        return "e"
    if tx == fx - 1:
        return "w"
    if ty == fy - 1:
        return "n"  # y decreases toward rank 8
    if ty == fy + 1:
        return "s"
    raise ValueError(f"non-adjacent step {frm}->{to}")


class MoveChooser:
    """Chooses a full turn's worth of steps on our board and renders AEI notation."""

    def __init__(
        self, policy="greedy", checkpoint=None, config_json=None, simulations=None
    ):
        # Lazy heavy import so the AEI handshake stays fast.
        from games import arimaa

        self.arimaa = arimaa
        arimaa.init_actions()
        self.policy = policy
        self.model = None
        self.mcts = None
        self.mcfg = None
        self.select_action = None
        self._jax = None
        if policy == "muzero":
            self._load_muzero(checkpoint, config_json, simulations)
        elif policy == "jax":
            self._load_jax(checkpoint, simulations)

    def _load_jax(self, checkpoint, simulations):
        """Load a jaxarimaa (JAX AlphaZero) checkpoint for inference."""
        import jax

        from jaxarimaa import checkpoint as jckpt
        from jaxarimaa import network as jnet
        from jaxarimaa.config import FeaturesConfig, NetConfig

        if not checkpoint:
            raise ValueError("--checkpoint is required for --policy jax")
        params, meta = jckpt.load(checkpoint)
        cfgd = meta.get("config", {})
        netcfg = NetConfig(**cfgd["net"]) if "net" in cfgd else NetConfig()
        feats = FeaturesConfig(**cfgd["features"]) if "features" in cfgd else FeaturesConfig()
        import jax.numpy as jnp
        dtype = jnp.bfloat16 if feats.bf16 else jnp.float32
        sims = simulations or cfgd.get("mcts", {}).get("num_simulations", 64)
        considered = cfgd.get("mcts", {}).get("max_num_considered_actions", 32)
        self._jax = {
            "jax": jax,
            "model": jnet.make_network(
                netcfg, dtype=dtype, moves_left_head=feats.moves_left_head,
                deep_supervision=feats.deep_supervision, mtp=feats.mtp,
                smolgen=feats.smolgen, rope=feats.rope),
            "params": params,
            "features": feats,
            "sims": int(sims),
            "considered": int(considered),
            "key": jax.random.PRNGKey(0),
        }

    def _load_muzero(self, checkpoint, config_json, simulations):
        import json

        import torch

        import models
        import self_play

        if not checkpoint:
            raise ValueError("--checkpoint is required for --policy muzero")
        cfg = self.arimaa.MuZeroConfig()
        if config_json:
            with open(config_json) as fh:
                for k, v in json.load(fh).items():
                    if hasattr(cfg, k):
                        setattr(cfg, k, v)
        if simulations:
            cfg.num_simulations = simulations
        # Inference is CPU-only in this wrapper.
        cfg.selfplay_on_gpu = False
        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
        weights = ckpt["weights"] if isinstance(ckpt, dict) and "weights" in ckpt else ckpt
        model = models.MuZeroNetwork(cfg)
        model.set_weights(weights)
        model.eval()
        torch.set_grad_enabled(False)
        self.model = model
        self.mcfg = cfg
        self.mcts = self_play.MCTS(cfg)
        self.select_action = self_play.SelfPlay.select_action

    def _board_from_short(self, color, short64):
        """Build our Board from a pyrimaa 64-char board string.

        pyrimaa's string is row-major (index = y*8 + x, a8..h1, ' '=empty). Our
        Board.decode is column-major, so we assign squares directly to avoid a
        transpose bug.
        """
        a = self.arimaa
        board = a.Board()
        board.state.setup = False
        board.state.end = False
        board.state.player = color
        board.state.left = 4
        for idx, c in enumerate(short64):
            x, y = idx % 8, idx // 8
            board[(x, y)] = None if c == " " else a.char_to_piece(c)
        return board

    def _part_str(self, board, frm, to):
        char = self.arimaa.piece_to_char(board[frm])
        return char + _alg(frm) + _dir(frm, to)

    def _render_step(self, board, step):
        """Render one of our Steps as one or two AEI step strings in execution order.

        The two atomic sub-steps of a push/pull always involve distinct pieces on
        distinct from-squares, so both chars can be read from the pre-step board.
        """
        if step.opOldPos is None:
            return [self._part_str(board, step.oldPos, step.newPos)]
        if step.newPos == step.opOldPos:
            # push: enemy moves first, then our piece into the vacated square
            return [
                self._part_str(board, step.opOldPos, step.opNewPos),
                self._part_str(board, step.oldPos, step.newPos),
            ]
        # pull: our piece moves first, then the enemy into our old square
        return [
            self._part_str(board, step.oldPos, step.newPos),
            self._part_str(board, step.opOldPos, step.opNewPos),
        ]

    def _legal_steps(self, board):
        return [
            s
            for s in board.possible_steps()
            if board.step_cost(s) <= board.state.left
        ]

    def _score_after(self, board, step, player):
        """Our evaluation of `player` after tentatively applying `step`."""
        saved = board.encode()
        env = self._env
        board.do_step(step)
        score = env._evaluate(player)
        board.decode(saved)
        return score

    def _fresh_env(self, board):
        """Wrap a Board in a minimally-initialised ArimaaEnv (no random setup)."""
        a = self.arimaa
        env = a.ArimaaEnv.__new__(a.ArimaaEnv)
        env.board = board
        env.position_counts = {}
        env.end_reason = None
        env.turn_progress = 0.0
        env.turn_steps_taken = 0
        return env

    def _choose_muzero(self, color, board):
        """Run step-by-step MCTS until END_TURN, assembling one AEI move."""
        a = self.arimaa
        env = self._fresh_env(board)
        aei_steps = []
        # At most 4 step-points per turn (a push/pull consumes 2), so cap the loop.
        for _ in range(4):
            if env.board.state.left == 0 or env.board.state.end:
                break
            if env.to_play() != color:  # turn auto-finished on the previous step
                break
            legal = env.legal_actions()
            if not legal:
                break
            obs = env.get_observation()
            root, _ = self.mcts.run(self.model, obs, legal, env.to_play(), False)
            action = self.select_action(root, 0)
            if action == a.ACTION_END_TURN:
                break
            step = env.action_to_step(action)
            aei_steps.extend(self._render_step(env.board, step))
            env.step(action)
        return " ".join(aei_steps)

    def _choose_jax(self, color, board):
        """Pick actions with the JAX AlphaZero net+search, render via the legacy
        Board (action indices are identical across engines by construction)."""
        a = self.arimaa
        from jaxarimaa import env as jenv
        from jaxarimaa import search as jsearch
        from jaxarimaa import difftest as jdiff
        J = self._jax
        jax = J["jax"]
        aei_steps = []
        turn_start = jdiff.array_from_legacy_board(board)  # board at the turn's start
        for _ in range(4):
            if board.state.left == 0:
                break
            arr = jdiff.array_from_legacy_board(board)
            st = jenv.state_from_board(arr, color, board.state.left,
                                       turn_start=turn_start)
            bst = jax.tree_util.tree_map(lambda x: x[None], st)  # batch of 1
            J["key"], k = jax.random.split(J["key"])
            out = jsearch.run_search(J["model"], J["params"], k, bst,
                                     J["sims"], J["considered"], J["features"])
            action = int(out.action[0])
            if action == a.ACTION_END_TURN:
                break
            spec = a.ACTION_LIST[action]
            step = a.Step()
            step.oldPos, step.newPos = spec.old_pos, spec.new_pos
            step.opOldPos, step.opNewPos = spec.op_old_pos, spec.op_new_pos
            aei_steps.extend(self._render_step(board, step))
            board.do_step(step)
        return " ".join(aei_steps)

    def choose(self, color, short64):
        """Return an AEI move string for `color` to move from the given position."""
        a = self.arimaa
        board = self._board_from_short(color, short64)
        if self.policy == "muzero":
            return self._choose_muzero(color, board)
        if self.policy == "jax":
            return self._choose_jax(color, board)
        # scratch env just to reuse _evaluate; _evaluate reads self.board live
        self._env = self._fresh_env(board)
        chosen = []
        aei_steps = []
        while board.state.left > 0:
            steps = self._legal_steps(board)
            if not steps:
                break
            if self.policy == "random":
                pick = random.choice(steps)
            elif self.policy == "greedy":
                pick = max(steps, key=lambda s: self._score_after(board, s, color))
            else:
                raise NotImplementedError(f"policy {self.policy!r} not wired yet")
            aei_steps.extend(self._render_step(board, pick))
            board.do_step(pick)
            chosen.append(pick)
            # Greedy: stop early once no single step improves our score and we've moved.
            if self.policy == "greedy" and chosen:
                cur = self._env._evaluate(color)
                remaining = self._legal_steps(board)
                if remaining:
                    best_next = max(
                        self._score_after(board, s, color) for s in remaining
                    )
                    if best_next <= cur:
                        break
            elif self.policy == "random" and chosen and random.random() < 0.25:
                break
        return " ".join(aei_steps)


# ---------------------------------------------------------------------------
# AEI engine
# ---------------------------------------------------------------------------
class OurEngine:
    def __init__(self, controller, policy, checkpoint=None, config_json=None,
                 simulations=None):
        from pyrimaa.board import BLANK_BOARD, Color, Position

        self._Position = Position
        self._Color = Color
        self._BLANK = BLANK_BOARD
        self.controller = controller
        self.policy = policy
        self.checkpoint = checkpoint
        self.config_json = config_json
        self.simulations = simulations
        self.chooser = None  # built lazily after handshake

        try:
            header = controller.messages.get(timeout=30)
        except Empty:
            raise AEIException("Timed out waiting for aei header") from None
        if header != "aei":
            raise AEIException(f"Did not receive aei header, instead ({header})")
        controller.send("protocol-version 1")
        controller.send(f"id name MuZeroArimaa-{policy}")
        controller.send("id author muzero-general-arimaa")
        controller.send("aeiok")
        self.newgame()

    def newgame(self):
        self.position = self._Position(self._Color.GOLD, 4, self._BLANK)
        self.insetup = True

    def setposition(self, side_str, pos_str):
        from pyrimaa.board import parse_short_pos

        side = "gswb".find(side_str) % 2
        self.position = parse_short_pos(side, 4, pos_str)
        self.insetup = False

    def makemove(self, move_str):
        from pyrimaa.board import Color, IllegalMove

        try:
            self.position = self.position.do_move_str(move_str)
        except IllegalMove:
            self.log(f"Error: received illegal move {move_str}")
            return False
        if self.insetup and self.position.color == Color.GOLD:
            self.insetup = False
        return True

    def go(self):
        pos = self.position
        if self.insetup:
            # Delegate setup placement to pyrimaa: always a legal arrangement.
            from pyrimaa.board import BASIC_SETUP, Color, Position

            setup = Position(Color.GOLD, 4, BASIC_SETUP)
            move_str = setup.to_placing_move()[pos.color][2:]
            self.bestmove(move_str)
            return
        if self.chooser is None:
            self.chooser = MoveChooser(
                self.policy,
                checkpoint=self.checkpoint,
                config_json=self.config_json,
                simulations=self.simulations,
            )
        short = pos._to_short_str()[1:-1]  # strip surrounding [ ]
        move_str = self.chooser.choose(pos.color, short)
        if not move_str:
            self.log("Warning: no legal move found; sending empty move.")
        self.bestmove(move_str)

    def info(self, msg):
        self.controller.send("info " + msg)

    def log(self, msg):
        self.controller.send("log " + msg)

    def bestmove(self, move_str):
        self.controller.send("bestmove " + move_str)

    def main(self):
        ctl = self.controller
        while not ctl.stop.is_set():
            msg = ctl.messages.get()
            if msg == "isready":
                ctl.send("readyok")
            elif msg == "newgame":
                self.newgame()
            elif msg.startswith("setposition"):
                side, pos_str = msg.split(None, 2)[1:]
                self.setposition(side, pos_str)
            elif msg.startswith("setoption"):
                pass  # we don't yet act on any options
            elif msg.startswith("makemove"):
                move_str = msg.split(None, 1)[1]
                if not self.makemove(move_str):
                    return
            elif msg.startswith("go"):
                if len(msg.split()) == 1:
                    self.go()
            elif msg == "stop":
                pass
            elif msg == "quit":
                return


def main():
    parser = argparse.ArgumentParser(description="AEI engine for our Arimaa bot")
    parser.add_argument(
        "--policy",
        choices=["random", "greedy", "muzero", "jax"],
        default="greedy",
        help="Move-selection policy ('jax' = JAX AlphaZero checkpoint).",
    )
    parser.add_argument(
        "--checkpoint",
        help="Path to a MuZero model.checkpoint (required for --policy muzero).",
    )
    parser.add_argument(
        "--config-json",
        dest="config_json",
        help="Optional JSON of MuZeroConfig overrides (must match the trained net).",
    )
    parser.add_argument(
        "--simulations",
        type=int,
        help="Override num_simulations for MCTS at play time.",
    )
    args = parser.parse_args()

    ctl = _ComThread()
    ctl.start()
    try:
        eng = OurEngine(
            ctl,
            args.policy,
            checkpoint=args.checkpoint,
            config_json=args.config_json,
            simulations=args.simulations,
        )
        eng.main()
    finally:
        ctl.stop.set()


if __name__ == "__main__":
    main()
