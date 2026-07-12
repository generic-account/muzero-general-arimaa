#!/usr/bin/env python3
"""AEI adapter for legacy file-based Arimaa sample bots (bot_Occam, bot_Faerie).

Neither bot_Occam ("getMove") nor bot_Faerie ("Fairy") speaks the AEI protocol
natively.  Each is a one-shot program that reads the current game state from a
file given as a command line argument and prints a single move to stdout:

  * Occam  : reads an Arimaa *move-list* / gamelog file
             (lines like "1w Ee2 ...", "2w Ee2n ...", final line is the bare
             turn tag e.g. "2w" which triggers move generation).
             Invoked as:  getMove <log> <log> <log>
             (it uses the 2nd positional arg as the move file; we pass the
             same path three times so both the move file and the tcmove file
             point at it).
  * Faerie : reads a *board diagram* position file (a "2w" style header plus
             the 8x8 ascii board) and prints a move.
             Invoked as:  Fairy <positionfile>

This adapter implements a minimal AEI engine on stdin/stdout, keeps track of the
game state (move history from newgame+makemove, or a board set via
setposition), and on each "go" builds the appropriate native input file, runs
the wrapped binary, and reports the result with "bestmove".

Usage:
    aei_adapter.py --bot {occam|faerie} --exe /abs/path/to/binary [--depth N]

Only depends on the Python standard library.
"""

import argparse
import os
import subprocess
import sys
import tempfile

COLS = "abcdefgh"

# ---------------------------------------------------------------------------
# Board representation.
#
# We keep an 8x8 board as a dict {(col,row): piece_char} where col in 0..7
# (a..h) and row in 1..8, piece_char is one of "RCDHMErcdhme" (upper = gold,
# lower = silver).  Empty squares are absent.
# ---------------------------------------------------------------------------

TRAP_SQUARES = {(2, 3), (5, 3), (2, 6), (5, 6)}  # (col,row) c3 f3 c6 f6


def empty_board():
    return {}


def board_from_setposition_string(s):
    """Parse the AEI 64-char board string.

    Squares are given left-to-right, top-to-bottom: a8..h8, a7..h7, ... a1..h1.
    A leading '[' and trailing ']' bracket the 64 cells; each cell is a piece
    letter or a space.
    """
    inner = s
    if "[" in inner and "]" in inner:
        inner = inner[inner.index("[") + 1: inner.rindex("]")]
    # Pad/truncate to 64.
    inner = (inner + " " * 64)[:64]
    board = {}
    idx = 0
    for row in range(8, 0, -1):          # 8 down to 1
        for col in range(8):             # a..h
            ch = inner[idx]
            idx += 1
            if ch not in " .xX":
                board[(col, row)] = ch
    return board


def alg_to_cr(sq):
    """'e2' -> (col=4, row=2)."""
    return (COLS.index(sq[0]), int(sq[1]))


def cr_to_alg(col, row):
    return "%s%d" % (COLS[col], row)


# ---------------------------------------------------------------------------
# Applying Arimaa steps to a board (needed to keep our own board current so we
# can render a diagram for Faerie regardless of how state was communicated).
#
# A step token looks like: <piece><col><row><dir> e.g. "Ee2n"; a capture
# token ends in 'x' e.g. "Rc3x" (piece removed on a trap).  A setup token has
# no direction: "Ee2".
# ---------------------------------------------------------------------------

DIRS = {"n": (0, 1), "s": (0, -1), "e": (1, 0), "w": (-1, 0)}


def apply_move_tokens(board, tokens):
    for tok in tokens:
        tok = tok.strip()
        if not tok:
            continue
        piece = tok[0]
        if tok[-1] == "x":                    # capture: remove piece
            col, row = alg_to_cr(tok[1:3])
            board.pop((col, row), None)
            continue
        if tok[-1] in DIRS:                    # a step
            col, row = alg_to_cr(tok[1:3])
            dc, dr = DIRS[tok[-1]]
            board.pop((col, row), None)
            board[(col + dc, row + dr)] = piece
        else:                                   # setup placement
            col, row = alg_to_cr(tok[1:3])
            board[(col, row)] = piece
    return board


# ---------------------------------------------------------------------------
# Rendering native input files.
# ---------------------------------------------------------------------------

def render_faerie_position(board, side, movenum):
    """Render the ascii-diagram position file Fairy expects.

    Header line is like "2w" (move number + side char).  Then the bordered
    board with rows 8..1.  Columns are printed as "| p p p ... |".
    """
    lines = []
    lines.append("%d%s" % (movenum, side))
    lines.append(" +-----------------+")
    for row in range(8, 0, -1):
        cells = []
        for col in range(8):
            ch = board.get((col, row))
            if ch is None:
                ch = "X" if (col, row) in TRAP_SQUARES else " "
            cells.append(ch)
        lines.append("%d| %s |" % (row, " ".join(cells)))
    lines.append(" +-----------------+")
    lines.append("   a b c d e f g h")
    return "\n".join(lines) + "\n"


def render_occam_movelist(history, side, movenum, tcmove):
    """Render Occam's move-list file.

    `history` is a list of (tag, movestring) already-played moves, e.g.
    [("1w", "Ra1 Rb1 ..."), ("1b", "ra8 ..."), ("2w", "Ee2n ...")].
    We then append the bare current turn tag which triggers Occam to move.
    """
    # NOTE: no tcmove= header here — Occam's do_bot() parses this file as
    # moves ONLY (a tcmove line silently corrupts the parse -> empty board ->
    # no move). tcmove goes in a SEPARATE gamestate file (read_gamestate()).
    lines = []
    for tag, mv in history:
        if mv:
            lines.append("%s %s" % (tag, mv))
        else:
            lines.append(tag)
    lines.append("%d%s" % (movenum, side))   # bare current tag -> produce move
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# The adapter engine.
# ---------------------------------------------------------------------------

class Adapter:
    def __init__(self, bot, exe, depth):
        self.bot = bot
        self.exe = exe
        self.depth = depth
        self.tcmove = 0
        self.reset_game()

    def reset_game(self):
        self.board = empty_board()
        self.side = "g"           # AEI uses g/s
        self.movenum = 1
        # Occam movelist history of (tag, movestring)
        self.history = []
        self.have_board = False   # whether self.board is populated

    def log(self, msg):
        sys.stdout.write("log %s\n" % msg)
        sys.stdout.flush()

    def send(self, msg):
        sys.stdout.write(msg + "\n")
        sys.stdout.flush()

    # -- side helpers -----------------------------------------------------
    @staticmethod
    def side_char_native(aei_side):
        return "w" if aei_side in ("g", "w") else "b"

    def advance_turn(self):
        if self.side in ("g", "w"):
            self.side = "s"
        else:
            self.side = "s"  # placeholder, corrected below
        # Proper toggle + movenumber increment (increment after silver moves).

    # -- command handlers -------------------------------------------------
    def handle_setposition(self, args):
        parts = args.split(None, 1)
        self.side = parts[0].strip()
        boardstr = parts[1] if len(parts) > 1 else ""
        self.board = board_from_setposition_string(boardstr)
        self.have_board = True
        # We cannot know the true move number; assume mid-game generic value.
        # Faerie only uses the header for side + parity; movenum>=2 => search.
        self.movenum = 2
        # For Occam, express the board as setup-only history (best effort).
        self.history = self._board_to_occam_setup_history()

    def _board_to_occam_setup_history(self):
        gold = []
        silver = []
        for (col, row), ch in self.board.items():
            tok = ch + cr_to_alg(col, row)
            if ch.isupper():
                gold.append(tok)
            else:
                silver.append(tok)
        hist = []
        hist.append(("1w", " ".join(sorted(gold))))
        hist.append(("1b", " ".join(sorted(silver))))
        return hist

    def handle_makemove(self, move):
        move = move.strip()
        # Match runners echo EVERY move to BOTH bots, including the bot that
        # just played it. handle_go already recorded our own move (and
        # advanced side/movenum) — recording the echo again desyncs the
        # movelist. Opponent moves can never be string-identical to ours
        # (piece letters are opposite case), so this comparison is safe.
        if self.history and self.history[-1][1] == move:
            return
        tag = "%d%s" % (self.movenum, self.side_char_native(self.side))
        self.history.append((tag, move))
        if self.have_board or not self.board == {}:
            apply_move_tokens(self.board, move.split())
        # toggle side / movenumber
        if self.side in ("g", "w"):
            self.side = "s"
        else:
            self.side = "g"
            self.movenum += 1

    def handle_newgame(self):
        self.reset_game()

    def run_bot(self):
        native_side = self.side_char_native(self.side)
        spath = None
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
            path = f.name
            if self.bot == "faerie":
                f.write(render_faerie_position(self.board, native_side,
                                               self.movenum))
            else:
                f.write(render_occam_movelist(self.history, native_side,
                                              self.movenum, self.tcmove))
        try:
            if self.bot == "faerie":
                cmd = [self.exe, path]
            else:
                # Occam getMove argv: <posfile> <movefile> <gamestatefile>.
                # do_bot() parses argv[-2] as moves ONLY; read_gamestate()
                # scans argv[-1] for tcmove=. They must be SEPARATE files.
                with tempfile.NamedTemporaryFile("w", suffix=".st",
                                                 delete=False) as sf:
                    spath = sf.name
                    sf.write("tcmove=%d\n" % int(self.tcmove or 60))
                cmd = [self.exe]
                if self.depth:
                    cmd += ["-d", str(self.depth)]
                cmd += [path, path, spath]
            proc = subprocess.run(cmd, stdout=subprocess.PIPE,
                                  stderr=subprocess.PIPE, timeout=300)
            out = proc.stdout.decode("utf-8", "replace")
        finally:
            for pth in (path, spath):
                if pth:
                    try:
                        os.unlink(pth)
                    except OSError:
                        pass
        # The move is the last non-empty stdout line for both bots.
        move = ""
        for line in out.splitlines():
            line = line.strip()
            if line:
                move = line
        return move

    GOLD_SETUP = ("Ra1 Rb1 Rc1 Rd1 Re1 Rf1 Rg1 Rh1 "
                  "Ha2 Db2 Cc2 Md2 Ee2 Cf2 Dg2 Hh2")
    SILVER_SETUP = ("ra8 rb8 rc8 rd8 re8 rf8 rg8 rh8 "
                    "ha7 db7 cc7 md7 ee7 cf7 dg7 hh7")

    def handle_go(self, args):
        if args.strip().startswith("ponder"):
            # We can't usefully ponder; just log and wait.
            self.log("ponder not supported, ignoring")
            return
        if self.movenum == 1:
            # Setup phase: the wrapped binaries never produce placement moves
            # (the original bot scripts hardcode them) — getMove returns
            # nothing and the empty bestmove crashes the match runner.
            move = (self.GOLD_SETUP if self.side in ("g", "w")
                    else self.SILVER_SETUP)
        else:
            move = self.run_bot()
        if not move:
            self.log("Error: wrapped bot produced no move")
            move = ""
        # record our own move so subsequent state stays consistent
        if move:
            tag = "%d%s" % (self.movenum, self.side_char_native(self.side))
            self.history.append((tag, move))
            if self.board:
                apply_move_tokens(self.board, move.split())
            if self.side in ("g", "w"):
                self.side = "s"
            else:
                self.side = "g"
                self.movenum += 1
        self.send("bestmove %s" % move)

    def handle_setoption(self, args):
        toks = args.split()
        name = None
        value = None
        if "name" in toks:
            i = toks.index("name")
            if i + 1 < len(toks):
                name = toks[i + 1]
        if "value" in toks:
            i = toks.index("value")
            if i + 1 < len(toks):
                value = toks[i + 1]
        if name == "tcmove" and value is not None:
            try:
                self.tcmove = int(float(value))
            except ValueError:
                pass
        elif name == "depth" and value is not None:
            try:
                self.depth = int(value)
            except ValueError:
                pass

    # -- main loop --------------------------------------------------------
    def loop(self):
        for raw in sys.stdin:
            line = raw.rstrip("\r\n")
            if line == "aei":
                self.send("protocol-version 1")
                self.send("id name bot_%s" % self.bot.capitalize())
                self.send("id author Arimaa sample bots (AEI adapter)")
                self.send("id version 1.0")
                self.send("aeiok")
            elif line == "isready":
                self.send("readyok")
            elif line == "newgame":
                self.handle_newgame()
            elif line.startswith("setposition"):
                self.handle_setposition(line[len("setposition"):].strip())
            elif line.startswith("setoption"):
                self.handle_setoption(line[len("setoption"):].strip())
            elif line.startswith("makemove"):
                self.handle_makemove(line[len("makemove"):].strip())
            elif line.startswith("go"):
                self.handle_go(line[len("go"):].strip())
            elif line == "stop":
                pass  # our search is synchronous; nothing to stop
            elif line == "quit":
                break
            elif line == "":
                continue
            else:
                self.log("Warning: unrecognized command: %s" % line)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bot", required=True, choices=["occam", "faerie"])
    ap.add_argument("--exe", required=True, help="absolute path to native binary")
    ap.add_argument("--depth", type=int, default=0,
                    help="Occam only: search depth in steps (-d). 0 = default")
    args = ap.parse_args()
    Adapter(args.bot, args.exe, args.depth).loop()


if __name__ == "__main__":
    main()
