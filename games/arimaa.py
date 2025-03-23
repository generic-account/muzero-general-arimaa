import datetime
import pathlib
import numpy
import torch
import random
from .abstract_game import AbstractGame
from typing import Any, Generator, NewType, TypeVar

class MuZeroConfig:
    def __init__(self):
        self.seed = 0
        self.max_num_gpus = None
        
        # Game
        self.observation_shape = (12, 8, 8)  # 6 piece types * 2 colors, 8x8 board
        self.action_space = list(range(2048))  # Approximate upper bound for Arimaa moves
        self.players = list(range(2))  # Two players: Gold and Silver
        self.stacked_observations = 0
        
        # Evaluate
        self.muzero_player = 0
        self.opponent = "expert"
        
        # Self-Play
        self.num_workers = 1
        self.selfplay_on_gpu = False
        self.max_moves = 200
        self.num_simulations = 800
        self.discount = 0.997
        self.temperature_threshold = None
        
        # Root prior exploration noise
        self.root_dirichlet_alpha = 0.3
        self.root_exploration_fraction = 0.25
        
        # UCB formula
        self.pb_c_base = 19652
        self.pb_c_init = 1.25
        
        # Network
        self.network = "resnet"
        self.support_size = 300
        
        # Residual Network
        self.downsample = False
        self.blocks = 20
        self.channels = 256
        self.reduced_channels_reward = 256
        self.reduced_channels_value = 256
        self.reduced_channels_policy = 256
        self.resnet_fc_reward_layers = [256, 256]
        self.resnet_fc_value_layers = [256, 256]
        self.resnet_fc_policy_layers = [256, 256]
        
        # Training
        self.results_path = pathlib.Path(__file__).resolve().parents[1] / "results" / pathlib.Path(__file__).stem / datetime.datetime.now().strftime("%Y-%m-%d--%H-%M-%S")
        self.save_model = True
        self.training_steps = 1000000
        self.batch_size = 1024
        self.checkpoint_interval = 10
        self.value_loss_weight = 0.25
        self.train_on_gpu = torch.cuda.is_available()
        self.optimizer = "SGD"
        self.weight_decay = 1e-4
        self.momentum = 0.9
        
        # Exponential learning rate schedule
        self.lr_init = 0.05
        self.lr_decay_rate = 0.1
        self.lr_decay_steps = 400000
        
        # Replay Buffer
        self.replay_buffer_size = 1000000
        self.num_unroll_steps = 5
        self.td_steps = 200
        self.PER = True
        self.PER_alpha = 1.0
        
        # Reanalyze
        self.use_last_model_value = True
        self.reanalyse_on_gpu = False

    def visit_softmax_temperature_fn(self, trained_steps):
        if trained_steps < 0.5 * self.training_steps:
            return 1.0
        elif trained_steps < 0.75 * self.training_steps:
            return 0.5
        else:
            return 0.25

class Game(AbstractGame):
    def __init__(self, seed=None):
        self.env = ArimaaEnv(seed)

    def step(self, action):
        observation, reward, done = self.env.step(action)
        return observation, reward, done

    def to_play(self):
        return self.env.to_play()

    def legal_actions(self):
        return self.env.legal_actions()

    def reset(self):
        return self.env.reset()

    def render(self):
        self.env.render()

    def human_to_action(self):
        while True:
            try:
                move_str = input("Enter your move (e.g. 'Ra1n Db2n Cc3n Dd4n'): ")
                return self.env.move_to_action(move_str)
            except ValueError:
                print("Invalid move. Try again.")

    def action_to_string(self, action_number):
        return self.env.action_to_string(action_number)

    def expert_agent(self):
        return self.env.expert_action()

class ArimaaEnv:
    def __init__(self, seed=None):
        self.board = Board()
        if seed is not None:
            numpy.random.seed(seed)

    def reset(self):
        self.board = Board()
        return self.get_observation()

    def step(self, action):
        move = self.action_to_move(action)
        self.board.do_move(move)
        observation = self.get_observation()
        reward = self.get_reward()
        done = self.board.state.end
        return observation, reward, done

    def to_play(self):
        return self.board.state.player

    def legal_actions(self):
        return list(range(len(list(self.board.possible_moves()))))

    def get_observation(self):
        observation = numpy.zeros((12, 8, 8), dtype=numpy.float32)
        for y in range(8):
            for x in range(8):
                piece = self.board[(x, y)]
                if piece is not None:
                    color, rank = parse_piece(piece)
                    channel = rank if color == COLORS.GOLD else rank + 6
                    observation[channel, y, x] = 1.0
        return observation

    def get_reward(self):
        if self.board.state.end:
            return 1 if self.board.state.player == self.to_play() else -1
        return 0

    def render(self):
        self.board.print()

    def move_to_action(self, move_str):
        move = self.board.parse_move(move_str)
        return list(self.board.possible_moves()).index(move)

    def action_to_move(self, action):
        return list(self.board.possible_moves())[action]

    def action_to_string(self, action):
        move = self.action_to_move(action)
        return self.board.move_str(move)

    def expert_action(self):
        return numpy.random.choice(self.legal_actions())

# The rest of the Arimaa implementation (Board, State, Step, etc.) goes here
# Copy the relevant parts from the provided code

# Arimaa-specific constants and helper functions

Piece = NewType('Piece', int)

class RANKS:
  """
  Stores the constants for the ranks
  """
  RABBIT = 0
  CAT = 1
  DOG = 2
  HORSE = 3
  CAMEL = 4
  ELEPHANT = 5

# Names of the ranks
RankNames = [
  "Rabbit",
  "Cat",
  "Dog",
  "Horse",
  "Camel",
  "Elephant"
]

# Characters that represent the ranks in display and serialization
RankChars = ["R", "C", "D", "H", "M", "E"]

class COLORS:
  """
  Constants for the colors/teams
  """
  GOLD = 0
  SILVER = 1

# Names for the colors
ColorNames = [
  "Gold",
  "Silver"
]

def parse_piece(piece: Piece):
  """
  Take a pice, return the color and rank
  """
  return piece >> 3, piece & ~8

def make_piece(color: int, rank: int) -> Piece:
  """
  Given a color and rank, make a piece
  """
  return (color << 3) + rank # type: ignore

def piece_to_char(piece: Piece):
  """
  Get the character that represents a piece
  """
  if piece == None:
    return "."
  color, rank = parse_piece(piece)
  char = RankChars[rank]
  if color == COLORS.SILVER:
    char = char.lower()
  elif color == COLORS.GOLD:
    char = char.upper()
  return char

def char_to_piece(char: str) -> Piece:
  """
  Get the piece a character represents
  """
  color = COLORS.GOLD
  if char.islower():
    color = COLORS.SILVER
    char = char.upper()
  rank = RankChars.index(char)
  return make_piece(color, rank)

Self = TypeVar("Self", bound="State")

class State:
  """
  Represents the state of the game
  """
  setup: bool # If the players are placing their initial setups
  end: bool # If the game is over
  # If setup is True, this is which player is up to place their initial setup
  # If end is True, this is which player one
  # Otherwise, this is whos turn it is
  player: int
  left: int # How many steps are left in the current player's turn
  
  def describe(self):
    """
    Get a written description of the board state
    """
    if self.setup:
      return ColorNames[self.player] + " to place pieces."
    if self.end:
      return ColorNames[self.player] + " wins!"
    return ColorNames[self.player] + " has " + str(self.left) + " steps left."

  def encode(self):
    """
    Create a string that represents this state
    """
    return ",".join(["S" if self.setup else "E" if self.end else ".",
                     str(self.player), str(self.left)])

  def decode(self, val: str):
    """
    Read a string from State.encode and set this state to match it
    """
    s, p, l = val.split(",")
    self.setup = False
    self.end = False
    if s == "S":
      self.setup = True
    if s == "E":
      self.end = True
    
    self.player = int(p)
    self.left = int(l)

# A position on the board
Pos = NewType('Pos', tuple[int, int])

def in_bounds(pos: Pos) -> bool:
  """
  Check if a position is in the bounds of the board
  """
  x, y = pos
  return x >= 0 and x < 8 and y >= 0 and y < 8

def neighbors(pos: Pos, exclude:int|None=None):
  """
  Get the neighbors of a position, checking if they are in bounds
  @param exclude sets a y direction to not use, either 1 or -1
  """
  x, y = pos
  res: list[Pos] = [
    (x + 1, y),
    (x - 1, y),
    (x, y + 1),
    (x, y - 1)
  ] # type: ignore
  if exclude != None:
    res.remove((x, y + exclude)) # type: ignore
  for pos2 in res:
    if in_bounds(pos2):
      yield pos2

def all_positions() -> Generator[Pos, Any, Any]:
  """
  Get all possible positions on the board
  """
  for i in range(8):
    for j in range(8):
      yield (i, j) # type: ignore

Self = TypeVar("Self", bound="Step")

class Step:
  """
  Represents a single step
  """
  oldPos: Pos # The old position of the piece
  newPos: Pos # The new position of the piece
  opOldPos: Pos | None # The old position of the enemy piece
  opNewPos: Pos | None # The new position of the enemy piece

  @staticmethod
  def create(oldPos: Pos, newPos: Pos) -> Self: # type: ignore
    """
    Create a step with no push or pull
    """
    move = Step()
    move.oldPos = oldPos
    move.newPos = newPos
    move.opOldPos = None
    move.opNewPos = None
    return move # type: ignore
  
  @staticmethod
  def create_push(oldPos: Pos, newPos: Pos, opOldPos: Pos, opNewPos: Pos) -> Self: # type: ignore
    """
    Create a step that has a push or pull
    """
    move = Step()
    move.oldPos = oldPos
    move.newPos = newPos
    move.opOldPos = opOldPos
    move.opNewPos = opNewPos
    return move # type: ignore
  
Move = tuple[Step] | tuple[Step, Step] | tuple[Step, Step, Step] | tuple[Step, Step, Step, Step]

class Board:
  """
  Represents a board of Arimaa, including the current game state
  """
  _data: list[list[Piece | None]] # The pieces on the board None means empty
  state: State # The state of the game

  # The locations of the traps
  TRAPS: list[Pos] = [(2, 2), (2, 5), (5, 2), (5, 5)] # type: ignore

  def __init__(self) -> None:
    self._data = []
    self.state = State()
    # At the start, the gold player places their starting pieces
    self.state.setup = True
    self.state.end = False
    self.state.player = COLORS.GOLD
    self.state.left = -1
    # Initialize the empty board
    for _ in range(8):
      inner = []
      for _ in range(8):
        inner.append(None)
      self._data.append(inner)

  def place_initial(self, player: int, pieces: list[list[int]]):
    """
    Place initial pieces. Front line goes first, then back line
    """
    if not self.state.setup:
      raise StateException("Cannot place initial after setup")
    if player != self.state.player:
      raise StateException("It is not " + ColorNames[player] + "'s turn to place initial.")
    if len(pieces) != 2:
      raise IndexError("Pieces must be 2x8")
    for i in range(2):
      if len(pieces[i]) != 8:
        raise IndexError("Pieces must be 2x8")
      for j in range(8):
        if player == COLORS.SILVER:
          # The silver player has their front line at 1 and back line at 0
          self._data[i][j] = make_piece(player, pieces[-(i + 1)][j])
        elif player == COLORS.GOLD:
          # The gold player has their front line at 6 (-2) and back line at 7 (-1)
          self._data[-(i + 1)][j] = make_piece(player, pieces[-(i + 1)][j])
    # If the gold player just placed, it's now the silver player's turn to place
    if self.state.player == COLORS.GOLD:
      self.state.player = COLORS.SILVER
    # If the silver player just placed, it's now the gold player's first turn
    elif self.state.player == COLORS.SILVER:
      self.state.setup = False
      self.state.player = COLORS.GOLD
      self.state.left = 4

  def __getitem__(self, pos: Pos):
    """
    Get a piece at the given position
    """
    x, y = pos
    return self._data[y][x]
  
  def __setitem__(self, pos: Pos, piece: Piece | None):
    """
    Set the piece at the given position
    """
    x, y = pos
    self._data[y][x] = piece

  def __iter__(self):
    """
    Iterate through all the pieces
    """
    return (piece for row in self._data for piece in row)

  def do_step(self, step: Step):
    """
    Execute a single step and modify the state as needed
    """
    if self.state.left == 0:
      raise StateException("Current player has no steps left.")
    toMove = self[step.oldPos]
    if toMove == None:
      raise StateException("No piece at starting location.")
    color, rank = parse_piece(toMove)
    if color != self.state.player:
      raise StateException("Cannot move opponent's pieces.")
    if self.is_frozen(step.oldPos):
      raise StateException("Cannot move a frozen piece.")

    enemy = None
    if step.opOldPos != None:
      enemy = self[step.opOldPos]
      if enemy == None:
        raise StateException("No piece at enemy location.")
      opColor, opRank = parse_piece(enemy)
      if opColor == color:
        raise StateException("Cannot push or pull your own pieces.")
      if opRank >= rank:
        raise StateException("Cannot push or pull higher rank pieces.")

    # You can move on top of a piece if you're pushing it
    if self[step.newPos] != None and step.newPos != step.opOldPos:
      raise StateException("Cannot move on top of another piece.")
    
    # You can move an enemy on top of your piece if you're pulling it
    if step.opNewPos != None and self[step.opNewPos] != None and step.oldPos != step.opNewPos:
      raise StateException("Cannot push a piece on top of another piece.")
    
    self[step.oldPos] = None
    if step.opOldPos != None:
      self[step.opOldPos] = None
    self[step.newPos] = toMove
    if step.opNewPos != None:
      self[step.opNewPos] = enemy

    self._check_traps()

    self.state.left -= 1

  def do_move(self, move: Move):
    if self.state.left < len(move):
      raise StateException("Cannot make a move with more steps than are left.")
    for step in move:
      self.do_step(step)
    self.finish_turn()

  def finish_turn(self):
    """
    Finish the current turn and start the next player's turn
    """
    if self.state.left == 4:
      raise StateException("Cannot fully pass the turn")
    
    self.state.player = 1 - self.state.player
    self.state.left = 4

    win = self._check_win()
    if win != None:
      self.state.end = True
      self.state.player = win

  def pieces(self):
    """
    Iterate through all the pieces on the board
      Unlike __iter__(), this skips the empty spaces
    """
    for piece in self:
      if piece != None:
        yield piece
  
  def _check_traps(self):
    """
    Check if any piece is on a trap and unprotected, if so, remove it
    """
    for trap in self.TRAPS:
      piece = self[trap]
      if piece != None:
        color, _ = parse_piece(piece)
        safe = False
        for pos in neighbors(trap):
          friend = self[pos]
          if friend != None:
            c, _ = parse_piece(friend)
            if color == c:
              safe = True
              break
        if not safe:
          self[trap] = None

  def _check_win(self) -> int | None:
    """
    Check if a player has won the game
    """
    def check_end(player: int):
      """
      Check if a player has brought their rabbit to the opposite side of the board
      """
      for piece in self._data[0 if player == COLORS.GOLD else -1]:
        if piece != None:
          color, rank = parse_piece(piece)
          if color == player and rank == RANKS.RABBIT:
            return player
      return None
    
    def check_rabbits(player: int):
      """
      Check if a player is out of rabbits
      """
      for piece in self.pieces():
        color, rank = parse_piece(piece)
        if color == player and rank == RANKS.RABBIT:
          return None
      return 1 - player
    
    playerA = 1 - self.state.player # Player A is the player who just finished their turn
    playerB = self.state.player # Player B is the player who's turn just started
    win = check_end(playerA)
    if win != None:
      return win
    win = check_end(playerB)
    if win != None:
      return win
    win = check_rabbits(playerB)
    if win != None:
      return win
    win = check_rabbits(playerA)
    if win != None:
      return win
    
    # If the current player has no possible moves, their opponent wins
    if not any(self.possible_steps()):
      return 1 - self.state.player
    
    return None
  
  def is_frozen(self, pos: Pos) -> bool:
    """
    Check if a piece is frozen, which happens when a stronger piece is adjacent to it
    """
    piece = self[pos]
    if piece == None:
      # This shouldn't happen
      return True
    color, rank = parse_piece(piece)
    frozen = False
    helped = False
    for pos2 in neighbors(pos):
      friend = self[pos2]
      if friend == None:
        continue
      color2, rank2 = parse_piece(friend)
      if color != color2 and rank2 > rank:
        frozen = True
      if color == color2:
        helped = True
    return frozen and not helped
  
  def possible_steps(self) -> Generator[Step, Any, None]:
    """
    Obtain all possible steps for the current player
    """
    for pos in all_positions():
      piece = self[pos]
      if piece == None:
        continue
      if self.is_frozen(pos): # Cannot move if frozen
        continue

      color, rank = parse_piece(piece)
      if color != self.state.player:
        continue

      # Rabbits cannot move backwards
      exclude = None
      if rank == RANKS.RABBIT:
        if color == COLORS.GOLD:
          exclude = 1
        if color == COLORS.SILVER:
          exclude = -1

      for pos2 in neighbors(pos, exclude):
        enemy = self[pos2]
        if enemy == None:
          yield Step.create(pos, pos2)
        else:
          color2, rank2 = parse_piece(enemy)
          if color != color2 and rank > rank2:
            for pos3 in neighbors(pos2):
              if self[pos3] == None:
                # Push the enemy onto a tile adjacent to them, then move to their old position
                yield Step.create_push(pos, pos2, pos2, pos3)
            for pos3 in neighbors(pos):
              if self[pos3] == None:
                # Step into an adjacent tile, them pull the enemy to my old tile
                yield Step.create_push(pos, pos3, pos2, pos)

  def possible_moves(self) -> Generator[Move, Any, None]:
    """
    Obtain all possible moves for the current player.
    Caveat: This method mutates the board while iterating.
      When a move is yielded, the board state is set to how
        it would look like if that series of steps was played.
      To avoid this, gather the elements into a list.
      The method will restore the original state of the
        board upon the generator being exhausted.
    """
    def expand(existing: list[Step]) -> Generator[Move, Any, None]:
      if self.state.left == 0:
        return
      savedState = self.encode()
      for step in self.possible_steps():
        self.do_step(step)
        yield tuple(existing + [step]) #type: ignore
        yield from expand(existing + [step])
        self.decode(savedState)

    yield from expand([])

  def random_step(self) -> Step | None:
    """
    Get a random step from the current position.
      Includes a sanity check for an empty return value.
    """
    steps = list(self.possible_steps())
    if len(steps) == 0:
      return None
    return random.choice(steps)

  def random_move(self) -> Move:
    """
    Return a random valid move from the current position.
      This is much more efficient than doing
        `random.choice(list(board.possible_moves()))`
        because it doesn't need to gather a list of all possible moves,
        which can be upwards of 300k moves long.
    """
    if self.state.left == 0:
      raise StateException("Current player is out of steps, no possible moves")
    # Add some extra options on top to bias the randomness towards doing as many steps as possible
    steps = random.randint(1, self.state.left + 3)
    steps = min(steps, self.state.left)
    move = []
    savedState = self.encode()
    for _ in range(steps):
      step = self.random_step()
      if step == None:
        break
      move.append(step)
      self.do_step(step)
    self.decode(savedState)
    return tuple(move)

  def print(self):
    """
    Print the board, including current state
    """
    print(self.state.describe())
    print(" +-----------------+")
    for col in range(8):
      print(str(8 - col) + "| ", end="")
      for row in range(8):
        if self[row,col] != None: # type: ignore
          print(piece_to_char(self[row,col]), end=" ") # type: ignore
        elif (row == 2 or row == 5) and (col == 2 or col == 5):
          print("x ", end="")
        else:
          print(". ", end="")
      print("|")
    print(" +-----------------+")
    print("   a b c d e f g h  ")
    print()

  def encode(self):
    """
    Encode the board into a string representation
    """
    def stringify():
      for pos in all_positions():
        piece = self[pos]
        if piece != None:
          yield piece_to_char(piece)
        else:
          yield "."

    return self.state.encode() + " " +  "".join(stringify())

  def decode(self, val: str):
    """
    Update the board with the data from a string from Board.encode
    """
    s, b = val.split(" ")
    self.state.decode(s)

    i = 0
    for pos in all_positions():
      c = b[i]
      if c == ".":
        self[pos] = None
      else:
        self[pos] = char_to_piece(c)
      i += 1

  def parse_step(self, val: str, push: str | None):
    """
    Parse a step string with optional push into a Move object
    """
    def parse_part(val: str) -> tuple[Pos, Pos]:
      if len(val) != 4:
        raise ValueError("This is probably an initial placement, not a step")
      if val[3] == "x":
        raise ValueError("This is an elimination, not a move")
      pos: Pos = (ord(val[1]) - 97, 8 - int(val[2])) # type: ignore
      piece = self[pos]
      if piece == None:
        raise StateException("Step is invalid, no piece at starting location.")
      char = piece_to_char(piece)
      if char != val[0]:
        raise StateException("Step is invalid, wrong piece at starting location")
      dir = val[3]
      x, y = pos
      if dir == "n":
        y -= 1
      elif dir == "s":
        y += 1
      elif dir == "e":
        x += 1
      elif dir == "w":
        x -= 1
      else:
        raise ValueError("Invalid direction: " + dir)
      return pos, (x, y) # type: ignore

    oldPos, newPos = parse_part(val)
    opOldPos = None
    opNewPos = None
    if push != None:
      opOldPos, opNewPos = parse_part(push)
    step = Step()
    step.oldPos = oldPos
    step.newPos = newPos
    step.opOldPos = opOldPos
    step.opNewPos = opNewPos
    return step
  
  def step_str(self, step: Step) -> tuple[str, str | None]:
    """
    Turn a step into a string, and maybe a push string as well
    """
    def part_str(oldPos: Pos, newPos: Pos):
      char = piece_to_char(self[oldPos]) # type: ignore
      x, y = oldPos
      pos = chr(x + 97) + str(8 - y)
      dir = None
      xn, yn = newPos
      dx = x - xn
      dy = y - yn
      if dx == 1:
        dir = "e"
      elif dx == -1:
        dir = "w"
      elif dy == 1:
        dir = "s"
      elif dy == -1:
        dir = "n"
      else:
        raise ValueError("Invalid move direction")
      return char + pos + dir
    push = None
    if step.opOldPos != None and step.opNewPos != None:
      push = part_str(step.opOldPos, step.opNewPos)
    return part_str(step.oldPos, step.newPos), push
  
  def move_str(self, move: Move) -> str:
    def tuple_str(arg: tuple[str, str | None]):
      if arg[1] == None:
        return arg[0]
      else:
        return arg[0] + "," + arg[1]
    return " ".join([tuple_str(self.step_str(step)) for step in move])
  
  def parse_move(self, val: str) -> Move:
    strs = val.split(" ")
    steps = []
    for stepStr in strs:
      args = stepStr.split(",")
      steps.append(self.parse_step(args[0], args[1]))
    
    return tuple(steps)

def parse_initial(strs: list[str]):
  """
  Parse an initial placement, which is basically a move list without the directions
  """
  initial: list[list[int]] = [[], []]
  for i in range(8):
    initial[0].append(None) # type: ignore
    initial[1].append(None) # type: ignore

  color = None
  for p in strs:
    if len(p) != 3:
      continue
    piece = char_to_piece(p[0])
    color, rank = parse_piece(piece)
    x = ord(p[1]) - 97
    y = int(p[2])
    if color == COLORS.GOLD:
      y -= 1
    elif color == COLORS.SILVER:
      y = 8 - y
    initial[y][x] = rank

  return color, initial

def initial_str(player: int, initial: list[list[int]]):
  """
  Create a string representation of an intial position
  """
  steps: list[str] = []
  for y in range(2):
    for x in range(8):
      rank = initial[y][x]
      color = player
      pos = y
      if color == COLORS.GOLD:
        pos += 1
      elif color == COLORS.SILVER:
        pos = 8 - pos
      pos = chr(x + 97) + str(pos)
      steps.append(piece_to_char(make_piece(color, rank)) + pos)
  return " ".join(steps)

class StateException(Exception):
    """Raised when a given action is invalid given the state of the board"""
    pass
