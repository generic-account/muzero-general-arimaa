"""Configuration dataclasses for the jaxarimaa AlphaZero stack.

These are plain (frozen) dataclasses used to *construct* modules and drive the
training loop. Keep them out of traced/jitted code paths (pass concrete scalars
into jitted functions); they carry the knobs, not runtime state.
"""

from dataclasses import dataclass, field, asdict, replace


@dataclass(frozen=True)
class FeaturesConfig:
    """Independent on/off toggles for model-quality/efficiency features, for
    ablation. Baseline (all-off) reproduces the original behavior exactly. Frozen &
    hashable so it can be a static jit argument. New features append fields here as
    they land (moves_left_head, arena_gating, playout_cap, planes_moved, ...)."""
    # --- extra input planes ---
    planes_frozen: bool = False        # 1 plane: squares holding a frozen piece
    planes_trap: bool = False          # 1 plane: the four trap squares (static)
    planes_step_in_turn: bool = False  # 1 plane: (4 - steps_left)/4
    planes_moved: bool = False         # 1 plane: squares changed this turn (board != turn_start)
    # --- training / compute ---
    symmetry_aug: bool = False         # left-right (x->7-x) data augmentation
    bf16: bool = False                 # bfloat16 compute (params kept fp32)
    arena_gating: bool = False         # run learner-vs-frozen-anchor matches -> chained Elo metric (train.py); NOT a data gate
    resign: bool = False               # adjudicate decided self-play games early (more games/rollout)
    playout_cap: bool = False          # KataGo playout-cap: cheap "fast" moves, train only on "full" moves
    adjudicate_truncation: bool = False  # truncated games: material/advancement adjudication (env.material_eval) instead of net bootstrap
    fast_search: bool = False          # batched sequential halving (wave-parallel Gumbel; see fast_search.py)
    # --- architecture (auxiliary heads) ---
    moves_left_head: bool = False      # aux head predicting (normalized) plies to game end
    deep_supervision: bool = False     # intermediate policy/value heads (deep supervision)
    mtp: bool = False                  # multi-token-prediction-style: predict next-step value
    # --- architecture (transformer backbone upgrades, LeelaZero-inspired) ---
    smolgen: bool = False              # dynamic position-dependent attention bias (transformer only)
    rope: bool = False                 # 2D rotary positional encoding (transformer only)


@dataclass(frozen=True)
class NetConfig:
    backbone: str = "resnet"        # key into backbones.BACKBONE_REGISTRY
    # channels is the conv N-dim; keep it a MULTIPLE OF 128 (the MXU width) so the
    # systolic array fills — 128 is the economical high-MFU sweet spot, 64 wastes
    # half the array (roofline analysis, scope doc §14).
    channels: int = 128             # conv width / transformer model dim
    blocks: int = 10                # resnet blocks / transformer layers
    use_se: bool = True             # squeeze-excitation (resnet only)
    # transformer-specific (ignored by resnet)
    num_heads: int = 4
    mlp_ratio: int = 4


@dataclass(frozen=True)
class MCTSConfig:
    num_simulations: int = 64
    max_num_considered_actions: int = 32   # Gumbel: root actions sampled w/o replacement
    # search uses the real simulator; discount handles two-player perspective flips.


@dataclass(frozen=True)
class SelfPlayConfig:
    batch_size: int = 128           # concurrent games per device
    max_steps: int = 300            # step-actions per game before truncation
    # resign / adjudication (features.resign)
    resign_threshold: float = 0.9   # |root value| above which a game is adjudicated
    # playout-cap randomization (features.playout_cap)
    full_search_prob: float = 0.25  # fraction of moves that get the full-sim search (+trained)
    fast_sims: int = 8              # simulations for cheap "fast" moves (not trained on)
    # optima-style decisiveness: play greedily (argmax of search weights) once a
    # game passes this many completed TURNS (0 = off; optima uses temp->0 @ 15)
    greedy_after_turns: int = 0


@dataclass(frozen=True)
class TrainConfig:
    optimizer: str = "adamw"        # adamw | adam | sgd (nesterov) | lion
    lr: float = 2e-3                # peak LR, specified for `lr_ref_batch`
    lr_ref_batch: int | None = None  # if set, LR scales by train_batch_size/ref
    weight_decay: float = 1e-4
    moves_left_weight: float = 0.15  # loss weight for the moves-left aux head (if enabled)
    deep_supervision_weight: float = 0.3  # loss weight applied to intermediate deep-sup heads
    mtp_weight: float = 0.15         # loss weight for the MTP next-value head
    warmup_steps: int = 200
    grad_clip: float = 1.0
    value_loss_weight: float = 1.0
    train_batch_size: int = 1024    # GLOBAL batch (sharded across devices)
    iterations: int = 100           # self-play/train iterations
    train_steps_per_iter: int = 16  # replay-ratio knob (grad steps per self-play round)
    replay_capacity: int = 100_000
    min_replay_size: int = 1        # per-device rows required before training
    multihost: bool = False         # call jax.distributed.initialize() at startup
    ckpt_interval: int = 0          # iters between Orbax saves (0 = disabled)
    ckpt_max_keep: int = 3          # rotate: keep this many checkpoints
    ckpt_dir: str | None = None     # durable checkpoint dir (gs://... for spot); None = local
    compile_cache_dir: str | None = None  # persist XLA compiles (gs:///local) for fast restart
    # arena Elo metric (used when features.arena_gating is on): learner vs a frozen
    # anchor; unfinished games count as draws; anchor re-frozen when the learner clears
    # arena_threshold, chaining the elo/estimate curve.
    arena_interval: int = 10        # iters between learner-vs-anchor match rounds
    arena_games: int = 32           # games per color (played both colors)
    arena_threshold: float = 0.55   # score at which the anchor re-freezes to the learner
    eval_max_steps: int | None = None  # eval game length (None = selfplay.max_steps);
                                       # set LONGER so eval games actually finish
    # Adaptive self-play game length: tiers to hop between (each = one cached
    # compile) keeping game-completion fraction >= completion_target as the
    # bot's game length drifts. None = fixed selfplay.max_steps.
    max_steps_tiers: tuple | None = None
    completion_target: float = 0.80
    # Anti-forgetting: fraction of train steps drawn from the expert corpus
    # (pretraining shards w/ sharp values) instead of self-play replay. Anneal
    # toward 0 over the run so the teacher never caps final strength.
    corpus_mix: float = 0.0
    corpus_path: str | None = None  # glob of annotated shards
    seed: int = 0


@dataclass(frozen=True)
class Config:
    net: NetConfig = field(default_factory=NetConfig)
    mcts: MCTSConfig = field(default_factory=MCTSConfig)
    selfplay: SelfPlayConfig = field(default_factory=SelfPlayConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    features: FeaturesConfig = field(default_factory=FeaturesConfig)

    def to_dict(self):
        return asdict(self)


def tiny_config() -> Config:
    """A CPU-runnable smoke-test config (tiny net, few sims/games/steps)."""
    return Config(
        net=NetConfig(backbone="resnet", channels=16, blocks=2, use_se=False),
        mcts=MCTSConfig(num_simulations=8, max_num_considered_actions=8),
        selfplay=SelfPlayConfig(batch_size=8, max_steps=40),
        train=TrainConfig(train_batch_size=64, iterations=2, train_steps_per_iter=4,
                          replay_capacity=4000, warmup_steps=5),
    )


def tiny_transformer_config() -> Config:
    """Same but with the transformer backbone, to exercise architecture swapping."""
    return replace(tiny_config(),
                   net=NetConfig(backbone="transformer", channels=32, blocks=2, num_heads=4))
