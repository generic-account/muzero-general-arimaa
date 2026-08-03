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
    visit_policy_targets: bool = False
    # --- Stage-2 (optima/KataGo-informed from-scratch loop) gates ---
    deblunder: bool = False            # value targets = outcome Q-mixed past exploration
                                       # blunders (records q_chosen/q_best per step)
    dense_aux: bool = False            # dense Arimaa aux targets: trap ownership [4],
                                       # capture-in-k [2], material trajectory [1]
    prune_policy_targets: bool = False  # zero action_weights mass on unvisited actions
                                        # + renormalize (closes the Q-imputation channel)
    certification: bool = False        # self-play uses last CERTIFIED (arena-passing)
                                       # params, not the raw learner
    truncation_draw: bool = False      # max_steps games scored 0 (draw) as REAL value
                                       # targets, optima-style (vs material adjudication)  # policy target = root visit counts (optima/AZ style;
                                        # zero mass on unvisited actions) instead of Gumbel
                                        # action_weights (which impute Q-reweighted mass for
                                        # NEVER-visited actions — poison when value is OOD)
    adjudicate_truncation: bool = False  # truncated games: material/advancement adjudication (env.material_eval) instead of net bootstrap
    # --- Stage-2.2 (breaking the equal-strength value wall; see memory
    #     warmstart-stage2: outcome-only value targets go silent/noisy as games
    #     approach 50/50 + truncation draws, and that ALONE collapses play) ---
    ml_steering: bool = False          # optima moves-left steering: child values
                                       # shifted toward faster wins/slower losses
                                       # inside search (requires moves_left_head)
    rollout_resolve: bool = False      # finish cap-truncated games with cheap
                                       # policy-only rollouts -> REAL outcome
                                       # labels instead of false draws
    qmix_value: bool = False           # value targets = qmix_lambda*outcome +
                                       # (1-lambda)*root search Q (TD-style
                                       # variance reduction at equal strength)
    handicap_games: bool = False       # a fraction of games start one non-rabbit
                                       # piece down: decisive, honestly-labeled
                                       # games at ANY strength (KataGo handicap)
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
    policy_loss_weight: float = 1.0
    # Trust region: weight of KL(pretrained prior || policy) computed ON the
    # training batches (needs init_params). Bounds warm-start policy drift at
    # its source; annealed to anneal_kl_prior_weight by the ratchet.
    kl_prior_weight: float = 0.0
    anneal_kl_prior_weight: float = 0.0
    # FIXED leash anchor: path to a pkl to hold the KL trust region to. None =
    # the run's own warm-start init (the ROLLING anchor that stair-stepped
    # segment boundaries downward: each relaunch re-anchored the leash to a
    # lower reference). Set this to the best-known-good prior (plus9).
    kl_anchor_path: str | None = None
    # AlphaGo-style value-calibration phase: train ONLY the value/aux heads on a
    # frozen trunk + frozen policy head (optax.masked). Lets the value head fit
    # the self-play outcome distribution without churning the features the
    # pretrained policy depends on (value-only WITH trunk measured 0.27-0.43
    # arena damage; head-only is the safe variant).
    freeze_trunk_policy: bool = False
    value_tail_weight: float = 1.0  # weight of bootstrapped/adjudicated rows in the value loss  # 0 = value-only warmup (protect a pretrained policy while value calibrates)
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
    # Certification-deadlock escape (measured on s2pilot 2026-07-13: six
    # consecutive gates at 0.42-0.49 with all losses flat — the learner fully
    # distills the frozen generator's data, and the n=128 improvement step is
    # smaller than the gate margin, so nothing ever promotes). After this many
    # consecutive failed gates, self-play generation switches to the LEARNER
    # (fresh on-policy data) until the next promotion re-certifies. 0 = off.
    probation_after: int = 0
    # --- Stage-2.2 knobs (active only with their feature gates) ---
    ml_steer_weight: float = 0.2    # bounded value shift: +-0.5*w at ml extremes
    resolve_steps: int = 256        # policy-only rollout budget for truncated games
    qmix_lambda: float = 0.7        # weight on OUTCOME in the value target mix
    handicap_frac: float = 0.1      # fraction of games starting a piece down
    # EVAL search shape, pinned independently of TRAINING search shape: arms
    # that train with different sims must still be MEASURED identically (arena,
    # rung, eval-vs-random), else readings compare search depth, not nets.
    # None = use cfg.mcts (backward compatible).
    eval_num_sims: int | None = None
    eval_num_considered: int | None = None
    # UNBIASED Elo: every ref_interval arena rounds, also play the learner vs a
    # FROZEN reference rung (initially the warm-start init) -> elo/vs_ref. No
    # promotion decision selects on this reading, unlike the chained estimate
    # (which inflates: promotions select on noise, ~+35 per false promotion,
    # never revert). When the rung saturates (score > 0.95) a new rung freezes
    # and the gap is calibrated with a DEDICATED fresh match. 0 = off.
    ref_interval: int = 0
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
    # Adaptive anneal ("trust ratchet", train.py): walk the value-conservative
    # warm-start knobs (value_loss_weight, value_tail_weight, corpus_mix) from
    # their configured values toward the anneal_* finals in `anneal_stages` equal
    # steps, gated on the arena anchor-Elo — advance one stage per arena while the
    # Elo is within anneal_hold_band of its best, retreat one stage if it drops
    # more than anneal_backoff below best. 0 stages = fixed knobs. Requires
    # features.arena_gating (the Elo signal). Stage changes are FREE: the three
    # knobs are traced scalars in train_step, so no re-jit on any transition.
    anneal_stages: int = 0
    anneal_value_loss_weight: float = 1.0   # final value_loss_weight
    anneal_value_tail_weight: float = 0.25  # final value_tail_weight
    anneal_corpus_mix: float = 0.0          # final corpus_mix
    anneal_hold_band: float = 30.0   # Elo below best still counted healthy (~1 sigma @128 games)
    anneal_backoff: float = 75.0     # Elo drop below best that triggers a retreat (~2.5 sigma)
    anneal_best_leak: float = 5.0    # Elo/round the best-so-far baseline decays: plateaus
                                     # re-probe eventually instead of parking regressed
                                     # (small vs backoff, so a real cliff still parks long)
    # --- Stage-2 knobs ---
    deblunder_threshold: float = 0.15  # q_best - q_chosen above this = blunder
    deblunder_width: float = 0.15      # mix ramps 0->1 over this width past threshold
    dense_aux_weight: float = 0.3      # loss weight of the dense aux head
    dense_aux_k: int = 32              # steps-horizon for capture-in-k / material traj
    surprise_weight: float = 0.0       # per-row weight 1 + s*KL(target||prior); 0 = off
    prior_temp: float = 1.0            # >1 flattens net priors fed to search (anti-
                                       # self-sharpening, optima uses 1.2)
    # Slow-bleed guard: a decline of ~hold_band per round never trips the
    # backoff (the leaky best follows it down). Two consecutive rounds scoring
    # below this floor vs the CURRENT anchor = actively losing to a fixed
    # opponent -> retreat regardless of the Elo baseline. 0 = off.
    anneal_score_floor: float = 0.45
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
