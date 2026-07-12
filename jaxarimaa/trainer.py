"""Training: AlphaZero loss (policy cross-entropy + value MSE) and a jitted,
sharding-friendly train step.

The train step is a plain jitted function operating on a (data-sharded) batch;
under GSPMD the batch-mean loss automatically all-reduces across devices, so the
same code is correct on 1 device or a whole slice.
"""

import functools

import jax
import jax.numpy as jnp
import optax
from flax.training import train_state

from . import constants as C
from . import network as net
from .config import Config


class TrainState(train_state.TrainState):
    pass


def make_optimizer(cfg: Config):
    tc = cfg.train
    # Linear-scaling rule: LR is specified for `lr_ref_batch`; scale to the actual
    # (global) train batch. Default ref = train_batch_size, so the scale is 1 (no-op)
    # until the user opts in by setting a smaller reference batch.
    ref = tc.lr_ref_batch or tc.train_batch_size
    peak = tc.lr * (tc.train_batch_size / ref)
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0, peak_value=peak, warmup_steps=tc.warmup_steps,
        decay_steps=max(tc.warmup_steps + 1, tc.iterations * tc.train_steps_per_iter),
        end_value=peak * 0.1,
    )
    name = tc.optimizer
    if name == "adamw":
        base = optax.adamw(schedule, weight_decay=tc.weight_decay)
    elif name == "adam":
        base = optax.adam(schedule)
    elif name == "lion":
        base = optax.lion(schedule, weight_decay=tc.weight_decay)
    elif name == "sgd":
        base = optax.chain(optax.add_decayed_weights(tc.weight_decay),
                           optax.sgd(schedule, momentum=0.9, nesterov=True))
    else:
        raise ValueError(f"unknown optimizer {name!r}")
    if tc.freeze_trunk_policy:
        # Value-calibration phase: zero updates for the backbone and the policy
        # head (Conv_0/Dense_0 are created first in ArimaaNet._heads, so the
        # names are stable across feature combos); value/aux heads still train.
        # NOTE optax.masked would PASS RAW GRADS THROUGH for masked-out leaves
        # (not freeze them) — multi_transform + set_to_zero is the freezing form.
        frozen = {"ResNetBackbone_0", "TransformerBackbone_0", "Conv_0", "Dense_0"}

        def _labels(params):
            return jax.tree_util.tree_map_with_path(
                lambda path, _: "freeze" if any(getattr(p, "key", None) in frozen
                                                for p in path) else "train", params)

        base = optax.multi_transform({"train": base,
                                      "freeze": optax.set_to_zero()}, _labels)
    return optax.chain(optax.clip_by_global_norm(tc.grad_clip), base)


def make_model(cfg: Config):
    """The network for `cfg`: compute dtype from features.bf16, aux heads from features."""
    import jax.numpy as _jnp
    dtype = _jnp.bfloat16 if cfg.features.bf16 else _jnp.float32
    f = cfg.features
    return net.make_network(cfg.net, dtype=dtype, moves_left_head=f.moves_left_head,
                            dense_aux=f.dense_aux,
                            deep_supervision=f.deep_supervision, mtp=f.mtp,
                            smolgen=f.smolgen, rope=f.rope)


def create_train_state(cfg: Config, rng) -> TrainState:
    from . import env as jenv
    model = make_model(cfg)
    obs = jenv.observe(jenv.init_state(rng), cfg.features)  # features fix input planes
    params = model.init(rng, obs)
    return TrainState.create(apply_fn=model.apply, params=params,
                             tx=make_optimizer(cfg))


def _weighted_mean(x, w, wsum):
    return jnp.sum(w * x) / wsum


def loss_fn(params, apply_fn, batch, value_weight, aux_weights=(0.0, 0.0, 0.0, 0.0),
            policy_weight=1.0, value_tail_weight=1.0, anchor_params=None,
            kl_weight=0.0):
    # aux_weights = (moves_left, deep_supervision, mtp, dense); 3-tuples accepted
    # for backward compatibility (dense weight 0).
    if len(aux_weights) == 3:
        aux_weights = (*aux_weights, 0.0)
    ml_w, deep_w, mtp_w, dense_w = aux_weights
    obs = batch["obs"]
    logits, value, aux = jax.vmap(lambda o: apply_fn(params, o))(obs)
    logp = jax.nn.log_softmax(logits, axis=-1)
    # targets may be stored bf16 in the replay buffer; accumulate the CE in f32
    policy_loss = -jnp.sum(batch["policy_target"].astype(jnp.float32) * logp, axis=-1)
    value_sq = (value - batch["value_target"]) ** 2
    value_loss = value_sq
    if "value_real" in batch:
        # bootstrapped/adjudicated tails get down-weighted in the VALUE loss only
        # (identity at value_tail_weight=1; kept traceable so the anneal can move
        # the weight without recompiling)
        vr = batch["value_real"]
        value_loss = value_sq * (vr + (1.0 - vr) * value_tail_weight)
    # Optional per-sample weights. No current producer emits "weight" (self-play
    # filters fast-move rows out rather than down-weighting), so this defaults to
    # ones => plain mean; kept as a hook (e.g. KataGo-style per-term weighting).
    w = batch.get("weight", jnp.ones_like(value_loss))
    wsum = jnp.sum(w) + 1e-8
    pol = _weighted_mean(policy_loss, w, wsum)
    val = _weighted_mean(value_loss, w, wsum)
    metrics = {"policy_loss": pol, "value_loss": val}
    if "value_real" in batch:
        # `value_loss` above is diluted by masked tail rows (numerator zeroed,
        # denominator counts all rows) — report the undiluted real-rows-only MSE,
        # the actual value-head health signal.
        rw = w * batch["value_real"]
        rsum = jnp.sum(rw) + 1e-8
        metrics["value_real_mse"] = jnp.sum(rw * value_sq) / rsum
        metrics["value_real_frac"] = jnp.sum(rw) / wsum
    total = policy_weight * pol + value_weight * val
    if anchor_params is not None:
        # Trust region to the pretrained prior ON THE TRAINING BATCH: bounds
        # policy drift exactly where gradients act, unlike corpus-mix (which
        # anchors on archive positions). Live self-play training measured ~10x
        # more damaging per step than identical static data — this caps the
        # drift at a knob (annealed to 0 by the ratchet as improvement proves
        # real, so it cannot cap final strength).
        a_logits, _, _ = jax.vmap(
            lambda o: apply_fn(jax.lax.stop_gradient(anchor_params), o))(obs)
        a_logp = jax.nn.log_softmax(a_logits.astype(jnp.float32), axis=-1)
        kl = jnp.sum(jnp.exp(a_logp) * (a_logp - logp), axis=-1)
        klm = _weighted_mean(kl, w, wsum)
        total = total + kl_weight * klm
        metrics["kl_prior"] = klm

    if "dense" in aux and "dense_target" in batch:
        dl = _weighted_mean(
            jnp.mean((aux["dense"] - batch["dense_target"]) ** 2, axis=-1), w, wsum)
        total = total + dense_w * dl
        metrics["dense_loss"] = dl
    if "moves_left" in aux and "moves_left_target" in batch:
        ml = _weighted_mean((aux["moves_left"] - batch["moves_left_target"]) ** 2, w, wsum)
        total = total + ml_w * ml
        metrics["moves_left_loss"] = ml
    if "mtp_value" in aux and "mtp_value_target" in batch:
        mm = w * batch["mtp_mask"]
        mtp = jnp.sum(mm * (aux["mtp_value"] - batch["mtp_value_target"]) ** 2) / (jnp.sum(mm) + 1e-8)
        total = total + mtp_w * mtp
        metrics["mtp_loss"] = mtp
    if "deep" in aux:
        dl = 0.0
        for pl_i, v_i in aux["deep"]:
            dp = -jnp.sum(batch["policy_target"].astype(jnp.float32)
                          * jax.nn.log_softmax(pl_i, axis=-1), axis=-1)
            dv = (v_i - batch["value_target"]) ** 2
            dl = dl + _weighted_mean(dp, w, wsum) + value_weight * _weighted_mean(dv, w, wsum)
        dl = dl / len(aux["deep"])
        total = total + deep_w * dl
        metrics["deep_loss"] = dl

    metrics["loss"] = total
    return total, metrics


_SYM_PERM = jnp.asarray(C.SYM_PERM)


def _augment_symmetry(batch, rng):
    """Left-right mirror (x->7-x) a random half of the batch: flip obs on the x axis
    and permute the policy target by the induced action permutation. Value is invariant."""
    obs, pol = batch["obs"], batch["policy_target"]
    m = jax.random.bernoulli(rng, 0.5, (obs.shape[0],))
    obs = jnp.where(m[:, None, None, None], jnp.flip(obs, axis=-1), obs)
    pol = jnp.where(m[:, None], pol[:, _SYM_PERM], pol)
    return {**batch, "obs": obs, "policy_target": pol}


# value/policy/tail weights are TRACED scalars (not static): they're plain
# multipliers in the loss, and the anneal controller (train.py) moves them
# mid-run — traced, a weight change costs nothing; static, it would recompile.
@functools.partial(jax.jit, static_argnums=(4, 5))
def train_step(state: TrainState, batch, value_weight, rng, symmetry=False,
               aux_weights=(0.0, 0.0, 0.0, 0.0), policy_weight=1.0,
               value_tail_weight=1.0, anchor_params=None, kl_weight=0.0):
    if symmetry:
        batch = _augment_symmetry(batch, rng)
    (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(
        state.params, state.apply_fn, batch, value_weight, aux_weights,
        policy_weight, value_tail_weight, anchor_params, kl_weight
    )
    state = state.apply_gradients(grads=grads)
    return state, metrics
