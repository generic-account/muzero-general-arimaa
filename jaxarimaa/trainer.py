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
    return optax.chain(optax.clip_by_global_norm(tc.grad_clip), base)


def make_model(cfg: Config):
    """The network for `cfg`: compute dtype from features.bf16, aux heads from features."""
    import jax.numpy as _jnp
    dtype = _jnp.bfloat16 if cfg.features.bf16 else _jnp.float32
    f = cfg.features
    return net.make_network(cfg.net, dtype=dtype, moves_left_head=f.moves_left_head,
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


def loss_fn(params, apply_fn, batch, value_weight, aux_weights=(0.0, 0.0, 0.0)):
    ml_w, deep_w, mtp_w = aux_weights
    obs = batch["obs"]
    logits, value, aux = jax.vmap(lambda o: apply_fn(params, o))(obs)
    logp = jax.nn.log_softmax(logits, axis=-1)
    # targets may be stored bf16 in the replay buffer; accumulate the CE in f32
    policy_loss = -jnp.sum(batch["policy_target"].astype(jnp.float32) * logp, axis=-1)
    value_loss = (value - batch["value_target"]) ** 2
    # Per-sample weights (playout-cap: fast moves have weight 0). Default weight 1 =>
    # weighted mean is the plain mean, so baseline behavior is unchanged.
    w = batch.get("weight", jnp.ones_like(value_loss))
    wsum = jnp.sum(w) + 1e-8
    pol = _weighted_mean(policy_loss, w, wsum)
    val = _weighted_mean(value_loss, w, wsum)
    metrics = {"policy_loss": pol, "value_loss": val}
    total = pol + value_weight * val

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
            dp = -jnp.sum(batch["policy_target"] * jax.nn.log_softmax(pl_i, axis=-1), axis=-1)
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


@functools.partial(jax.jit, static_argnums=(2, 4, 5))
def train_step(state: TrainState, batch, value_weight, rng, symmetry=False,
               aux_weights=(0.0, 0.0, 0.0)):
    if symmetry:
        batch = _augment_symmetry(batch, rng)
    (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(
        state.params, state.apply_fn, batch, value_weight, aux_weights
    )
    state = state.apply_gradients(grads=grads)
    return state, metrics
