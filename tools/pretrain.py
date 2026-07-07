"""Supervised pretraining on imitation shards (human archive + sharp self-play).

Trains the SAME network (policy + value + moves-left heads) used in self-play, on
expert games, to produce an initialization checkpoint that skips the value
cold-start (see memory/cold-start-value-collapse): every position carries a real
game-outcome value, optionally blended with sharp's static-eval win-probability
(NNUE-style; the `sharp_value` shard column, added by tools/annotate_sharp.py).

Policy target = one-hot expert action (cross-entropy). Value target =
(1-w)*game_outcome + w*sharp_value on shards that have sharp_value, else pure
outcome. Observations are built on the fly per batch at the run's FeaturesConfig
(so obs planes always match the training config; no giant precomputed obs array).

After training it runs the value-discrimination diagnostic — the single number
(value-output std across diverse mid-game positions) that says whether the value
head is genuinely grounded (val2's collapsed head was 0.017).

Usage:
    python tools/pretrain.py --shards 'results/archive_ds/*.npz' 'results/sharp_ds/*.npz' \
        --channels 256 --blocks 15 --epochs 3 --sharp-weight 0.5 \
        --out results/jaxarimaa/pretrained.pkl
"""

import argparse
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from jaxarimaa import checkpoint, env as jenv, trainer  # noqa: E402
from jaxarimaa.config import (Config, FeaturesConfig, NetConfig, TrainConfig)  # noqa: E402
from jaxarimaa import constants as C  # noqa: E402
from tools import shard_io  # noqa: E402

MLCAP = C.MOVES_LEFT_CAP  # single source of truth (matches selfplay normalization)


def build_features(args):
    # Match the intended self-play run's planes so the init transfers directly.
    return FeaturesConfig(
        bf16=args.bf16, planes_frozen=True, planes_trap=True,
        planes_step_in_turn=True, planes_moved=True,
        moves_left_head=True, symmetry_aug=True)


def _eval_val(model, params, feats, val, batch, sharp_w):
    """Held-out metrics: policy top-1 accuracy (predicts the expert move?),
    value MSE, and value-output std (grounding). Split by shard/year, so no
    intra-game leakage into train."""
    import numpy as _np
    boards, players, lefts, tstarts = (jnp.asarray(val[k]) for k in
                                       ("board", "player", "steps_left", "turn_start"))
    acts = jnp.asarray(val["action"].astype(_np.int32))
    outcome = jnp.asarray(val["value"]); sharpv = jnp.asarray(val["sharp_value"])
    N = len(acts); corr = vmse = 0.0; vals = []
    for s in range(0, N, batch):
        idx = slice(s, min(s + batch, N))
        st = jenv.state_from_batch(boards[idx], players[idx], lefts[idx],
                                   turn_start=tstarts[idx])
        obs = jax.vmap(lambda s: jenv.observe(s, feats))(st)
        logits, value, _ = jax.vmap(lambda o: model.apply(params, o))(obs)
        pred = jnp.argmax(logits, -1)
        corr += float(jnp.sum(pred == acts[idx]))
        vt = (1 - sharp_w) * outcome[idx] + sharp_w * sharpv[idx]
        vmse += float(jnp.sum((value - vt) ** 2))
        vals.append(_np.asarray(value))
    vals = _np.concatenate(vals)
    return corr / N, vmse / N, float(vals.std())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", nargs="+", required=True)
    ap.add_argument("--val-shards", nargs="+", default=None,
                    help="held-out shards for validation (e.g. a whole year) — "
                         "keeps game-correlated positions out of train")
    ap.add_argument("--channels", type=int, default=256)
    ap.add_argument("--blocks", type=int, default=15)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--sharp-weight", type=float, default=0.0,
                    help="value target = (1-w)*outcome + w*sharp_winprob")
    ap.add_argument("--value-weight", type=float, default=1.0,
                    help="value-loss weight relative to policy CE")
    ap.add_argument("--bf16", action="store_true", default=True)
    ap.add_argument("--out", default="results/jaxarimaa/pretrained.pkl")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    files = [f for p in args.shards for f in sorted(glob.glob(p))]
    if not files:
        raise SystemExit(f"no shards matched {args.shards}")
    data = shard_io.load_shards(files)
    N = len(data["action"])
    print(f"loaded {N} samples from {len(files)} shards | "
          f"sharp_value {'PRESENT' if data['_has_sharp'] else 'absent (outcome-only)'}")
    val = None
    if args.val_shards:
        vfiles = [f for p in args.val_shards for f in sorted(glob.glob(p))]
        val = shard_io.load_shards(vfiles)
        print(f"validation: {len(val['action'])} samples from {len(vfiles)} shards "
              f"({vfiles})")

    # Build the config once: the optimizer's cosine schedule (create_train_state)
    # needs the total step count up front, so derive it before constructing cfg.
    steps_per_epoch = N // args.batch
    cfg = Config(net=NetConfig(channels=args.channels, blocks=args.blocks),
                 features=build_features(args),
                 train=TrainConfig(lr=args.lr, warmup_steps=200,
                                   train_batch_size=args.batch, iterations=1,
                                   train_steps_per_iter=max(1, args.epochs * steps_per_epoch)))
    feats = cfg.features
    key = jax.random.PRNGKey(args.seed)
    key, kinit = jax.random.split(key)
    state = trainer.create_train_state(cfg, kinit)
    model = trainer.make_model(cfg)

    boards = jnp.asarray(data["board"]); players = jnp.asarray(data["player"])
    lefts = jnp.asarray(data["steps_left"]); tstarts = jnp.asarray(data["turn_start"])
    actions = jnp.asarray(data["action"].astype(np.int32))
    outcome = jnp.asarray(data["value"])
    sharpv = jnp.asarray(data["sharp_value"])
    ml = jnp.asarray(np.minimum(data["moves_left"], MLCAP).astype(np.float32) / MLCAP)

    @jax.jit  # fuse the gather + obs-build + one-hot per step (offline speed)
    def make_batch(idx):
        st = jenv.state_from_batch(boards[idx], players[idx], lefts[idx],
                                   turn_start=tstarts[idx])
        obs = jax.vmap(lambda s: jenv.observe(s, feats))(st)  # vmap over batched State
        pol = jax.nn.one_hot(actions[idx], C.N_ACTIONS, dtype=jnp.float32)
        vt = (1.0 - args.sharp_weight) * outcome[idx] + args.sharp_weight * sharpv[idx]
        # value target = (1-w)*game_outcome + w*sharp_winprob; no per-sample weight
        # (loss_fn defaults missing "weight" to ones — all rows train equally).
        return {"obs": obs, "policy_target": pol, "value_target": vt,
                "moves_left_target": ml[idx]}

    aux_w = (cfg.train.moves_left_weight, 0.0, 0.0)
    step = 0
    for epoch in range(args.epochs):
        key, kperm = jax.random.split(key)
        perm = np.asarray(jax.random.permutation(kperm, N))
        for b in range(steps_per_epoch):
            idx = jnp.asarray(perm[b * args.batch:(b + 1) * args.batch])
            key, krng = jax.random.split(key)
            batch = make_batch(idx)
            state, m = trainer.train_step(state, batch, args.value_weight, krng,
                                          feats.symmetry_aug, aux_w)
            step += 1
            if step % 100 == 0:
                print(f"epoch {epoch} step {step}: loss={float(m['loss']):.3f} "
                      f"pol={float(m['policy_loss']):.3f} val={float(m['value_loss']):.3f}",
                      flush=True)
        checkpoint.save(args.out, state.params,
                        {"config": cfg.to_dict(), "epoch": epoch + 1, "step": step})
        msg = f"[epoch {epoch}] saved -> {args.out}"
        if val is not None:
            acc, vmse, vstd = _eval_val(model, state.params, feats, val,
                                        args.batch, args.sharp_weight)
            msg += (f" | VAL policy-acc={acc:.3f} value-mse={vmse:.3f} "
                    f"value-std={vstd:.3f}")
        print(msg, flush=True)

    # --- value-discrimination diagnostic (on fresh random-play states) ---
    diag_value_discrimination(model, state.params, feats)


def diag_value_discrimination(model, params, feats):
    """Value-output std across diverse mid-game positions. Grounded value >> 0.05;
    val2's collapsed head measured 0.017."""
    keys = jax.random.split(jax.random.PRNGKey(12345), 256)
    states = jax.jit(jax.vmap(jenv.init_state))(keys)
    stepv = jax.jit(jax.vmap(lambda s, a: jenv.step(s, a)))
    for i in range(14):
        legal = jax.vmap(jenv.legal_action_mask)(states)
        g = jax.random.gumbel(jax.random.PRNGKey(i + 1), legal.shape)
        states = stepv(states, jnp.argmax(jnp.where(legal, g, -jnp.inf), -1))
    obs = jax.vmap(lambda s: jenv.observe(s, feats))(states)
    _, val, _ = jax.vmap(lambda o: model.apply(params, o))(obs)
    val = np.asarray(val)
    print(f"\n=== value-discrimination diagnostic ===")
    print(f"value output: mean={val.mean():+.3f} std={val.std():.3f} "
          f"min={val.min():+.3f} max={val.max():+.3f}")
    print(f"  -> {'GROUNDED' if val.std() > 0.05 else 'COLLAPSED (like val2 0.017)'}")


if __name__ == "__main__":
    main()
