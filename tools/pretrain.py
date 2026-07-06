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

MLCAP = 64.0  # matches selfplay moves-left normalization


def load_shards(patterns):
    files = []
    for p in patterns:
        files += sorted(glob.glob(p))
    if not files:
        raise SystemExit(f"no shards matched {patterns}")
    keys = ["board", "player", "steps_left", "turn_start", "action", "value", "moves_left"]
    cols = {k: [] for k in keys}
    has_sharp = True
    sharp = []
    for f in files:
        d = np.load(f)
        for k in keys:
            cols[k].append(d[k])
        if "sharp_value" in d:
            sharp.append(d["sharp_value"])
        else:
            has_sharp = False
    out = {k: np.concatenate(v) for k, v in cols.items()}
    out["sharp_value"] = (np.concatenate(sharp) if has_sharp
                          else out["value"].copy())  # fallback = outcome (no blend effect)
    out["_has_sharp"] = has_sharp
    return out, files


def build_features(args):
    # Match the intended self-play run's planes so the init transfers directly.
    return FeaturesConfig(
        bf16=args.bf16, planes_frozen=True, planes_trap=True,
        planes_step_in_turn=True, planes_moved=True,
        moves_left_head=True, symmetry_aug=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", nargs="+", required=True)
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

    data, files = load_shards(args.shards)
    N = len(data["action"])
    print(f"loaded {N} samples from {len(files)} shards | "
          f"sharp_value {'PRESENT' if data['_has_sharp'] else 'absent (outcome-only)'}")

    cfg = Config(net=NetConfig(channels=args.channels, blocks=args.blocks),
                 train=TrainConfig(lr=args.lr, iterations=args.epochs,
                                   train_batch_size=args.batch,
                                   warmup_steps=200),
                 features=build_features(args))
    feats = cfg.features
    key = jax.random.PRNGKey(args.seed)
    key, kinit = jax.random.split(key)
    # NB: create_train_state builds the optimizer from cfg.train; the cosine
    # decay_steps below is derived from epochs*steps, so set iterations already.
    steps_per_epoch = N // args.batch
    cfg = cfg.__class__(net=cfg.net, mcts=cfg.mcts, selfplay=cfg.selfplay,
                        features=cfg.features,
                        train=TrainConfig(lr=args.lr, warmup_steps=200,
                                          train_batch_size=args.batch,
                                          iterations=1,
                                          train_steps_per_iter=max(1, args.epochs * steps_per_epoch)))
    state = trainer.create_train_state(cfg, kinit)
    model = trainer.make_model(cfg)

    boards = jnp.asarray(data["board"]); players = jnp.asarray(data["player"])
    lefts = jnp.asarray(data["steps_left"]); tstarts = jnp.asarray(data["turn_start"])
    actions = jnp.asarray(data["action"].astype(np.int32))
    outcome = jnp.asarray(data["value"])
    sharpv = jnp.asarray(data["sharp_value"])
    ml = jnp.asarray(np.minimum(data["moves_left"], MLCAP).astype(np.float32) / MLCAP)

    def make_batch(idx):
        st = jenv.state_from_batch(boards[idx], players[idx], lefts[idx],
                                   turn_start=tstarts[idx])
        obs = jax.vmap(lambda s: jenv.observe(s, feats))(st)  # vmap over batched State
        pol = jax.nn.one_hot(actions[idx], C.N_ACTIONS, dtype=jnp.float32)
        vt = (1.0 - args.sharp_weight) * outcome[idx] + args.sharp_weight * sharpv[idx]
        return {"obs": obs, "policy_target": pol, "value_target": vt,
                "moves_left_target": ml[idx], "weight": jnp.ones_like(vt)}

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
        print(f"[epoch {epoch}] saved -> {args.out}", flush=True)

    # --- value-discrimination diagnostic ---
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
