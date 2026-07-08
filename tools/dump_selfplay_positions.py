"""Dump SELF-PLAY positions (raw boards) to a shard for sharp annotation.

Value re-grounding pipeline (fix for the warm-start drift: the value head is
pretrained on ARCHIVE positions but steers search on SELF-PLAY positions,
where real outcomes are near coin-flips and unlearnable — sharp's static eval
is dense and position-predictable on ANY distribution):

  1. this tool (TPU or CPU): pretrained net plays itself with production
     search knobs; every position is dumped in the 7-col shard schema
  2. tools/annotate_sharp.py (local, has the sharp binary): adds sharp_value
  3. tools/pretrain.py --init-params <pretrained> --freeze-trunk-policy
     --policy-weight 0 --sharp-weight 1.0: value/aux heads re-ground on the
     self-play distribution, policy+trunk untouched

Usage: PYTHONPATH=. python -u tools/dump_selfplay_positions.py \
           --params results/jaxarimaa/pretrained_c256.pkl \
           --out results/selfplay_positions/sp0.npz \
           [--games 512] [--max-steps 512] [--seed 0]
"""
import argparse
import os

import numpy as np
import jax
import jax.numpy as jnp

from jaxarimaa import checkpoint, env as jenv, fast_search, trainer
from jaxarimaa.config import Config, FeaturesConfig, MCTSConfig, NetConfig

ap = argparse.ArgumentParser()
ap.add_argument("--params", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--games", type=int, default=512)
ap.add_argument("--max-steps", type=int, default=512)
ap.add_argument("--sims", type=int, default=32)
ap.add_argument("--considered", type=int, default=16)
ap.add_argument("--greedy-after", type=int, default=15)
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()

feats = FeaturesConfig(
    bf16=True, fast_search=True, adjudicate_truncation=True,
    moves_left_head=True, planes_frozen=True, planes_trap=True,
    planes_step_in_turn=True, planes_moved=True)
cfg = Config(net=NetConfig(channels=256, blocks=15), features=feats,
             mcts=MCTSConfig(num_simulations=args.sims,
                             max_num_considered_actions=args.considered))
params, _ = checkpoint.load(args.params)
model = trainer.make_model(cfg)


@jax.jit
def play_step(states, key):
    out = fast_search.run_search(model, params, key, states, args.sims,
                                 args.considered, feats)
    action = out.action
    if args.greedy_after:
        greedy = states.rep_ptr - 1 >= args.greedy_after
        action = jnp.where(greedy,
                           jnp.argmax(out.action_weights, -1).astype(action.dtype),
                           action)
    nstates = jax.vmap(jenv.step)(states, action)
    return nstates, action


key = jax.random.PRNGKey(args.seed)
key, kinit = jax.random.split(key)
states = jax.vmap(jenv.init_state)(jax.random.split(kinit, args.games))
cols = {k: [] for k in ("board", "player", "steps_left", "turn_start", "action")}
for t in range(args.max_steps):
    key, ks, kr = jax.random.split(key, 3)
    cols["board"].append(np.asarray(states.board))
    cols["player"].append(np.asarray(states.player))
    cols["steps_left"].append(np.asarray(states.steps_left))
    cols["turn_start"].append(np.asarray(states.turn_start_board))
    nstates, action = play_step(states, ks)
    cols["action"].append(np.asarray(action))
    fresh = jax.vmap(jenv.init_state)(jax.random.split(kr, args.games))
    states = jenv.where_state(nstates.terminated, fresh, nstates)  # auto-reset
    if (t + 1) % 64 == 0:
        print(f"step {t + 1}/{args.max_steps}", flush=True)

out = {k: np.concatenate(v) for k, v in cols.items()}
n = len(out["action"])
# schema compat: value/moves_left placeholders (sharp_value replaces value's
# role downstream at --sharp-weight 1.0; moves_left is a capped don't-care)
out["value"] = np.zeros(n, np.float32)
out["moves_left"] = np.full(n, 64, np.int32)
out["action"] = out["action"].astype(np.int32)
os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
np.savez_compressed(args.out, **out)
print(f"wrote {args.out}: {n} positions "
      f"({args.games} lanes x {args.max_steps} steps)", flush=True)
