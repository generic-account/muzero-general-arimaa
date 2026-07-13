"""A/B strength check: identical params, v2 search vs v3 compact search.

Both sides use the pinned eval shape (n=32/m=16). Expected result ~0.50: the
implementations are mathematically identical except documented fp-tie and
duplicated-edge-value semantics. A significant deviation would veto v3.

Usage (VM, repo root): python tools/ab_search.py <model.pkl> [games_per_color]
"""
import dataclasses
import pickle
import sys

import jax

from jaxarimaa import distributed, evaluate, trainer
from jaxarimaa.config import (Config, FeaturesConfig, MCTSConfig, NetConfig,
                              SelfPlayConfig, TrainConfig)


def main():
    path = sys.argv[1]
    games = int(sys.argv[2]) if len(sys.argv) > 2 else 128
    with open(path, "rb") as f:
        obj = pickle.load(f)
    params = obj["params"] if isinstance(obj, dict) and "params" in obj else obj

    feats_a = FeaturesConfig(
        bf16=True, fast_search=True, moves_left_head=True, dense_aux=True,
        planes_frozen=True, planes_trap=True, planes_step_in_turn=True,
        planes_moved=True)
    feats_b = dataclasses.replace(feats_a, compact_search=True)
    cfg = Config(net=NetConfig(channels=128, blocks=10), features=feats_a)
    model = trainer.make_model(cfg)
    mesh = distributed.make_mesh()
    params = distributed.replicate_tree(mesh, params)

    k1, k2 = jax.random.split(jax.random.PRNGKey(20260713))
    # A(v2) as gold vs B(v3) as silver, then colors swapped.
    a1, b1, u1 = evaluate.play_match(model, params, params, k1, 0, games, 384,
                                     32, 16, feats_a, True,
                                     features_b=feats_b)
    b2, a2, u2 = evaluate.play_match(model, params, params, k2, 0, games, 384,
                                     32, 16, feats_b, True,
                                     features_b=feats_a)
    w_b = int(b1) + int(b2)
    w_a = int(a1) + int(a2)
    d = int(u1) + int(u2)
    total = w_a + w_b + d
    score_b = (w_b + 0.5 * d) / max(total, 1)
    print(f"A/B(v3 compact) score: {score_b:.3f} "
          f"(v3 W{w_b} v2 W{w_a} D{d}, n={total})")


if __name__ == "__main__":
    main()
