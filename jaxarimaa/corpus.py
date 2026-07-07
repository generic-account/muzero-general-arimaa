"""Expert-corpus batches for mixing into self-play training (anti-forgetting).

Loads the sharp-annotated imitation shards (see tools/archive_dataset.py /
tools/annotate_sharp.py) and serves training batches in the same format as the
replay buffer: obs + one-hot expert policy target + blended value target
(+ moves-left). Mixing a small annealed fraction of these into self-play
training keeps the pretrained knowledge anchored without capping strength
(the fraction can decay to zero as the learner surpasses the teacher).
"""

import glob

import jax
import jax.numpy as jnp
import numpy as np

from . import constants as C
from . import env as jenv


class CorpusSampler:
    """Host-resident corpus + jitted batch builder (same keys as replay batches)."""

    def __init__(self, pattern, features, sharp_weight=0.5, with_moves_left=True):
        files = sorted(glob.glob(pattern))
        if not files:
            raise FileNotFoundError(f"no corpus shards match {pattern}")
        cols = {k: [] for k in ("board", "player", "steps_left", "turn_start",
                                "action", "value", "moves_left")}
        sharp = []
        for f in files:
            d = np.load(f)
            for k in cols:
                cols[k].append(d[k])
            sharp.append(d["sharp_value"] if "sharp_value" in d else d["value"])
        self.data = {k: np.concatenate(v) for k, v in cols.items()}
        self.data["sharp_value"] = np.concatenate(sharp)
        self.n = len(self.data["action"])
        self.features = features
        self.with_moves_left = with_moves_left
        w = float(sharp_weight)

        @jax.jit
        def build(board, player, steps_left, turn_start, action, value, sharpv, ml):
            st = jenv.state_from_batch(board, player, steps_left,
                                       turn_start=turn_start)
            # bf16 to match replay-batch dtypes exactly -> one train_step graph
            obs = jax.vmap(lambda s: jenv.observe(s, features))(st).astype(jnp.bfloat16)
            pol = jax.nn.one_hot(action, C.N_ACTIONS, dtype=jnp.bfloat16)
            vt = (1.0 - w) * value + w * sharpv
            out = {"obs": obs, "policy_target": pol, "value_target": vt}
            if with_moves_left:
                out["moves_left_target"] = ml
            return out

        self._build = build

    def sample(self, rng_np, batch_size, shard_fn=None):
        """Draw a random batch. `shard_fn` (e.g. distributed.shard_batch partial)
        places the raw arrays before the jitted build so obs-building parallelizes."""
        idx = rng_np.integers(0, self.n, size=batch_size)
        d = self.data
        raw = dict(board=d["board"][idx], player=d["player"][idx],
                   steps_left=d["steps_left"][idx], turn_start=d["turn_start"][idx],
                   action=d["action"][idx].astype(np.int32),
                   value=d["value"][idx], sharpv=d["sharp_value"][idx],
                   ml=(np.minimum(d["moves_left"][idx], C.MOVES_LEFT_CAP)
                       .astype(np.float32) / C.MOVES_LEFT_CAP))
        if shard_fn is not None:
            raw = shard_fn(raw)
        return self._build(**raw)
