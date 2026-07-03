"""Policy/value network: backbone(obs) -> policy logits (N_ACTIONS) + value.

The backbone is fully swappable/parameterized via NetConfig (see backbones.py).
Convention: modules take a SINGLE unbatched observation (C,H,W); batch with
jax.vmap. The value head returns a scalar in [-1, 1] (tanh).
"""

import jax
import flax.linen as nn
import jax.numpy as jnp

from . import constants as C
from .backbones import build_backbone
from .config import NetConfig


class ArimaaNet(nn.Module):
    cfg: NetConfig
    num_actions: int = C.N_ACTIONS
    dtype: object = jnp.float32     # compute dtype (bf16 for mixed precision)
    moves_left_head: bool = False   # aux head: normalized plies to game end
    deep_supervision: bool = False  # intermediate policy/value heads
    mtp: bool = False               # aux head: predict next-step value
    smolgen: bool = False           # transformer: dynamic attention bias
    rope: bool = False              # transformer: 2D rotary positions

    def _heads(self, feats, conv, dense):
        """Policy logits + scalar value from a (H,W,F) feature map (shared by the main
        head and each deep-supervision head)."""
        p = nn.relu(conv(4)(feats)).reshape(-1)
        policy = dense(self.num_actions)(p)
        v = nn.relu(conv(2)(feats)).reshape(-1)
        v = nn.relu(dense(self.cfg.channels)(v))
        value = jnp.tanh(dense(1)(v)).reshape(())
        return policy, value

    @nn.compact
    def __call__(self, obs, train: bool = False):
        x = jnp.transpose(obs, (1, 2, 0)).astype(self.dtype)  # (H,W,C)
        feats, inters = build_backbone(self.cfg, self.dtype, self.smolgen, self.rope)(
            x, train)
        conv = lambda c: nn.Conv(c, (1, 1), dtype=self.dtype, param_dtype=jnp.float32)
        dense = lambda n: nn.Dense(n, dtype=self.dtype, param_dtype=jnp.float32)

        policy_logits, value = self._heads(feats, conv, dense)

        aux = {}
        if self.moves_left_head or self.mtp:
            stem = nn.relu(conv(2)(feats)).reshape(-1)  # shared scalar-head stem

            def scalar_head(activation):
                h = nn.relu(dense(self.cfg.channels)(stem))
                return activation(dense(1)(h)).reshape(()).astype(jnp.float32)

            if self.moves_left_head:  # normalized plies to game end
                aux["moves_left"] = scalar_head(jax.nn.sigmoid)
            if self.mtp:              # predict the next step's value (foresight)
                aux["mtp_value"] = scalar_head(jnp.tanh)
        if self.deep_supervision and len(inters) >= 2:
            # tap two evenly-spaced intermediate feature maps
            idxs = (len(inters) // 3, 2 * len(inters) // 3)
            deep = []
            for i in idxs:
                pl, vl = self._heads(inters[i], conv, dense)
                deep.append((pl.astype(jnp.float32), vl.astype(jnp.float32)))
            aux["deep"] = deep

        return policy_logits.astype(jnp.float32), value.astype(jnp.float32), aux


def make_network(cfg: NetConfig, dtype=jnp.float32, moves_left_head=False,
                 deep_supervision=False, mtp=False, smolgen=False, rope=False):
    return ArimaaNet(cfg=cfg, dtype=dtype, moves_left_head=moves_left_head,
                     deep_supervision=deep_supervision, mtp=mtp,
                     smolgen=smolgen, rope=rope)
