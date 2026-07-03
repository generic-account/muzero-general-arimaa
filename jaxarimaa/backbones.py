"""Swappable, parameterized backbones (Flax linen).

A backbone maps a channels-last observation (H, W, C_in) -> (final_features (H,W,F),
intermediate_features [list of (H,W,F)]). The intermediates feed optional
deep-supervision heads; when unused they are just references XLA prunes.

  * "resnet"      — conv ResNet with optional Squeeze-Excitation (AlphaZero-style)
  * "transformer" — 64 squares as tokens + a custom multi-head attention that supports
                    two LeelaZero-inspired options: Smolgen (dynamic position-dependent
                    attention bias) and 2D RoPE (rotary positions using file/rank).

Compute dtype (bf16 mixed precision) via `dtype`; params stay fp32 (`_PARAM_DTYPE`).
LayerNorm/positional math kept in fp32 for stability.
"""

import flax.linen as nn
import jax
import jax.numpy as jnp

_PARAM_DTYPE = jnp.float32
BACKBONE_REGISTRY: dict[str, type] = {}


def register_backbone(name):
    def deco(cls):
        BACKBONE_REGISTRY[name] = cls
        return cls
    return deco


# ---------------------------------------------------------------------------
# ResNet (+ optional Squeeze-Excitation)
# ---------------------------------------------------------------------------
class _SE(nn.Module):
    channels: int
    ratio: int = 4
    dtype: object = jnp.float32

    @nn.compact
    def __call__(self, x):  # x: (H,W,C)
        s = jnp.mean(x, axis=(0, 1))
        s = nn.relu(nn.Dense(self.channels // self.ratio, dtype=self.dtype,
                             param_dtype=_PARAM_DTYPE)(s))
        s = nn.sigmoid(nn.Dense(self.channels, dtype=self.dtype,
                                param_dtype=_PARAM_DTYPE)(s))
        return x * s[None, None, :]


@register_backbone("resnet")
class ResNetBackbone(nn.Module):
    channels: int = 64
    blocks: int = 6
    use_se: bool = True
    dtype: object = jnp.float32

    @nn.compact
    def __call__(self, x, train: bool = False):
        conv = lambda c: nn.Conv(c, (3, 3), padding="SAME", dtype=self.dtype,
                                 param_dtype=_PARAM_DTYPE)
        x = nn.relu(conv(self.channels)(x))
        inters = []
        for _ in range(self.blocks):
            y = nn.relu(conv(self.channels)(x))
            y = conv(self.channels)(y)
            if self.use_se:
                y = _SE(self.channels, dtype=self.dtype)(y)
            x = nn.relu(x + y)
            inters.append(x)
        return x, inters  # (H,W,channels), [per-block (H,W,channels)]


# ---------------------------------------------------------------------------
# Transformer with LeelaZero-inspired options (Smolgen, 2D RoPE)
# ---------------------------------------------------------------------------
def _rope_half(x, pos, m):
    """Rotate the first `m` dims of x[...,:] (m even) by angle pos*inv_freq. x:[T,h,dh]."""
    half = m // 2
    inv = 1.0 / (10000.0 ** (jnp.arange(half, dtype=jnp.float32) / half))
    ang = pos[:, None].astype(jnp.float32) * inv[None, :]        # [T, half]
    cos = jnp.cos(ang)[:, None, :]
    sin = jnp.sin(ang)[:, None, :]
    x1, x2, rest = x[..., :half], x[..., half:m], x[..., m:]
    rot = jnp.concatenate([x1 * cos - x2 * sin, x1 * sin + x2 * cos], axis=-1)
    return jnp.concatenate([rot, rest], axis=-1)


def _rope_2d(x):
    """2D rotary on [T=64, h, dh]: file (x) rotates the first dh/2 dims, rank (y) the rest."""
    T, _, dh = x.shape
    xs = (jnp.arange(T) % 8).astype(jnp.float32)
    ys = (jnp.arange(T) // 8).astype(jnp.float32)
    half = (dh // 2) - ((dh // 2) % 2)  # even sub-block per axis
    x = _rope_half(x, xs, half)
    # rotate the second half using ranks by viewing it as the leading block
    x2 = _rope_half(x[..., half:], ys, half)
    return jnp.concatenate([x[..., :half], x2], axis=-1)


class _MHA(nn.Module):
    dim: int
    num_heads: int
    dtype: object = jnp.float32
    smolgen: bool = False
    rope: bool = False

    @nn.compact
    def __call__(self, x):  # x: [T, dim]
        T, h, dh = x.shape[0], self.num_heads, self.dim // self.num_heads
        dense = lambda: nn.Dense(self.dim, dtype=self.dtype, param_dtype=_PARAM_DTYPE)
        q = dense()(x).reshape(T, h, dh)
        k = dense()(x).reshape(T, h, dh)
        v = dense()(x).reshape(T, h, dh)
        if self.rope:
            q, k = _rope_2d(q), _rope_2d(k)
        logits = jnp.einsum("thd,shd->hts", q, k) / jnp.sqrt(dh).astype(self.dtype)
        if self.smolgen:
            logits = logits + self._smolgen_bias(x, h, T)
        attn = jax.nn.softmax(logits.astype(jnp.float32), axis=-1).astype(self.dtype)
        out = jnp.einsum("hts,shd->thd", attn, v).reshape(T, self.dim)
        return dense()(out)

    def _smolgen_bias(self, x, h, T):
        """Global position summary -> per-head [h,T,T] additive attention bias
        (a compact take on LeelaZero's Smolgen)."""
        c = nn.Dense(8, dtype=self.dtype, param_dtype=_PARAM_DTYPE)(x)  # compress tokens
        g = nn.relu(nn.Dense(256, dtype=self.dtype, param_dtype=_PARAM_DTYPE)(c.reshape(-1)))
        b = nn.Dense(h * T * T, dtype=self.dtype, param_dtype=_PARAM_DTYPE)(g)
        return b.reshape(h, T, T)


class _EncoderBlock(nn.Module):
    dim: int
    num_heads: int
    mlp_ratio: int
    dtype: object = jnp.float32
    smolgen: bool = False
    rope: bool = False

    @nn.compact
    def __call__(self, x):
        h = _MHA(self.dim, self.num_heads, self.dtype, self.smolgen, self.rope)(
            nn.LayerNorm()(x))
        x = x + h
        y = nn.LayerNorm()(x)
        y = nn.Dense(self.dim * self.mlp_ratio, dtype=self.dtype,
                     param_dtype=_PARAM_DTYPE)(y)
        y = nn.gelu(y)
        y = nn.Dense(self.dim, dtype=self.dtype, param_dtype=_PARAM_DTYPE)(y)
        return x + y


@register_backbone("transformer")
class TransformerBackbone(nn.Module):
    channels: int = 64
    blocks: int = 6
    num_heads: int = 4
    mlp_ratio: int = 4
    dtype: object = jnp.float32
    smolgen: bool = False
    rope: bool = False

    @nn.compact
    def __call__(self, x, train: bool = False):
        H, W, C = x.shape
        tok = nn.Dense(self.channels, dtype=self.dtype, param_dtype=_PARAM_DTYPE)(
            x.reshape(H * W, C))
        # Learned positional embedding still helps even with RoPE (keeps a global anchor).
        pos = self.param("pos_emb", nn.initializers.normal(0.02),
                         (H * W, self.channels))
        h = tok + pos.astype(tok.dtype)
        inters = []
        for _ in range(self.blocks):
            h = _EncoderBlock(self.channels, self.num_heads, self.mlp_ratio,
                              self.dtype, self.smolgen, self.rope)(h)
            inters.append(h.reshape(H, W, self.channels))
        h = nn.LayerNorm()(h)
        return h.reshape(H, W, self.channels), inters


def build_backbone(cfg, dtype=jnp.float32, smolgen=False, rope=False):
    """Instantiate the selected backbone, passing only the config fields it declares."""
    import dataclasses

    cls = BACKBONE_REGISTRY[cfg.backbone]
    available = {"channels": cfg.channels, "blocks": cfg.blocks, "use_se": cfg.use_se,
                 "num_heads": cfg.num_heads, "mlp_ratio": cfg.mlp_ratio, "dtype": dtype,
                 "smolgen": smolgen, "rope": rope}
    valid = {f.name for f in dataclasses.fields(cls)}
    return cls(**{k: v for k, v in available.items() if k in valid})
