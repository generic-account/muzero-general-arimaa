"""On-device (HBM) replay buffer, sharded across the device mesh.

The buffer lives entirely in device memory (never host RAM). It is sharded on the
row axis across the 'data' mesh axis, and add/sample are `shard_map`ped so each
device maintains an INDEPENDENT local ring over its shard — no cross-device traffic.
All devices advance in lockstep (same #adds), so a single host-side (pos, size) in
per-device-row units describes every shard.

add/sample are compiled once; the changing (pos, size) are passed as traced scalars
so there is no per-call recompilation.
"""

import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec as P


class DeviceReplay:
    def __init__(self, mesh, capacity):
        self.mesh = mesh
        self.n = mesh.shape["data"]
        if capacity % self.n:
            raise ValueError(f"capacity {capacity} must divide mesh size {self.n}")
        self.cap_local = capacity // self.n
        self.shard = NamedSharding(mesh, P("data"))
        self.pos = 0     # per-device write cursor (0..cap_local)
        self.size = 0    # per-device filled rows (0..cap_local)
        self.data = None
        self._add_fn = None
        self._sample_fn = None
        self._m_local = None

    @property
    def total_size(self):
        return self.size * self.n

    def _alloc(self, sample):
        self.data = {
            k: jax.device_put(
                jnp.zeros((self.cap_local * self.n,) + v.shape[1:], v.dtype), self.shard)
            for k, v in sample.items()
        }

    def add(self, samples):
        m_total = next(iter(samples.values())).shape[0]
        if m_total % self.n:
            raise ValueError(f"add batch {m_total} must divide mesh size {self.n}")
        m_local = m_total // self.n
        if self.data is None:
            self._alloc(samples)
        if self._add_fn is None or self._m_local != m_local:
            self._m_local = m_local
            cap_local = self.cap_local

            def per_shard(buf, s, pos):
                idx = (pos + jnp.arange(m_local)) % cap_local
                return {k: buf[k].at[idx].set(s[k]) for k in buf}

            self._add_fn = jax.jit(jax.shard_map(
                per_shard, mesh=self.mesh,
                in_specs=(P("data"), P("data"), P()), out_specs=P("data"),
                check_vma=False))

        self.data = self._add_fn(self.data, samples, jnp.int32(self.pos))
        self.pos = (self.pos + m_local) % self.cap_local
        self.size = min(self.size + m_local, self.cap_local)

    def sample(self, rng, batch):
        if batch % self.n:
            raise ValueError(f"sample batch {batch} must divide mesh size {self.n}")
        g_local = batch // self.n
        if self._sample_fn is None:
            def per_shard(buf, rng, size):
                r = jax.random.fold_in(rng, jax.lax.axis_index("data"))
                idx = jax.random.randint(r, (g_local,), 0, size)
                return {k: buf[k][idx] for k in buf}

            self._sample_fn = jax.jit(jax.shard_map(
                per_shard, mesh=self.mesh,
                in_specs=(P("data"), P(), P()), out_specs=P("data"),
                check_vma=False))
        return self._sample_fn(self.data, rng, jnp.int32(self.size))
