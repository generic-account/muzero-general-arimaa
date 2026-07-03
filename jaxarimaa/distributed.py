"""Data-parallel sharding helpers (single-host now; scales to multi-host slices).

We use JAX's modern jit + NamedSharding (GSPMD) rather than pmap. A 1-D device
mesh named 'data' shards the self-play/training batch across devices; params are
replicated. On 1 device this is a no-op; on a TPU/GPU slice the SAME code shards
across all chips. For multi-host slices, call jax.distributed.initialize() at
process start (not needed single-host) and build the mesh from jax.devices().

Key rule: batch leading dim must be divisible by mesh size.
"""

import jax
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P


def init_distributed(enabled: bool):
    """Initialise the multi-host runtime (a no-op single-host).

    Call once at process start, BEFORE touching devices, only when running a
    multi-host slice. On a single host leave `enabled=False`: single-controller JAX
    already sees all local devices, so this costs the small case nothing.
    Coordinator address / process id / count come from the launcher's env vars.
    """
    if enabled:
        jax.distributed.initialize()
    return jax.process_count()


def enable_compilation_cache(cache_dir):
    """Persist XLA compilations to `cache_dir` (local path or gs://). After a spot
    preemption the restart recompiles as a CACHE HIT instead of from scratch — the key
    to cheap restarts. Must be called before the first jit compiles (i.e., at startup).
    """
    if cache_dir:
        jax.config.update("jax_compilation_cache_dir", cache_dir)
        # Cache everything (defaults skip fast/small compiles).
        jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.0)
        jax.config.update("jax_persistent_cache_min_entry_size_bytes", 0)


def make_mesh():
    """1-D data-parallel mesh over all (global, after init_distributed) devices."""
    return Mesh(jax.devices(), ("data",))


def data_sharding(mesh):
    """Shard an array along its leading (batch) axis across the data mesh."""
    return NamedSharding(mesh, P("data"))


def replicated(mesh):
    """Replicate an array (e.g. params/opt-state) across all devices."""
    return NamedSharding(mesh, P())


def shard_batch(mesh, batch):
    """Place a pytree batch with its leading axis sharded across devices."""
    sh = data_sharding(mesh)
    return jax.tree_util.tree_map(lambda x: jax.device_put(x, sh), batch)


def replicate_tree(mesh, tree):
    sh = replicated(mesh)
    return jax.tree_util.tree_map(lambda x: jax.device_put(x, sh), tree)
