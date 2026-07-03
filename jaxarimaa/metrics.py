"""Lightweight metric logging for training.

Writes scalars to TensorBoard (via flax.metrics.tensorboard — no TF/torch import)
and, optionally, Weights & Biases. Only the CHIEF process (process_index 0) writes,
so multi-host slices don't produce N duplicate/conflicting logs.

Monitoring real instances (see module notes / README):
  * Local / single VM : point `--logdir` at a local dir, run `tensorboard --logdir`,
    and SSH port-forward 6006 to view from your laptop.
  * Cloud TPU / pods  : set `--logdir gs://<bucket>/<run>`; TensorBoard reads event
    files straight from GCS, decoupling viewing from the (possibly ephemeral) hosts.
  * Anywhere          : `--wandb` streams metrics to the W&B cloud dashboard.
Loss scalars are already GSPMD global-means; throughput counts are global batch, so the
chief can report cluster-wide numbers with no extra cross-host communication.
"""

import jax


def _open_tensorboard(logdir):
    """Return (scalar_fn, close_fn) or (None, None). Prefer the flax/TF writer
    (supports gs:// for cloud/TPU); fall back to torch's writer (TF-free, local)."""
    try:
        from flax.metrics import tensorboard as ftb
        w = ftb.SummaryWriter(logdir)
        def scalar(k, v, s):
            w.scalar(k, v, s)
        return scalar, w.close
    except Exception:
        pass
    try:
        from torch.utils.tensorboard import SummaryWriter
        w = SummaryWriter(logdir)
        def scalar(k, v, s):
            w.add_scalar(k, v, s)
        return scalar, w.close
    except Exception:
        pass
    try:  # TF-free and torch-free (what TPU VMs get)
        from tensorboardX import SummaryWriter
        w = SummaryWriter(logdir)
        def scalar(k, v, s):
            w.add_scalar(k, v, s)
        return scalar, w.close
    except Exception as exc:  # pragma: no cover - optional dep
        print(f"[metrics] no TensorBoard backend ({exc}); stdout only")
        return None, None


class Logger:
    def __init__(self, logdir=None, use_wandb=False, run_name="jaxarimaa", config=None):
        self.chief = jax.process_index() == 0
        self._tb_scalar = self._tb_close = None
        self.wandb = None
        if not self.chief:
            return
        if logdir:
            self._tb_scalar, self._tb_close = _open_tensorboard(logdir)
        if use_wandb:
            import wandb
            wandb.init(project="jaxarimaa", name=run_name, config=config)
            self.wandb = wandb

    def write(self, step, metrics: dict):
        if not self.chief:
            return
        if self._tb_scalar:
            for k, v in metrics.items():
                self._tb_scalar(k, float(v), step)
        if self.wandb:
            self.wandb.log({k: float(v) for k, v in metrics.items()}, step=step)

    def close(self):
        if self._tb_close:
            self._tb_close()
        if self.wandb:
            self.wandb.finish()
