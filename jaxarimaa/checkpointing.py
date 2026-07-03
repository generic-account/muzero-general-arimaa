"""Preemption-safe, multi-host training checkpoints via Orbax.

Saves the FULL training state (params + optimizer state + step) so a run can resume
exactly after a preemption — essential for long TPU-pod runs. Orbax handles sharded,
multi-host arrays and cross-process coordination automatically (every process calls
save/restore). Checkpoints rotate (`max_to_keep`) and save every `interval` iters.

This is separate from `checkpoint.py`, which writes the small pickled (params + config)
artifact used by the AEI inference bridge — that stays as the portable single-host export.
"""

import os

import orbax.checkpoint as ocp


class CheckpointManager:
    def __init__(self, directory, interval, max_keep=3):
        options = ocp.CheckpointManagerOptions(
            save_interval_steps=interval, max_to_keep=max_keep, create=True)
        # Pass URIs (gs://, s3://) through unchanged; only local paths get abspath.
        # For spot, use a gs:// dir so checkpoints survive the preempted VM's disk.
        d = directory if "://" in str(directory) else os.path.abspath(directory)
        self.mngr = ocp.CheckpointManager(d, options=options)

    def maybe_restore(self, state_template):
        """Restore the latest checkpoint into `state_template` (which supplies the
        target structure + shardings). Returns (state, next_iteration)."""
        step = self.mngr.latest_step()
        if step is None:
            return state_template, 0
        state = self.mngr.restore(step, args=ocp.args.StandardRestore(state_template))
        return state, step + 1

    def save(self, step, state):
        """Save at `step`; Orbax no-ops unless it's a save-interval step."""
        self.mngr.save(step, args=ocp.args.StandardSave(state))

    def close(self):
        self.mngr.wait_until_finished()
        self.mngr.close()
