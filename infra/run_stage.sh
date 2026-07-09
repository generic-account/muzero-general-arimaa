#!/usr/bin/env bash
# Supervised staged-training runner for a spot TPU VM (crash/preemption tolerant).
#
# Usage: bash infra/run_stage.sh <run-name> <config-script> [iterations]
# e.g.:  bash infra/run_stage.sh stage_a configs/stage_a.py 400
#
# - Relaunches on crash; training auto-resumes from Orbax in GCS (see the
#   config script's ckpt_dir) and restarts fast via the GCS compile cache.
# - Removes the stale libtpu lockfile a killed process leaves behind.
# - Continuously rsyncs TensorBoard events + the portable weights pickle to GCS,
#   so progress and current strength can be monitored from anywhere mid-run.
set -uo pipefail

RUN=${1:?usage: run_stage.sh <run-name> <config-script> [iterations]}
SCRIPT=${2:?usage: run_stage.sh <run-name> <config-script> [iterations]}
ITERS=${3:-400}
BUCKET=${BUCKET:-gs://arimaa-tpu-2026-artifacts}
PY=${PY:-$HOME/venv/bin/python}

cd "$(dirname "$0")/.."

sync_artifacts() {
  gsutil -m -q rsync -r "results/jaxarimaa/${RUN}_tb" "$BUCKET/runs/$RUN/tb" || true
  gsutil -q cp "results/jaxarimaa/$RUN.pkl" "$BUCKET/runs/$RUN/model.pkl" 2>/dev/null || true
  # Orbax checkpoints are written locally (multi-device async saves to gs://
  # time out); mirror them to GCS for preemption durability. NO -d and exclude
  # Orbax's atomic-write tmp dirs: a delete-mirroring rsync racing a mid-write
  # checkpoint could leave GCS with a half-uploaded latest AND older steps
  # already deleted (stale steps in GCS cost pennies; a corrupt latest costs
  # a crash-looped run).
  gsutil -m -q rsync -r -x '.*orbax-checkpoint-tmp.*' \
    "results/jaxarimaa/${RUN}_ckpt" "$BUCKET/runs/$RUN/ckpt" || true
}

# Fresh VM after a preemption: restore the latest mirrored checkpoints first,
# and the warm-start/KL-anchor init the config's INIT_PARAMS points at (loaded
# on EVERY relaunch -> missing file = eternal crash loop on a fresh VM).
if [ ! -d "results/jaxarimaa/${RUN}_ckpt" ]; then
  mkdir -p "results/jaxarimaa/${RUN}_ckpt"
  gsutil -m -q rsync -r "$BUCKET/runs/$RUN/ckpt" "results/jaxarimaa/${RUN}_ckpt" 2>/dev/null || true
fi
if [ ! -f "results/jaxarimaa/regrounded_c256.pkl" ]; then
  gsutil -q cp "$BUCKET/regrounded_c256.pkl" "results/jaxarimaa/regrounded_c256.pkl" 2>/dev/null || true
fi
if [ ! -f "results/jaxarimaa/regrounded_plus9_c256.pkl" ]; then
  gsutil -q cp "$BUCKET/regrounded_plus9_c256.pkl" "results/jaxarimaa/regrounded_plus9_c256.pkl" 2>/dev/null || true
fi

rm -f "$HOME/RUN_DONE_$RUN"   # completion sentinel (janitors wait on THIS, not
                              # pgrep: the sync subshell shares this script's
                              # cmdline and once blocked a teardown as a phantom)
( while true; do sleep 300; sync_artifacts; done ) &
SYNC_PID=$!
trap 'kill $SYNC_PID 2>/dev/null' EXIT

FAST_FAILS=0
while true; do
  rm -f /tmp/libtpu_lockfile   # stale lock from a killed process blocks TPU init
  T0=$SECONDS
  PYTHONPATH=. "$PY" -u "$SCRIPT" "$RUN" "$ITERS"
  code=$?
  sync_artifacts
  if [ $code -eq 0 ]; then
    echo "[supervisor] run complete"
    break
  fi
  # Fast-crash guard: a preempted/OOM'd run dies after MINUTES of work; a run
  # that dies within 5 min repeatedly is a deterministic failure (bad ckpt,
  # missing file) — relaunching forever would burn the whole budget doing
  # nothing on an unattended spot run.
  if [ $((SECONDS - T0)) -lt 300 ]; then
    FAST_FAILS=$((FAST_FAILS + 1))
    if [ $FAST_FAILS -ge 5 ]; then
      echo "[supervisor] $FAST_FAILS consecutive fast crashes; giving up"
      touch "$HOME/RUN_FAILED_$RUN"
      break
    fi
  else
    FAST_FAILS=0
  fi
  echo "[supervisor] training died (exit $code); relaunching in 30s (auto-resume)"
  sleep 30
done
kill $SYNC_PID 2>/dev/null; wait $SYNC_PID 2>/dev/null
touch "$HOME/RUN_DONE_$RUN"
