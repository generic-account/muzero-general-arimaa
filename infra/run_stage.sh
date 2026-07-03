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
  gsutil -m -q rsync -r "results/jaxarimaa/${RUN}_tb" "$BUCKET/runs/$RUN/tb" 2>/dev/null || true
  gsutil -q cp "results/jaxarimaa/$RUN.pkl" "$BUCKET/runs/$RUN/model.pkl" 2>/dev/null || true
}

( while true; do sleep 300; sync_artifacts; done ) &
SYNC_PID=$!
trap 'kill $SYNC_PID 2>/dev/null' EXIT

while true; do
  rm -f /tmp/libtpu_lockfile   # stale lock from a killed process blocks TPU init
  PYTHONPATH=. "$PY" -u "$SCRIPT" "$RUN" "$ITERS"
  code=$?
  sync_artifacts
  if [ $code -eq 0 ]; then
    echo "[supervisor] run complete"
    break
  fi
  echo "[supervisor] training died (exit $code); relaunching in 30s (auto-resume)"
  sleep 30
done
