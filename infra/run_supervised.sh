#!/usr/bin/env bash
# Preemption/crash-tolerant training supervisor for a spot TPU VM.
#
# Relaunches training whenever the process dies (crash, OOM, transient TPU error).
# Each relaunch auto-resumes from the latest Orbax checkpoint in --ckpt-dir and
# recompiles via the persistent --compile-cache, so a restart costs seconds, not
# minutes. A VM *preemption* kills the VM itself — this loop covers the relaunch
# after you (or a queued-resource/MIG policy) recreate the VM; state lives in GCS.
#
# Usage: bash infra/run_supervised.sh <run-name> [extra train.py args...]
set -uo pipefail

RUN=${1:?usage: run_supervised.sh <run-name> [args...]}; shift || true
BUCKET=${BUCKET:-gs://arimaa-tpu-2026-artifacts}
PY=${PY:-$HOME/venv/bin/python}

cd "$(dirname "$0")/.."
while true; do
  # Stale libtpu lockfile from a killed process blocks TPU acquisition forever.
  rm -f /tmp/libtpu_lockfile
  "$PY" -u -m jaxarimaa.train \
    --ckpt-interval "${CKPT_INTERVAL:-5}" \
    --ckpt-dir "$BUCKET/runs/$RUN/ckpt" \
    --compile-cache "$BUCKET/compile-cache" \
    --out "results/jaxarimaa/$RUN.pkl" \
    "$@"
  code=$?
  if [ $code -eq 0 ]; then
    echo "[supervisor] training exited cleanly; uploading final weights"
    ~/google-cloud-sdk/bin/gsutil -q cp "results/jaxarimaa/$RUN.pkl" \
      "$BUCKET/runs/$RUN/model.pkl" 2>/dev/null || \
      gsutil -q cp "results/jaxarimaa/$RUN.pkl" "$BUCKET/runs/$RUN/model.pkl" || true
    break
  fi
  echo "[supervisor] training died (exit $code); relaunching in 30s (auto-resume)"
  sleep 30  # give the TPU driver time to release the dead process's resources
done
