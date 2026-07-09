#!/usr/bin/env bash
# One-shot run status from the laptop, no VM access needed (reads the GCS
# mirror, which run_stage syncs every 300s). Works during preemptions.
#
#   bash infra/status.sh [run=stage_a]
set -uo pipefail
RUN=${1:-stage_a}
ZONE=${ZONE:-us-west4-a}
BUCKET=${BUCKET:-gs://arimaa-tpu-2026-artifacts}
GC=${GCLOUD:-$HOME/Downloads/google-cloud-sdk/bin/gcloud}
GS=${GSUTIL:-$HOME/Downloads/google-cloud-sdk/bin/gsutil}
QR=${QR:-${RUN}-q}

echo "=== $RUN ==="
STATE=$($GC compute tpus queued-resources describe "$QR" --zone="$ZONE" \
        --format="value(state.state)" 2>/dev/null || echo "none")
echo "VM: ${STATE:-none}"
if $GS -q stat "$BUCKET/runs/$RUN/FINISHED" 2>/dev/null; then echo "run: FINISHED"; fi
if $GS -q stat "$BUCKET/runs/$RUN/RUN_FAILED" 2>/dev/null; then echo "run: FAILED (needs human)"; fi

TMP=$(mktemp /tmp/status_XXXXXXXX.log)
if $GS -q cp "$BUCKET/runs/$RUN/$RUN.log" "$TMP" 2>/dev/null; then
  AGE=$($GS ls -l "$BUCKET/runs/$RUN/$RUN.log" 2>/dev/null | awk '{print $2}' | head -1)
  ITERS=$(grep -c "^\[iter" "$TMP" || true)
  echo "log synced: $AGE (UTC) | iterations logged: $ITERS"
  echo "--- last iteration ---"
  grep "^\[iter" "$TMP" | tail -1
  echo "--- unbiased rung readings (elo/vs_ref) ---"
  grep -E "ref: score|\[rung\]" "$TMP" | tail -6
  echo "--- recent arena / ratchet ---"
  grep -E "arena:|\[anneal\]" "$TMP" | tail -6
  echo "--- restarts / incidents ---"
  R=$(grep -c "training died" "$TMP" || true)
  P=$(grep -c "resumed from checkpoint" "$TMP" || true)
  echo "relaunches: $R | checkpoint resumes: $P"
  grep -E "fast crashes|giving up" "$TMP" | tail -2
else
  echo "(no GCS log yet — falling back to direct VM read)"
  NODE=${NODE:-arimaa-$(echo "$RUN" | tr -d '_')}
  $GC compute tpus tpu-vm ssh "$NODE" --zone="$ZONE" \
    --command="grep -E '^\[iter|arena:|ref: score|\[anneal\]|training died' ~/$RUN.log 2>/dev/null | tail -8" 2>/dev/null || echo "(VM unreachable)"
fi
rm -f "$TMP"
