#!/usr/bin/env bash
# Idempotent run doctor: ensure <run> is training on <node>, resurrecting the
# spot VM from GCS state if it was preempted/deleted. Safe to run on a timer.
#
#   bash infra/resurrect.sh <run> <config-script> <iters> [qr-name] [node-name]
#
# Requires (staged by the launcher): $BUCKET/runs/<run>/code.tar.gz, corpus/,
# pretrained+regrounded pkls, and (after first sync) runs/<run>/ckpt.
# Exit 0 = healthy or resurrected; 2 = run finished (sentinel in GCS);
# 3 = gave up (RUN_FAILED or capacity).
set -uo pipefail

RUN=${1:?usage: resurrect.sh <run> <config> <iters> [qr] [node]}
CONFIG=${2:?}; ITERS=${3:?}
QR=${4:-${RUN}-q}; NODE=${5:-arimaa-$RUN}
ZONE=${ZONE:-us-west4-a}
BUCKET=${BUCKET:-gs://arimaa-tpu-2026-artifacts}
GC=${GCLOUD:-$HOME/Downloads/google-cloud-sdk/bin/gcloud}
GS=${GSUTIL:-$HOME/Downloads/google-cloud-sdk/bin/gsutil}
SSH() { $GC compute tpus tpu-vm ssh "$NODE" --zone="$ZONE" --command="$1" 2>/dev/null; }

# 0. Already finished or failed? (janitor uploads the final log; run_stage
#    drops RUN_FAILED on deterministic crash loops)
if $GS -q stat "$BUCKET/runs/$RUN/FINISHED" 2>/dev/null; then
  echo "[resurrect] run already FINISHED"; exit 2; fi
if $GS -q stat "$BUCKET/runs/$RUN/RUN_FAILED" 2>/dev/null; then
  echo "[resurrect] run marked FAILED — human needed"; exit 3; fi

# 1. Healthy? QR ACTIVE and the supervisor alive on the VM.
STATE=$($GC compute tpus queued-resources describe "$QR" --zone="$ZONE" \
        --format="value(state.state)" 2>/dev/null || true)
if [ "$STATE" = "ACTIVE" ]; then
  ALIVE=$(SSH 'pgrep -f "run_stage[.]sh" >/dev/null && echo YES || echo NO' | tail -1)
  if [ "$ALIVE" = "YES" ]; then echo "[resurrect] healthy"; exit 0; fi
  echo "[resurrect] VM up but supervisor dead — relaunching in place"
else
  echo "[resurrect] QR state='$STATE' — recreating"
  $GC compute tpus queued-resources delete "$QR" --zone="$ZONE" --force --quiet 2>/dev/null || true
  sleep 30
  $GC compute tpus queued-resources create "$QR" --node-id="$NODE" --zone="$ZONE" \
    --accelerator-type=v5litepod-4 --runtime-version=tpu-ubuntu2204-base --spot || exit 3
  for i in $(seq 1 60); do
    S=$($GC compute tpus queued-resources describe "$QR" --zone="$ZONE" \
        --format="value(state.state)" 2>/dev/null)
    [ "$S" = "ACTIVE" ] && break; sleep 20
  done
  [ "$S" = "ACTIVE" ] || { echo "[resurrect] no capacity"; exit 3; }
fi

# 2. Bootstrap if fresh (marker survives in-place relaunches).
BOOT=$(SSH '[ -f ~/BOOTSTRAP_DONE ] && echo YES || echo NO' | tail -1)
if [ "$BOOT" != "YES" ]; then
  echo "[resurrect] bootstrapping"
  SSH 'sudo systemctl stop unattended-upgrades 2>/dev/null; while sudo fuser /var/lib/dpkg/lock-frontend >/dev/null 2>&1; do sleep 5; done; nohup bash -c "sudo apt-get -qq update && sudo apt-get -qq install -y python3-venv git && python3 -m venv ~/venv && ~/venv/bin/pip -q install --upgrade pip && ~/venv/bin/pip -q install \"jax[tpu]\" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html && ~/venv/bin/pip -q install flax optax chex mctx orbax-checkpoint==0.11.16 numpy gcsfs tensorboardX && touch ~/BOOTSTRAP_DONE" > ~/bootstrap.log 2>&1 & echo started'
  for i in $(seq 1 45); do
    B=$(SSH '[ -f ~/BOOTSTRAP_DONE ] && echo YES || echo NO' | tail -1)
    [ "$B" = "YES" ] && break; sleep 20
  done
  [ "$B" = "YES" ] || { echo "[resurrect] bootstrap failed"; exit 3; }
fi

# 3. Stage code + data from GCS (idempotent).
SSH "set -e
mkdir -p ~/muzero-general-arimaa
gsutil -q cp $BUCKET/runs/$RUN/code.tar.gz ~/code.tar.gz
tar xzf ~/code.tar.gz -C ~/muzero-general-arimaa
cd ~/muzero-general-arimaa
mkdir -p results/archive_ds_sharp results/jaxarimaa
[ -f results/archive_ds_sharp/year2016.npz ] || gsutil -m -q cp '$BUCKET/corpus/*.npz' results/archive_ds_sharp/
[ -f results/jaxarimaa/regrounded_c256.pkl ] || gsutil -q cp $BUCKET/regrounded_c256.pkl results/jaxarimaa/regrounded_c256.pkl
[ -f results/jaxarimaa/regrounded_plus9_c256.pkl ] || gsutil -q cp $BUCKET/regrounded_plus9_c256.pkl results/jaxarimaa/regrounded_plus9_c256.pkl
[ -f results/jaxarimaa/${RUN}_init.pkl ] || gsutil -q cp $BUCKET/runs/$RUN/init.pkl results/jaxarimaa/${RUN}_init.pkl 2>/dev/null || true
if [ ! -d results/jaxarimaa/${RUN}_ckpt ]; then
  mkdir -p results/jaxarimaa/${RUN}_ckpt
  gsutil -m -q rsync -r $BUCKET/runs/$RUN/ckpt results/jaxarimaa/${RUN}_ckpt 2>/dev/null || true
fi
echo staged" | tail -1

# 4. Launch supervisor + janitor (janitor shipped as a file to avoid nesting).
TMPJ=$(mktemp /tmp/janitor_XXXX.sh)
cat > "$TMPJ" <<EOF
#!/bin/bash
DEADLINE=\$(( \$(date +%s) + 100*3600 ))
while [ ! -f \$HOME/RUN_DONE_$RUN ] && [ ! -f \$HOME/RUN_FAILED_$RUN ] && [ \$(date +%s) -lt \$DEADLINE ]; do sleep 60; done
cd \$HOME/muzero-general-arimaa
gsutil -m -q rsync -r results/jaxarimaa/${RUN}_tb $BUCKET/runs/$RUN/tb 2>/dev/null
gsutil -q cp results/jaxarimaa/$RUN.pkl $BUCKET/runs/$RUN/model.pkl 2>/dev/null
gsutil -m -q rsync -r -x ".*orbax-checkpoint-tmp.*" results/jaxarimaa/${RUN}_ckpt $BUCKET/runs/$RUN/ckpt 2>/dev/null
gsutil -q cp \$HOME/$RUN.log $BUCKET/runs/$RUN/$RUN.log 2>/dev/null
if [ -f \$HOME/RUN_FAILED_$RUN ]; then gsutil -q cp \$HOME/RUN_FAILED_$RUN $BUCKET/runs/$RUN/RUN_FAILED; fi
if [ -f \$HOME/RUN_DONE_$RUN ]; then echo done | gsutil -q cp - $BUCKET/runs/$RUN/FINISHED; fi
TOK=\$(curl -s -H "Metadata-Flavor: Google" "http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token" | python3 -c "import sys,json;print(json.load(sys.stdin)['access_token'])")
curl -s -X DELETE -H "Authorization: Bearer \$TOK" "https://tpu.googleapis.com/v2/projects/arimaa-tpu-2026/locations/$ZONE/queuedResources/$QR?force=true"
EOF
# deliver via ssh+base64 (scp failed SILENTLY once -> janitorless lanes
# idled after completion; always verify delivery)
B64=$(base64 < "$TMPJ" | tr -d '\n')
SSH "echo $B64 | base64 -d > ~/janitor.sh"
JOK=$(SSH '[ -s ~/janitor.sh ] && echo YES || echo NO' | tail -1)
[ "$JOK" = "YES" ] || { echo "[resurrect] JANITOR DELIVERY FAILED"; exit 3; }
SSH "chmod +x ~/janitor.sh
rm -f /tmp/libtpu_lockfile ~/RUN_DONE_$RUN ~/RUN_FAILED_$RUN
cd ~/muzero-general-arimaa
nohup bash infra/run_stage.sh $RUN $CONFIG $ITERS > \$HOME/$RUN.log 2>&1 &
nohup bash ~/janitor.sh > ~/janitor.log 2>&1 &
echo launched" | tail -1

# 5. Verify (separate session; bracketed, no self-matchable literals).
sleep 20
OK=$(SSH 'pgrep -f "run_stage[.]sh" >/dev/null && echo UP || echo DOWN' | tail -1)
echo "[resurrect] supervisor: $OK"
[ "$OK" = "UP" ] || exit 3
echo "[resurrect] resurrected"
