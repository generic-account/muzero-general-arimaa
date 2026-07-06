#!/usr/bin/env bash
# Pull a run's current weights from GCS and play the local AEI ladder.
# Usage: bash infra/ladder_eval.sh <run-name> [opponent=occam] [rounds=2] [sims=32]
# Opponents: simple | occam | sharp  (see third_party/opponents.example.cfg)
set -euo pipefail

RUN=${1:?usage: ladder_eval.sh <run-name> [opponent] [rounds] [sims]}
OPP=${2:-occam}
ROUNDS=${3:-2}
SIMS=${4:-32}
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUCKET=${BUCKET:-gs://arimaa-tpu-2026-artifacts}
GS=${GSUTIL:-gsutil}
PKL="$ROOT/results/jaxarimaa/${RUN}_ladder.pkl"

$GS -q cp "$BUCKET/runs/$RUN/model.pkl" "$PKL"
echo "weights: $PKL ($(du -h "$PKL" | cut -f1))"

case "$OPP" in
  simple) OPP_CMD="$ROOT/.venv/bin/simple_engine" ;;
  occam)  OPP_CMD="$ROOT/.venv/bin/python $ROOT/third_party/sample_bots/aei_adapter.py --bot occam --exe $ROOT/third_party/sample_bots/occam/arimaa/getMove --depth 4" ;;
  sharp)  OPP_CMD="$ROOT/third_party/bot_sharp/arimaasharp/build/sharp aei" ;;
  *) echo "unknown opponent $OPP"; exit 1 ;;
esac

CFG=$(mktemp /tmp/ladder_XXXX.cfg)
cat > "$CFG" <<EOF
[global]
rounds = $ROUNDS
loglevel = WARN
timecontrol = 20s/60s/100/0/10m
bots = candidate $OPP

[candidate]
cmdline = $ROOT/.venv/bin/python $ROOT/arimaa_aei_engine.py --policy jax --checkpoint $PKL --simulations $SIMS

[$OPP]
cmdline = $OPP_CMD
EOF

"$ROOT/.venv/bin/roundrobin" --config "$CFG" 2>&1 | grep -aE "beat|wins and|timeouts" | tail -8
