#!/usr/bin/env bash
# Pre-launch validation gate: run before ANY training launch (local or via a
# launch checklist on the VM). Every test here exists because its absence
# once cost money. Exits non-zero on the first failure.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-.venv/bin/python}

echo "=== 1/5 config lint (measured-safe envelope) ==="
PYTHONPATH=. "$PY" jaxarimaa/tests_config.py

echo "=== 1b/5 stage-2 mechanism tests ==="
PYTHONPATH=. "$PY" jaxarimaa/tests_stage2.py
echo "=== 1c/5 selfplay semantics tests ==="
PYTHONPATH=. "$PY" jaxarimaa/tests_selfplay.py

echo "=== 2/5 ratchet trajectory tests ==="
PYTHONPATH=. "$PY" jaxarimaa/tests_anneal.py

echo "=== 3/5 fast_search / mctx equivalence ==="
PYTHONPATH=. "$PY" jaxarimaa/tests_fast_search_v2.py
echo "=== 3b/5 compact search (v3) equivalence ==="
PYTHONPATH=. "$PY" jaxarimaa/tests_compact_search.py

echo "=== 4/5 env oracle difftest ==="
PYTHONPATH=. "$PY" -m jaxarimaa.difftest

echo "=== 5/5 supervisor script syntax ==="
bash -n infra/run_stage.sh

echo "PREFLIGHT PASSED"
