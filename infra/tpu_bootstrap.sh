#!/usr/bin/env bash
# Bootstrap a fresh TPU VM for jaxarimaa training.
# Usage (on the TPU VM): bash infra/tpu_bootstrap.sh
set -euo pipefail

sudo apt-get -qq update
sudo apt-get -qq install -y python3-venv git

python3 -m venv ~/venv
~/venv/bin/pip -q install --upgrade pip
# jax[tpu] pulls the matching libtpu from Google's releases index.
~/venv/bin/pip -q install "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
~/venv/bin/pip -q install flax optax chex mctx orbax-checkpoint numpy

echo "=== JAX device check ==="
~/venv/bin/python -c "import jax; print(jax.__version__, jax.devices())"
