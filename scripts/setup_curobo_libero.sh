#!/usr/bin/env bash
# Install only into the project-local cuRobo Conda environment.
set -euo pipefail
TASK_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TASK_PREFIX="$TASK_ROOT/.runtime/envs/curobo-libero"
TASK_UPSTREAM="$TASK_ROOT/.runtime/curobo-v0.7.8"
TASK_REVISION=d64c4b005459db10c5dd867d8b30a87d5bda9bdb
TASK_CONDA="${CONDA_EXE:-/home/duo/miniforge3/bin/conda}"
if [[ ! -x "$TASK_PREFIX/bin/python" ]]; then
  "$TASK_CONDA" create -y --prefix "$TASK_PREFIX" python=3.10 pip
fi
if [[ ! -d "$TASK_UPSTREAM/.git" ]]; then
  git clone --depth 1 --branch v0.7.8 https://github.com/NVlabs/curobo.git "$TASK_UPSTREAM"
fi
[[ "$(git -C "$TASK_UPSTREAM" rev-parse HEAD)" == "$TASK_REVISION" ]]
[[ -z "$(git -C "$TASK_UPSTREAM" status --short --untracked-files=no)" ]]
"$TASK_PREFIX/bin/python" -m pip install -r "$TASK_ROOT/requirements-curobo-libero.txt"
TASK_CUDA="${CUDA_HOME:-/usr/local/cuda}"
[[ -x "$TASK_CUDA/bin/nvcc" ]]
CUDA_HOME="$TASK_CUDA" TORCH_CUDA_ARCH_LIST=8.9 MAX_JOBS=2 \
  "$TASK_PREFIX/bin/python" -m pip install -e "$TASK_UPSTREAM" --no-build-isolation --no-deps
