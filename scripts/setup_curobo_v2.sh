#!/usr/bin/env bash
# Separate environment: never upgrades the retained v1, LIBERO or GraspGen envs.
set -euo pipefail
TASK_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TASK_PREFIX="$TASK_ROOT/.runtime/envs/curobo-v2"
TASK_UPSTREAM="$TASK_ROOT/.runtime/curobo-v0.8.0"
TASK_REVISION=4ea77366ca48ee453e7df139e39fa6532af49f3b
TASK_CONDA="${CONDA_EXE:-$(command -v conda)}"
export CONDA_PKGS_DIRS="$TASK_ROOT/.runtime/conda-pkgs"
export PIP_CACHE_DIR="$TASK_ROOT/.runtime/pip-cache"
if [[ ! -x "$TASK_PREFIX/bin/python" ]]; then
  "$TASK_CONDA" create -y --prefix "$TASK_PREFIX" python=3.10 pip
fi
if [[ ! -d "$TASK_UPSTREAM/.git" ]]; then
  git clone --depth 1 --branch v0.8.0 https://github.com/NVlabs/curobo.git "$TASK_UPSTREAM"
fi
[[ "$(git -C "$TASK_UPSTREAM" rev-parse HEAD)" == "$TASK_REVISION" ]]
[[ -z "$(git -C "$TASK_UPSTREAM" status --short --untracked-files=no)" ]]
"$TASK_PREFIX/bin/python" -m pip install -r "$TASK_ROOT/requirements-curobo-v2.txt"
"$TASK_PREFIX/bin/python" -m pip install -e "$TASK_UPSTREAM[cu12]" --no-build-isolation
"$TASK_PREFIX/bin/python" -m pip check
