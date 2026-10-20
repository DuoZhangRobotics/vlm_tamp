#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
env_name="${1:-vlm-tamp}"
if ! command -v conda >/dev/null 2>&1; then
    printf '%s\n' 'Activate your Conda shell first (source ~/miniforge3/etc/profile.d/conda.sh).'
    exit 1
fi
if ! conda run -n "$env_name" python --version >/dev/null 2>&1; then
    conda create -y -n "$env_name" python=3.10 pip
fi
mkdir -p third_party
libero_commit="8f1084e3132a39270c3a13ebe37270a43ece2a01"
if [ ! -d third_party/LIBERO/.git ]; then
    git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git third_party/LIBERO
fi
if [ "$(git -C third_party/LIBERO rev-parse HEAD)" != "$libero_commit" ]; then
    if [ -n "$(git -C third_party/LIBERO status --porcelain)" ]; then
        printf '%s\n' 'LIBERO checkout has local changes; preserve them before selecting the pinned commit.'
        exit 1
    fi
    git -C third_party/LIBERO checkout "$libero_commit"
fi
conda run -n "$env_name" python -m pip install -r requirements-sim.txt
conda run -n "$env_name" python -m pip install -e third_party/LIBERO --no-deps
conda run -n "$env_name" python -m pip install -e '.[api,test]'
conda run -n "$env_name" python -m pip check
