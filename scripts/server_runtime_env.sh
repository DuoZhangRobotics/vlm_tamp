#!/usr/bin/env bash
# Source from bash before running the restored Blackwell server tools.
# This selects runtime paths only; it does not change task or numerical settings.
VLM_SERVER_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export VLM_TAMP_SIM_PYTHON="$VLM_SERVER_ROOT/.runtime/envs/vlm-tamp/bin/python"
export PATH="$VLM_SERVER_ROOT/.runtime/envs/rekep-libero/bin:$PATH"
export PYTHONPATH="$VLM_SERVER_ROOT/src:$VLM_SERVER_ROOT/scripts:$VLM_SERVER_ROOT/.runtime/mc-vamp-deps"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.9}"
export CPATH="$CUDA_HOME/targets/x86_64-linux/include${CPATH:+:$CPATH}"
export LIBRARY_PATH="$CUDA_HOME/targets/x86_64-linux/lib${LIBRARY_PATH:+:$LIBRARY_PATH}"
export LD_LIBRARY_PATH="$CUDA_HOME/targets/x86_64-linux/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-12.0}"
export MAX_JOBS="${MAX_JOBS:-4}"
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export XDG_CACHE_HOME="$VLM_SERVER_ROOT/.runtime/cache"
export MPLCONFIGDIR="$VLM_SERVER_ROOT/.runtime/cache/matplotlib"
