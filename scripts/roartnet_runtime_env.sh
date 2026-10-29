#!/usr/bin/env bash
# Source before perception-only commands. Does not modify other environments.
ROART_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PATH="$ROART_ROOT/.runtime/envs/roartnet/bin:$PATH"
export PYTHONPATH="$ROART_ROOT/src:$ROART_ROOT/.runtime/MinkowskiEngine"
export LD_LIBRARY_PATH="$ROART_ROOT/.runtime/envs/roartnet/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export CUDA_HOME=/usr/local/cuda-12.9
export CUPY_CACHE_DIR="$ROART_ROOT/.runtime/cache/roartnet-cupy"
export XDG_CACHE_HOME="$ROART_ROOT/.runtime/cache/roartnet"
export MPLCONFIGDIR="$ROART_ROOT/.runtime/cache/roartnet/matplotlib"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
