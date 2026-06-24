#!/bin/bash

set -e

APP_DIR=$(pwd)

# ==============================================================================
# FIX CUDA / ONNX GPU / WSL
# ==============================================================================

# Ambil lokasi site-packages dari python environment saat ini
SITE_PACKAGES=$(python3 -c "import site; print(site.getsitepackages()[0])")

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$SITE_PACKAGES/nvidia/cublas/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$SITE_PACKAGES/nvidia/cudnn/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$SITE_PACKAGES/nvidia/cuda_runtime/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$SITE_PACKAGES/nvidia/cufft/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$SITE_PACKAGES/nvidia/curand/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$SITE_PACKAGES/nvidia/cusolver/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$SITE_PACKAGES/nvidia/cusparse/lib

echo "[INFO] LD_LIBRARY_PATH configured for CUDA."

# ==============================================================================
# RUN PYTHON SCRIPT
# ==============================================================================

exec python3 object-face-detection.py
