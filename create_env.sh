#!/bin/bash
set -e

ENV_NAME=frost

# Create conda env with Python 3.12 (better performance than 3.10)
conda create -n $ENV_NAME python=3.12 -y

# Activate
eval "$(conda shell.bash hook)"
conda activate $ENV_NAME

# PyTorch 2.5 + CUDA 12.4 (compatible with driver 535 / CUDA 12.2)
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124

# Core dependencies
pip install \
    torchmetrics>=1.0.0 \
    hydra-core>=1.3.2 \
    omegaconf>=2.3.0 \
    "numpy>=2.0" \
    "pandas>=2.0" \
    tqdm \
    PyYAML \
    scikit-learn \
    tbb

# Dev/test
pip install pytest pytest-cov ruff

# Build tools for C++ extension (seq_sampler)
pip install scikit-build-core nanobind

# Install frost in editable mode (builds C++ extension)
cd /home/sqp17/Projects/lei_pipe
pip install -e .

echo ""
echo "Done! Activate with: conda activate $ENV_NAME"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__, torch.cuda.is_available())')"
echo "NumPy:   $(python -c 'import numpy; print(numpy.__version__)')"
