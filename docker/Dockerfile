FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    CONDA_DIR=/opt/conda \
    CUDA_LAUNCH_BLOCKING=0 \
    TORCH_USE_CUDA_DSA=0 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    TORCH_CUDNN_BENCHMARK=1 \
    TORCH_NCCL_ASYNC_ERROR_HANDLING=1 \
    TORCH_NCCL_BLOCKING_WAIT=0 \
    NCCL_IB_DISABLE=0 \
    NCCL_P2P_DISABLE=0 \
    NCCL_DEBUG=WARN \
    TORCH_DISTRIBUTED_DEBUG=OFF \
    HYDRA_FULL_ERROR=0 \
    LOG_LEVEL=WARNING \
    OMP_NUM_THREADS=8 \
    MKL_NUM_THREADS=8 \
    TOKENIZERS_PARALLELISM=false \
    PYTHONOPTIMIZE=2

ENV PATH=${CONDA_DIR}/bin:${PATH}

RUN apt-get update && apt-get install -y --no-install-recommends \
    bash \
    build-essential \
    ca-certificates \
    cmake \
    curl \
    git \
    libomp-dev \
    ninja-build \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

RUN curl -fsSL -o /tmp/miniforge.sh \
    https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh \
    && bash /tmp/miniforge.sh -b -p "${CONDA_DIR}" \
    && rm -f /tmp/miniforge.sh \
    && conda clean -afy

RUN conda create -y -n frost python=3.12 && conda clean -afy

ENV CONDA_DEFAULT_ENV=frost
ENV PATH=${CONDA_DIR}/envs/frost/bin:${CONDA_DIR}/bin:${PATH}

WORKDIR /workspace/frost

COPY . /workspace/frost

RUN python -m pip install --upgrade pip \
    && python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124 \
    && python -m pip install tbb pytest pytest-cov ruff mypy build \
    && python -m pip install -e .

CMD ["/bin/bash"]
