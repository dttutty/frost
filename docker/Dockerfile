FROM ghcr.io/astral-sh/uv:0.10.12 AS uv

FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
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

COPY --from=uv /uv /uvx /bin/

WORKDIR /workspace/frost

COPY pyproject.toml uv.lock .python-version readme.md /workspace/frost/

RUN uv sync --frozen --no-install-project --python 3.12

COPY . /workspace/frost

RUN uv sync --frozen --python 3.12

ENV PATH=/workspace/frost/.venv/bin:${PATH}

CMD ["/bin/bash"]
