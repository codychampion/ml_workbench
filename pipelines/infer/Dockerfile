# =============================================================================
# Pipeline Stage 4: Model Inference
# =============================================================================
# Running trained models for generation and prediction.
#
# Usage:
#   docker-compose run --rm infer python -m pipelines.infer.run_generation --help
# =============================================================================
FROM python:3.12-slim

LABEL maintainer="MLOps Team"
LABEL pipeline.stage="4-infer"

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl libgl1 libglib2.0-0 libsm6 libxext6 libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast package management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

RUN useradd --create-home --shell /bin/bash --uid 1000 mlops \
    && mkdir -p /workspace && chown -R mlops:mlops /workspace

# Inference dependencies - use PyTorch nightly with CUDA 12.8 for RTX 5090 sm_120 support
RUN uv pip install --system --no-cache --pre \
    --extra-index-url https://download.pytorch.org/whl/nightly/cu128 \
    torch torchvision \
    transformers>=4.36.0 \
    diffusers>=0.30.0 \
    accelerate \
    peft>=0.7.0 \
    fiftyone \
    aim \
    hydra-core>=1.3.0 \
    omegaconf>=2.3.0 \
    boto3 \
    hvac \
    Pillow numpy tqdm requests \
    python-dotenv pyyaml \
    opencv-python-headless \
    ultralytics \
    av \
    imageio[ffmpeg] \
    imageio-ffmpeg

# Install CLIP for adversarial pipeline
RUN uv pip install --system --no-cache \
    git+https://github.com/openai/CLIP.git

RUN mkdir -p /workspace/models /workspace/outputs /workspace/data \
    && chown -R mlops:mlops /workspace

COPY --chown=mlops:mlops config.py /workspace/
COPY --chown=mlops:mlops data_transfer/ /workspace/data_transfer/
COPY --chown=mlops:mlops pipelines/ /workspace/pipelines/
COPY --chown=mlops:mlops conf/ /workspace/conf/
COPY --chown=mlops:mlops utils/ /workspace/utils/

USER mlops

ENV HF_HOME=/workspace/.cache/huggingface
ENV AIM_REPO=/workspace/outputs/aim
ENV COMPUTE_DEVICE=cpu
ENV PREFECT_API_URL=

CMD ["python", "-m", "pipelines.infer.run_generation", "--help"]
