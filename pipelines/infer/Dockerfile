# =============================================================================
# Pipeline Stage 4: Model Inference
# =============================================================================
# Running trained models for generation and prediction.
#
# Usage:
#   docker-compose run --rm infer python -m pipelines.infer.run_generation --help
# =============================================================================
FROM python:3.13-slim

LABEL maintainer="MLOps Team"
LABEL pipeline.stage="4-infer"

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl libgl1 libglib2.0-0 libsm6 libxext6 libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

RUN useradd --create-home --shell /bin/bash --uid 1000 mlops \
    && mkdir -p /workspace && chown -R mlops:mlops /workspace

# Inference dependencies
RUN pip install --no-cache-dir \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    torch torchvision \
    transformers>=4.36.0 \
    diffusers \
    accelerate \
    peft>=0.7.0 \
    fiftyone \
    wandb \
    Pillow numpy tqdm requests \
    python-dotenv pyyaml \
    opencv-python-headless

RUN mkdir -p /workspace/models /workspace/outputs /workspace/data \
    && chown -R mlops:mlops /workspace

COPY --chown=mlops:mlops config.py /workspace/
COPY --chown=mlops:mlops data_transfer/ /workspace/data_transfer/
COPY --chown=mlops:mlops pipelines/ /workspace/pipelines/

USER mlops

ENV HF_HOME=/workspace/.cache/huggingface
ENV WANDB_MODE=offline
ENV COMPUTE_DEVICE=cpu

CMD ["python", "-m", "pipelines.infer.run_generation", "--help"]
