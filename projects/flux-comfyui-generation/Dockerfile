# =============================================================================
# Flux ComfyUI Generation Worker
# =============================================================================
FROM python:3.13-slim

LABEL maintainer="MLOps Team"
LABEL project="flux-comfyui-generation"

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /workspace

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl libgl1 libglib2.0-0 libsm6 libxext6 libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Create user
RUN useradd --create-home --shell /bin/bash --uid 1000 mlops \
    && mkdir -p /workspace && chown -R mlops:mlops /workspace

# Python dependencies
RUN pip install --no-cache-dir \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    torch torchvision \
    wandb Pillow numpy b2sdk python-dotenv

# Create directories
RUN mkdir -p /workspace/outputs/flux-generations /workspace/data \
    && chown -R mlops:mlops /workspace

# Copy project files
COPY --chown=mlops:mlops config.py /workspace/
COPY --chown=mlops:mlops data_transfer/ /workspace/data_transfer/
COPY --chown=mlops:mlops projects/flux-comfyui-generation/ /workspace/projects/flux-comfyui-generation/

USER mlops

ENV WANDB_MODE=offline
ENV COMPUTE_DEVICE=cpu

CMD ["python", "projects/flux-comfyui-generation/run_generation.py"]
