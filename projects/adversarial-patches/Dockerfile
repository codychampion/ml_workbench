# =============================================================================
# Adversarial Patches Worker
# =============================================================================
FROM python:3.13-slim

LABEL maintainer="MLOps Team"
LABEL project="adversarial-patches"

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
    wandb fiftyone Pillow numpy b2sdk python-dotenv opencv-python-headless

# Create directories
RUN mkdir -p /workspace/outputs/adversarial-patches /workspace/data \
    && chown -R mlops:mlops /workspace

# Copy project files
COPY --chown=mlops:mlops config.py /workspace/
COPY --chown=mlops:mlops data_transfer/ /workspace/data_transfer/
COPY --chown=mlops:mlops projects/adversarial-patches/ /workspace/projects/adversarial-patches/

USER mlops

ENV WANDB_MODE=offline
ENV COMPUTE_DEVICE=cpu

CMD ["python", "projects/adversarial-patches/generate_patch.py"]
