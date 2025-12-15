# =============================================================================
# Adversarial Patches Worker
# =============================================================================
# Worker container for adversarial patch generation and evaluation.
# Includes FiftyOne for visualization.
#
# Build: docker build -t mlops-adv-patches:latest -f projects/adversarial-patches/Dockerfile .
# Run:   docker run mlops-adv-patches:latest python generate_patch.py
# =============================================================================

ARG BASE_IMAGE=mlops-torch-cpu:latest
FROM ${BASE_IMAGE}

LABEL maintainer="MLOps Team"
LABEL description="Adversarial patch generation worker"
LABEL version="1.0.0"
LABEL project="adversarial-patches"

# Install project dependencies
RUN pip install \
    fiftyone>=0.23.0 \
    wandb>=0.16.0 \
    opencv-python-headless>=4.8.0

# Create project directories
RUN mkdir -p /workspace/projects/adversarial-patches \
    /workspace/outputs/adversarial-patches \
    /workspace/data/adversarial \
    && chown -R mlops:mlops /workspace

# Copy project files
COPY --chown=mlops:mlops config.py /workspace/
COPY --chown=mlops:mlops data_transfer/ /workspace/data_transfer/
COPY --chown=mlops:mlops projects/adversarial-patches/ /workspace/projects/adversarial-patches/

USER mlops
WORKDIR /workspace

# Environment
ENV WANDB_MODE=offline
ENV WANDB_PROJECT=adversarial-patches
ENV WANDB_DIR=/workspace/outputs/wandb
ENV FIFTYONE_DATABASE_DIR=/workspace/data/fiftyone
ENV COMPUTE_DEVICE=cpu

# Default command
CMD ["python", "projects/adversarial-patches/generate_patch.py"]
