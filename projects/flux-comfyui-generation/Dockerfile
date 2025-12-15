# =============================================================================
# Flux ComfyUI Generation Worker
# =============================================================================
# Worker container for running image generation jobs.
# Can be run as a batch job or as a persistent worker.
#
# Build: docker build -t mlops-flux-gen:latest -f projects/flux-comfyui-generation/Dockerfile .
# Run:   docker run mlops-flux-gen:latest python run_generation.py --prompts "..."
# =============================================================================

ARG BASE_IMAGE=mlops-diffusers:latest
FROM ${BASE_IMAGE}

LABEL maintainer="MLOps Team"
LABEL description="Flux image generation worker"
LABEL version="1.0.0"
LABEL project="flux-comfyui-generation"

# Install additional dependencies for this project
RUN pip install \
    wandb>=0.16.0

# Create project directories
RUN mkdir -p /workspace/projects/flux-comfyui-generation \
    /workspace/outputs/flux-generations \
    /workspace/data/flux \
    && chown -R mlops:mlops /workspace

# Copy project files
COPY --chown=mlops:mlops config.py /workspace/
COPY --chown=mlops:mlops data_transfer/ /workspace/data_transfer/
COPY --chown=mlops:mlops projects/flux-comfyui-generation/ /workspace/projects/flux-comfyui-generation/

USER mlops
WORKDIR /workspace

# Environment
ENV WANDB_MODE=offline
ENV WANDB_PROJECT=flux-comfyui-generation
ENV WANDB_DIR=/workspace/outputs/wandb
ENV COMPUTE_DEVICE=cpu

# PHASE 2/3 TODO: GPU configuration
# ENV COMPUTE_DEVICE=cuda
# ENV CUDA_VISIBLE_DEVICES=0

# Default command
CMD ["python", "projects/flux-comfyui-generation/run_generation.py"]
