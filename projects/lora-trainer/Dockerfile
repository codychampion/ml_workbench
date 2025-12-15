# =============================================================================
# LoRA Trainer Worker
# =============================================================================
# Worker container for LoRA fine-tuning jobs.
# Optimized for memory-efficient training.
#
# Build: docker build -t mlops-lora-trainer:latest -f projects/lora-trainer/Dockerfile .
# Run:   docker run mlops-lora-trainer:latest python train_lora.py
# =============================================================================

ARG BASE_IMAGE=mlops-diffusers:latest
FROM ${BASE_IMAGE}

LABEL maintainer="MLOps Team"
LABEL description="LoRA fine-tuning worker"
LABEL version="1.0.0"
LABEL project="lora-trainer"

# Install LoRA training dependencies
RUN pip install \
    wandb>=0.16.0 \
    datasets>=2.14.0 \
    bitsandbytes>=0.41.0

# PHASE 2/3 TODO: Install PEFT for actual LoRA training
# RUN pip install peft>=0.6.0

# Create project directories
RUN mkdir -p /workspace/projects/lora-trainer \
    /workspace/outputs/lora-checkpoints \
    /workspace/data/lora-training \
    && chown -R mlops:mlops /workspace

# Copy project files
COPY --chown=mlops:mlops config.py /workspace/
COPY --chown=mlops:mlops data_transfer/ /workspace/data_transfer/
COPY --chown=mlops:mlops projects/lora-trainer/ /workspace/projects/lora-trainer/

USER mlops
WORKDIR /workspace

# Environment
ENV WANDB_MODE=offline
ENV WANDB_PROJECT=lora-trainer
ENV WANDB_DIR=/workspace/outputs/wandb
ENV COMPUTE_DEVICE=cpu

# PHASE 2/3 TODO: GPU optimization settings
# ENV COMPUTE_DEVICE=cuda
# ENV PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Default command
CMD ["python", "projects/lora-trainer/train_lora.py"]
