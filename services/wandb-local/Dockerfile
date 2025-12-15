# =============================================================================
# W&B Local Service Container
# =============================================================================
# Provides local W&B experiment tracking infrastructure.
# Phase 1: Runs wandb local server for offline syncing.
# Phase 2/3: Can be replaced with W&B Server or cloud.
#
# Build: docker build -t mlops-wandb-local:latest -f services/wandb-local/Dockerfile .
# =============================================================================

ARG BASE_IMAGE=mlops-base:latest
FROM ${BASE_IMAGE}

LABEL maintainer="MLOps Team"
LABEL description="W&B local server for experiment tracking"
LABEL version="1.0.0"
LABEL service="wandb-local"

# Install W&B
RUN pip install \
    wandb>=0.16.0

# Create W&B directories
RUN mkdir -p /workspace/wandb/{runs,artifacts,cache} \
    && chown -R mlops:mlops /workspace/wandb

# Copy service files
COPY --chown=mlops:mlops services/wandb-local/server.py /workspace/wandb/
COPY --chown=mlops:mlops services/wandb-local/config.yaml /workspace/wandb/

USER mlops
WORKDIR /workspace/wandb

# W&B environment
ENV WANDB_DIR=/workspace/wandb/runs
ENV WANDB_CACHE_DIR=/workspace/wandb/cache
ENV WANDB_MODE=offline
ENV WANDB_SILENT=false

# The local server typically uses port 8080
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD python -c "import wandb; print('healthy')" || exit 1

CMD ["python", "server.py"]
