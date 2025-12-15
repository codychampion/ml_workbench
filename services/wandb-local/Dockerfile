# =============================================================================
# W&B Local Service Container
# =============================================================================
FROM python:3.11-slim

LABEL maintainer="MLOps Team"
LABEL service="wandb-local"

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /workspace

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl tini \
    && rm -rf /var/lib/apt/lists/*

# Create user
RUN useradd --create-home --shell /bin/bash --uid 1000 mlops \
    && mkdir -p /workspace && chown -R mlops:mlops /workspace

# Python dependencies
RUN pip install --no-cache-dir wandb

# Create directories
RUN mkdir -p /workspace/wandb/{runs,artifacts,cache} \
    && chown -R mlops:mlops /workspace/wandb

# Copy service files
COPY --chown=mlops:mlops services/wandb-local/server.py /workspace/wandb/
COPY --chown=mlops:mlops services/wandb-local/config.yaml /workspace/wandb/

USER mlops
WORKDIR /workspace/wandb

ENV WANDB_MODE=offline
ENV WANDB_DIR=/workspace/wandb/runs

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["python", "server.py"]
