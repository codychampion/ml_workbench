# =============================================================================
# Pipeline Stage 1: Data Collection
# =============================================================================
# Lightweight container for data gathering. No ML dependencies.
#
# Usage:
#   docker-compose run --rm collect python -m pipelines.collect.collect --help
# =============================================================================
FROM python:3.12-slim

LABEL maintainer="MLOps Team"
LABEL pipeline.stage="1-collect"

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /workspace

# Minimal system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast package management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

RUN useradd --create-home --shell /bin/bash --uid 1000 mlops \
    && mkdir -p /workspace && chown -R mlops:mlops /workspace

# Lightweight Python dependencies - NO ML FRAMEWORKS
RUN uv pip install --system --no-cache \
    gallery-dl \
    yt-dlp \
    boto3 \
    hvac \
    prefect>=2.14.0 \
    "dvc[s3]>=3.0.0" \
    Pillow requests tqdm \
    python-dotenv pyyaml

RUN mkdir -p /workspace/data/{raw,collected} \
    && chown -R mlops:mlops /workspace

COPY --chown=mlops:mlops config.py /workspace/
COPY --chown=mlops:mlops data_transfer/ /workspace/data_transfer/
COPY --chown=mlops:mlops pipelines/ /workspace/pipelines/

USER mlops

ENV GALLERY_DL_CONFIG=/workspace/pipelines/collect/gallery-dl.conf
ENV PREFECT_API_URL=http://prefect-server:4200/api

CMD ["python", "-m", "pipelines.collect.collect", "--help"]
