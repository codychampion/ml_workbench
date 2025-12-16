# =============================================================================
# Pipeline Stage 1: Data Collection
# =============================================================================
# Lightweight container for data gathering. No ML dependencies.
#
# Usage:
#   docker-compose run --rm collect python -m pipelines.collect.collect --help
# =============================================================================
FROM python:3.13-slim

LABEL maintainer="MLOps Team"
LABEL pipeline.stage="1-collect"

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /workspace

# Minimal system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl ffmpeg \
    && rm -rf /var/lib/apt/lists/*

RUN useradd --create-home --shell /bin/bash --uid 1000 mlops \
    && mkdir -p /workspace && chown -R mlops:mlops /workspace

# Lightweight Python dependencies - NO ML FRAMEWORKS
RUN pip install --no-cache-dir \
    gallery-dl \
    yt-dlp \
    "dvc[all]>=3.0.0" \
    Pillow requests tqdm \
    python-dotenv pyyaml

RUN mkdir -p /workspace/data/{raw,collected} \
    && chown -R mlops:mlops /workspace

COPY --chown=mlops:mlops config.py /workspace/
COPY --chown=mlops:mlops data_transfer/ /workspace/data_transfer/
COPY --chown=mlops:mlops pipelines/ /workspace/pipelines/

USER mlops

ENV GALLERY_DL_CONFIG=/workspace/pipelines/collect/gallery-dl.conf

CMD ["python", "-m", "pipelines.collect.collect", "--help"]
