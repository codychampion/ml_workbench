# =============================================================================
# Pipeline Stage 2: Data Annotation
# =============================================================================
# Auto-captioning and dataset creation. Uses trained captioner models.
#
# Usage:
#   docker-compose run --rm annotate python -m pipelines.annotate.caption --help
#   docker-compose run --rm annotate python -m pipelines.annotate.create_dataset --help
# =============================================================================
FROM python:3.12-slim

LABEL maintainer="MLOps Team"
LABEL pipeline.stage="2-annotate"

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl libgl1 libglib2.0-0 libsm6 libxext6 libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast package management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

RUN useradd --create-home --shell /bin/bash --uid 1000 mlops \
    && mkdir -p /workspace && chown -R mlops:mlops /workspace

# ML dependencies for captioning
RUN uv pip install --system --no-cache \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    torch torchvision \
    transformers>=4.36.0 \
    accelerate \
    peft>=0.7.0 \
    fiftyone \
    "dvc[s3]>=3.0.0" \
    Pillow numpy tqdm requests \
    python-dotenv pyyaml \
    sentencepiece

RUN mkdir -p /workspace/data/{raw,annotated,processed} /workspace/models \
    && chown -R mlops:mlops /workspace

COPY --chown=mlops:mlops config.py /workspace/
COPY --chown=mlops:mlops data_transfer/ /workspace/data_transfer/
COPY --chown=mlops:mlops pipelines/ /workspace/pipelines/

USER mlops

ENV HF_HOME=/workspace/.cache/huggingface
ENV TRANSFORMERS_CACHE=/workspace/.cache/huggingface/transformers

CMD ["python", "-m", "pipelines.annotate.caption", "--help"]
