# =============================================================================
# Data Collector Container
# =============================================================================
# Collects images/videos from Reddit and other sources, with auto-captioning.
#
# Usage:
#   docker-compose run --rm data-collector python collect.py --subreddit earthporn --limit 100
#   docker-compose run --rm data-collector python caption.py --input-dir ./data/collected
# =============================================================================
FROM python:3.13-slim

LABEL maintainer="MLOps Team"
LABEL project="data-collector"

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /workspace

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl ffmpeg libgl1 libglib2.0-0 libsm6 libxext6 libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Create user
RUN useradd --create-home --shell /bin/bash --uid 1000 mlops \
    && mkdir -p /workspace && chown -R mlops:mlops /workspace

# Python dependencies
RUN pip install --no-cache-dir \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    torch torchvision \
    # Data collection
    gallery-dl \
    yt-dlp \
    # Captioning
    transformers \
    accelerate \
    # Dataset management
    fiftyone \
    # Utilities
    Pillow numpy tqdm requests \
    python-dotenv pyyaml

# Create directories
RUN mkdir -p /workspace/data/{collected,captioned,datasets} /workspace/outputs \
    && chown -R mlops:mlops /workspace

# Copy project files
COPY --chown=mlops:mlops config.py /workspace/
COPY --chown=mlops:mlops data_transfer/ /workspace/data_transfer/
COPY --chown=mlops:mlops projects/data-collector/ /workspace/projects/data-collector/

USER mlops

ENV GALLERY_DL_CONFIG=/workspace/projects/data-collector/gallery-dl.conf
ENV HF_HOME=/workspace/.cache/huggingface

CMD ["python", "projects/data-collector/collect.py", "--help"]
