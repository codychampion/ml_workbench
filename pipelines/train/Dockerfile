# =============================================================================
# Pipeline Stage 3: Model Training
# =============================================================================
# Fine-tuning captioners, LoRA adapters, and other models.
#
# Usage:
#   docker-compose run --rm train python -m pipelines.train.finetune --help
#   docker-compose run --rm train python -m pipelines.train.train_lora --help
# =============================================================================
FROM python:3.11-slim

LABEL maintainer="MLOps Team"
LABEL pipeline.stage="3-train"

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

# Full ML stack for training
RUN uv pip install --system --no-cache \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    torch torchvision \
    transformers>=4.36.0 \
    accelerate \
    peft>=0.7.0 \
    datasets>=2.16.0 \
    aim \
    prefect>=2.14.0 \
    hydra-core>=1.3.0 \
    omegaconf>=2.3.0 \
    boto3 \
    hvac \
    "dvc[s3]>=3.0.0" \
    Pillow numpy tqdm requests \
    python-dotenv pyyaml \
    sentencepiece bitsandbytes

RUN mkdir -p /workspace/data/annotated /workspace/models /workspace/outputs \
    && chown -R mlops:mlops /workspace

COPY --chown=mlops:mlops config.py /workspace/
COPY --chown=mlops:mlops data_transfer/ /workspace/data_transfer/
COPY --chown=mlops:mlops pipelines/ /workspace/pipelines/
COPY --chown=mlops:mlops conf/ /workspace/conf/
COPY --chown=mlops:mlops utils/ /workspace/utils/

USER mlops

ENV HF_HOME=/workspace/.cache/huggingface
ENV AIM_REPO=/workspace/outputs/aim
ENV COMPUTE_DEVICE=cpu
ENV PREFECT_API_URL=http://prefect-server:4200/api

CMD ["python", "-m", "pipelines.train.finetune", "--help"]
