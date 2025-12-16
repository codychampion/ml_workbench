# =============================================================================
# Pipeline Stage 5: Model Evaluation
# =============================================================================
# Benchmarking, metrics calculation, and model evaluation.
#
# Usage:
#   docker-compose run --rm evaluate python -m pipelines.evaluate.benchmark --help
#   docker-compose run --rm evaluate python -m pipelines.evaluate.metrics --help
# =============================================================================
FROM python:3.13-slim

LABEL maintainer="MLOps Team"
LABEL pipeline.stage="5-evaluate"

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl libgl1 libglib2.0-0 libsm6 libxext6 libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

RUN useradd --create-home --shell /bin/bash --uid 1000 mlops \
    && mkdir -p /workspace && chown -R mlops:mlops /workspace

# Evaluation dependencies
RUN pip install --no-cache-dir \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    torch torchvision \
    transformers>=4.36.0 \
    accelerate \
    peft>=0.7.0 \
    datasets>=2.16.0 \
    scikit-learn \
    scipy \
    nltk \
    rouge-score \
    matplotlib \
    seaborn \
    pandas \
    aim \
    hydra-core>=1.3.0 \
    omegaconf>=2.3.0 \
    fiftyone \
    Pillow numpy tqdm requests \
    python-dotenv pyyaml \
    tabulate jinja2

RUN mkdir -p /workspace/models /workspace/outputs/{metrics,reports,benchmarks} /workspace/data \
    && chown -R mlops:mlops /workspace

COPY --chown=mlops:mlops config.py /workspace/
COPY --chown=mlops:mlops data_transfer/ /workspace/data_transfer/
COPY --chown=mlops:mlops pipelines/ /workspace/pipelines/
COPY --chown=mlops:mlops conf/ /workspace/conf/
COPY --chown=mlops:mlops utils/ /workspace/utils/

USER mlops

ENV HF_HOME=/workspace/.cache/huggingface
ENV AIM_REPO=/workspace/outputs/aim

CMD ["python", "-m", "pipelines.evaluate.benchmark", "--help"]
