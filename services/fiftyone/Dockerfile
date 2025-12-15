# =============================================================================
# FiftyOne Service Container
# =============================================================================
FROM python:3.13-slim

LABEL maintainer="MLOps Team"
LABEL service="fiftyone"

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /workspace

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl libgl1 libglib2.0-0 libsm6 libxext6 libxrender-dev tini \
    && rm -rf /var/lib/apt/lists/*

# Create user
RUN useradd --create-home --shell /bin/bash --uid 1000 mlops \
    && mkdir -p /workspace && chown -R mlops:mlops /workspace

# Python dependencies
RUN pip install --no-cache-dir fiftyone pymongo

# Create directories
RUN mkdir -p /workspace/fiftyone/{datasets,cache,database} \
    && chown -R mlops:mlops /workspace/fiftyone

# Copy service files
COPY --chown=mlops:mlops services/fiftyone/server.py /workspace/fiftyone/
COPY --chown=mlops:mlops services/fiftyone/config.yaml /workspace/fiftyone/

USER mlops
WORKDIR /workspace/fiftyone

ENV FIFTYONE_DATABASE_DIR=/workspace/fiftyone/database
ENV FIFTYONE_APP_ADDRESS=0.0.0.0
ENV FIFTYONE_APP_PORT=5151

EXPOSE 5151

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:5151 || exit 1

ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["python", "server.py"]
