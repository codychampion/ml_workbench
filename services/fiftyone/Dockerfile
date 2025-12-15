# =============================================================================
# FiftyOne Service Container
# =============================================================================
# Runs FiftyOne App as a standalone service for data visualization.
# Exposes web UI on port 5151.
#
# Build: docker build -t mlops-fiftyone:latest -f services/fiftyone/Dockerfile .
# Run:   docker run -p 5151:5151 mlops-fiftyone:latest
# =============================================================================

ARG BASE_IMAGE=mlops-torch-cpu:latest
FROM ${BASE_IMAGE}

LABEL maintainer="MLOps Team"
LABEL description="FiftyOne service for ML data visualization"
LABEL version="1.0.0"
LABEL service="fiftyone"

# Install FiftyOne and dependencies
RUN pip install \
    fiftyone>=0.23.0 \
    fiftyone-brain>=0.16.0 \
    pymongo>=4.5.0

# Create FiftyOne directories
RUN mkdir -p /workspace/fiftyone/{datasets,cache,database} \
    && chown -R mlops:mlops /workspace/fiftyone

# Copy service files
COPY --chown=mlops:mlops services/fiftyone/server.py /workspace/fiftyone/
COPY --chown=mlops:mlops services/fiftyone/config.yaml /workspace/fiftyone/

USER mlops
WORKDIR /workspace/fiftyone

# FiftyOne environment
ENV FIFTYONE_DATABASE_DIR=/workspace/fiftyone/database
ENV FIFTYONE_DATASET_ZOO_DIR=/workspace/fiftyone/datasets
ENV FIFTYONE_DEFAULT_DATASET_DIR=/workspace/fiftyone/datasets
ENV FIFTYONE_APP_ADDRESS=0.0.0.0
ENV FIFTYONE_APP_PORT=5151

EXPOSE 5151

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:5151 || exit 1

CMD ["python", "server.py"]
