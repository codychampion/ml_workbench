# =============================================================================
# MLOps Workbench - Integration Test Runner
# =============================================================================
# Runs integration tests to verify all services are working correctly.
# This container waits for services to be healthy before running tests.

FROM python:3.11-slim

LABEL maintainer="MLOps Workbench"
LABEL description="Integration test runner for MLOps services"

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    netcat-openbsd \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast package management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /workspace

# Copy test requirements and install
COPY tests/requirements.txt /workspace/tests/requirements.txt
RUN uv pip install --system --no-cache -r /workspace/tests/requirements.txt

# Copy test files
COPY tests/ /workspace/tests/
COPY pytest.ini /workspace/pytest.ini

# Copy wait script
COPY tests/docker/wait-for-services.sh /workspace/wait-for-services.sh
RUN chmod +x /workspace/wait-for-services.sh

# Default environment
ENV SERVICE_HOST=localhost
ENV PYTHONUNBUFFERED=1

# Run tests
CMD ["/workspace/wait-for-services.sh"]
