# =============================================================================
# Zotero Web Service - Paper Management and Reference Server
# =============================================================================
# REST API for paper management with metadata extraction

FROM python:3.11-slim

LABEL maintainer="MLOps Workbench"
LABEL description="Zotero-style REST API for paper management"

WORKDIR /app

# Install uv for fast package management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Install Python dependencies with uv
COPY services/zotero/requirements.txt /app/requirements.txt
RUN uv pip install --system --no-cache -r /app/requirements.txt

# Copy REST API server
COPY services/zotero/server.py /app/server.py

# Create data directories
RUN mkdir -p /data/zotero /data/storage

EXPOSE 8085

# Run the REST API server directly
CMD ["python", "/app/server.py"]
