# =============================================================================
# Zotero Web Service - Paper Management and Reference Server
# =============================================================================
# REST API for paper management with metadata extraction

FROM python:3.11-slim

LABEL maintainer="MLOps Workbench"
LABEL description="Zotero-style REST API for paper management"

WORKDIR /app

# Install Python dependencies
COPY services/zotero/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy REST API server
COPY services/zotero/server.py /app/server.py

# Create data directories
RUN mkdir -p /data/zotero /data/storage

EXPOSE 8085

# Run the REST API server directly (skip translation-server for now)
CMD ["python", "/app/server.py"]
