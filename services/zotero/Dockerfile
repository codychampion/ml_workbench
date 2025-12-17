# =============================================================================
# Zotero Web Service - Paper Management and Reference Server
# =============================================================================
# Combines Zotero translation-server with a REST API for:
# - Paper metadata extraction from URLs/DOIs
# - BibTeX/CSL-JSON export
# - PDF storage with S3 sync
# - Integration with Khoj for AI-powered paper search

FROM node:20-slim

LABEL maintainer="MLOps Workbench"
LABEL description="Zotero translation server with REST API for paper management"

# Install dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    python3 \
    python3-pip \
    python3-venv \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Clone and build Zotero translation-server
RUN git clone --depth 1 https://github.com/zotero/translation-server.git /app/translation-server \
    && cd /app/translation-server \
    && npm install

# Create Python virtual environment for REST API
RUN python3 -m venv /app/venv
ENV PATH="/app/venv/bin:$PATH"

# Install Python dependencies for REST API wrapper
COPY services/zotero/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy REST API server
COPY services/zotero/server.py /app/server.py

# Create data directories
RUN mkdir -p /data/zotero /data/storage

EXPOSE 8085
EXPOSE 1969

# Start both translation-server and REST API
COPY services/zotero/startup.sh /app/startup.sh
RUN chmod +x /app/startup.sh

CMD ["/app/startup.sh"]
