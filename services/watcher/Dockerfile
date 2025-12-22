# =============================================================================
# Knowledge/Run Watcher
# =============================================================================
FROM python:3.12-slim

WORKDIR /workspace

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends git curl && rm -rf /var/lib/apt/lists/*

# Use uv for fast installs
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy repo
COPY . /workspace

# Install minimal deps for knowledge syncing
RUN uv pip install --system --no-cache aim requests python-dotenv pyyaml

CMD ["python", "services/watcher/watcher.py"]
