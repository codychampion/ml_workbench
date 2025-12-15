# =============================================================================
# ComfyUI Service Container
# =============================================================================
FROM python:3.11-slim

LABEL maintainer="MLOps Team"
LABEL service="comfyui"

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
RUN pip install --no-cache-dir \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    torch torchvision \
    aiohttp einops Pillow numpy

# Create directories
RUN mkdir -p /workspace/comfyui/{models,input,output,custom_nodes} \
    && chown -R mlops:mlops /workspace/comfyui

# Copy service files
COPY --chown=mlops:mlops services/comfyui/server.py /workspace/comfyui/
COPY --chown=mlops:mlops services/comfyui/config.yaml /workspace/comfyui/

USER mlops
WORKDIR /workspace/comfyui

ENV COMFYUI_PORT=8188
ENV COMFYUI_HOST=0.0.0.0

EXPOSE 8188

HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8188/health || exit 1

ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["python", "server.py"]
