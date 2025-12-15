# =============================================================================
# ComfyUI Service Container
# =============================================================================
# Runs ComfyUI as a standalone service for node-based image generation.
# Exposes web UI on port 8188.
#
# Build: docker build -t mlops-comfyui:latest -f services/comfyui/Dockerfile .
# Run:   docker run -p 8188:8188 mlops-comfyui:latest
# =============================================================================

ARG BASE_IMAGE=mlops-torch-cpu:latest
FROM ${BASE_IMAGE}

LABEL maintainer="MLOps Team"
LABEL description="ComfyUI service for node-based image generation"
LABEL version="1.0.0"
LABEL service="comfyui"

# Install ComfyUI dependencies
RUN pip install \
    aiohttp>=3.9.0 \
    einops>=0.7.0 \
    kornia>=0.7.0 \
    spandrel>=0.1.0

# PHASE 2/3 TODO: Clone and setup actual ComfyUI
# For Phase 1, we create a minimal mock server
# RUN git clone https://github.com/comfyanonymous/ComfyUI.git /opt/comfyui
# WORKDIR /opt/comfyui
# RUN pip install -r requirements.txt

# Create ComfyUI directories
RUN mkdir -p /workspace/comfyui/{models,input,output,custom_nodes} \
    && chown -R mlops:mlops /workspace/comfyui

# Copy service files
COPY --chown=mlops:mlops services/comfyui/server.py /workspace/comfyui/
COPY --chown=mlops:mlops services/comfyui/config.yaml /workspace/comfyui/

USER mlops
WORKDIR /workspace/comfyui

# Environment
ENV COMFYUI_PORT=8188
ENV COMFYUI_HOST=0.0.0.0
ENV COMFYUI_OUTPUT_DIR=/workspace/comfyui/output
ENV COMFYUI_INPUT_DIR=/workspace/comfyui/input
ENV COMFYUI_MODELS_DIR=/workspace/comfyui/models

EXPOSE 8188

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8188/health || exit 1

# PHASE 2/3 TODO: Run actual ComfyUI
# CMD ["python", "main.py", "--listen", "0.0.0.0", "--port", "8188"]
CMD ["python", "server.py"]
