FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Clone ComfyUI
RUN git clone https://github.com/comfyanonymous/ComfyUI.git . && \
    git checkout master

# Install Python dependencies
RUN pip3 install --no-cache-dir \
    torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 && \
    pip3 install --no-cache-dir -r requirements.txt

# Create directories for models, outputs, and inputs
RUN mkdir -p models/checkpoints models/vae models/loras \
    output input/custom user

# Expose ComfyUI port
EXPOSE 8188

# Run as non-root user
RUN useradd -m -u 1000 comfyui && \
    chown -R comfyui:comfyui /app

USER comfyui

CMD ["python3", "main.py", "--listen", "0.0.0.0", "--port", "8188"]
