# =============================================================================
# Renumics Spotlight - Data Exploration & Visualization
# =============================================================================
# Interactive data exploration for ML datasets with embedding visualization,
# automatic outlier detection, and data quality analysis.
#
# Features:
#   - Embedding visualization (UMAP, t-SNE)
#   - Automatic data issue detection
#   - Image/audio/text preview
#   - Integration with FiftyOne datasets
#
# Usage:
#   docker-compose up spotlight
#   # Access at http://localhost:8083
# =============================================================================

FROM python:3.11-slim

LABEL maintainer="mlops-workbench"
LABEL service="spotlight"
LABEL description="Renumics Spotlight data exploration service"

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Create workspace
WORKDIR /workspace

# Install Python dependencies
RUN pip install --no-cache-dir \
    renumics-spotlight>=1.6.0 \
    pandas \
    numpy \
    pillow \
    h5py \
    pyarrow \
    requests \
    fiftyone

# Copy startup script
COPY services/spotlight/start.py /workspace/start.py

# Environment
ENV SPOTLIGHT_HOST=0.0.0.0
ENV SPOTLIGHT_PORT=8000
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

# Default command - start Spotlight server
CMD ["python", "/workspace/start.py"]
