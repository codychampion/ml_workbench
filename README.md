# MLOps Workspace

A professional MLOps repository with a **multi-container architecture** focused on reproducibility, experiment tracking, and modular service deployment.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    Multi-Container MLOps Architecture                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  BASE IMAGES (Build Hierarchy)                                               │
│  ┌────────────┐    ┌────────────┐    ┌────────────┐    ┌────────────┐       │
│  │ mlops-base │───▶│torch-cpu/  │───▶│ diffusers  │───▶│  Workers   │       │
│  │  (python)  │    │  torch-gpu │    │ (HF libs)  │    │            │       │
│  └────────────┘    └────────────┘    └────────────┘    └────────────┘       │
│                                                                              │
│  SERVICE CONTAINERS (Long-running)                                           │
│  ┌────────────┐    ┌────────────┐    ┌────────────┐                         │
│  │  ComfyUI   │    │  FiftyOne  │    │ W&B Local  │                         │
│  │   :8188    │    │   :5151    │    │   :8080    │                         │
│  └────────────┘    └────────────┘    └────────────┘                         │
│                                                                              │
│  WORKER CONTAINERS (Job Executors)                                           │
│  ┌────────────┐    ┌────────────┐    ┌────────────┐                         │
│  │  flux-gen  │    │adv-patches │    │lora-trainer│                         │
│  │ (generate) │    │ (attacks)  │    │ (training) │                         │
│  └────────────┘    └────────────┘    └────────────┘                         │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Container Strategy

| Type | Purpose | Examples |
|------|---------|----------|
| **Base Images** | Shared dependencies, faster builds | `mlops-base`, `mlops-torch-cpu`, `mlops-diffusers` |
| **Service Containers** | Long-running servers | ComfyUI, FiftyOne, W&B utilities |
| **Worker Containers** | Job execution | Image generation, LoRA training, adversarial patches |

### Benefits

- **Dependency Isolation**: Each service has its own dependencies
- **GPU Allocation**: Training gets A100, serving gets T4
- **Independent Scaling**: Scale inference without touching training
- **Faster Builds**: Layered images share common dependencies
- **Easy Updates**: Update ComfyUI without rebuilding training images

---

## Quick Start

### Prerequisites

- Docker and Docker Compose v2
- 8GB RAM recommended (4GB minimum)
- For GPU: NVIDIA Container Toolkit

### Phase 1: Build Images

```bash
# Build all images in dependency order
docker-compose -f docker-compose.build.yml build

# Or build specific images
docker-compose -f docker-compose.build.yml build base torch-cpu
```

### Phase 1: Start Services

```bash
# Start all services (ComfyUI, FiftyOne, W&B)
docker-compose up -d

# Start specific services
docker-compose up -d comfyui fiftyone

# View logs
docker-compose logs -f comfyui
```

### Access Services

| Service | URL | Description |
|---------|-----|-------------|
| ComfyUI | http://localhost:8188 | Node-based image generation |
| FiftyOne | http://localhost:5151 | Data visualization |
| W&B Local | http://localhost:8080 | Experiment tracking utilities |

### Run Workers

```bash
# Run image generation
docker-compose run --rm flux-gen python projects/flux-comfyui-generation/run_generation.py \
    --prompts "A mountain at sunset" "A futuristic city"

# Run adversarial patches
docker-compose run --rm adv-patches python projects/adversarial-patches/generate_patch.py

# Run LoRA training
docker-compose run --rm lora-trainer python projects/lora-trainer/train_lora.py --epochs 10
```

### Development Shell

```bash
# Start interactive development container
docker-compose --profile dev up -d dev
docker-compose exec dev bash

# Inside container, all tools are available
python projects/flux-comfyui-generation/run_generation.py
```

---

## Project Structure

```
mlops-workspace/
├── docker/
│   └── base/
│       ├── Dockerfile.base          # Python base image
│       ├── Dockerfile.torch-cpu     # PyTorch CPU
│       ├── Dockerfile.torch-gpu     # PyTorch GPU (Phase 2/3)
│       └── Dockerfile.diffusers     # HuggingFace stack
│
├── services/
│   ├── comfyui/
│   │   ├── Dockerfile               # ComfyUI service
│   │   ├── server.py                # Mock server (Phase 1)
│   │   └── config.yaml
│   ├── fiftyone/
│   │   ├── Dockerfile               # FiftyOne service
│   │   ├── server.py
│   │   └── config.yaml
│   └── wandb-local/
│       ├── Dockerfile               # W&B utilities
│       ├── server.py
│       └── config.yaml
│
├── projects/
│   ├── flux-comfyui-generation/
│   │   ├── Dockerfile               # Generation worker
│   │   └── run_generation.py
│   ├── adversarial-patches/
│   │   ├── Dockerfile               # Adversarial worker
│   │   └── generate_patch.py
│   └── lora-trainer/
│       ├── Dockerfile               # Training worker
│       └── train_lora.py
│
├── data_transfer/
│   ├── __init__.py
│   └── b2_client.py                 # Mocked B2 client
│
├── docker-compose.yml               # Main compose file
├── docker-compose.override.yml      # Local dev overrides
├── docker-compose.gpu.yml           # GPU configuration
├── docker-compose.build.yml         # Build order
├── config.py                        # Central configuration
├── requirements.txt                 # Python dependencies
└── .b2_local_manifest.json          # Mocked B2 manifest
```

---

## Docker Compose Profiles

The compose file uses profiles to organize containers:

```bash
# Default: Start services only (comfyui, fiftyone, wandb-local)
docker-compose up -d

# Include workers
docker-compose --profile workers up -d

# Include training workers
docker-compose --profile training up -d

# Development shell
docker-compose --profile dev up -d

# GPU-enabled (Phase 2/3)
docker-compose -f docker-compose.yml -f docker-compose.gpu.yml up -d
```

---

## GPU Configuration (Phase 2/3)

For GPU-enabled containers:

```bash
# Use the GPU override file
docker-compose -f docker-compose.yml -f docker-compose.gpu.yml up -d

# Or set as default
export COMPOSE_FILE=docker-compose.yml:docker-compose.gpu.yml
docker-compose up -d
```

The GPU compose file configures:
- NVIDIA Container Toolkit integration
- CUDA visibility settings
- GPU memory management
- Multi-GPU distributed training (planned)

---

## Tool Integration

### Weights & Biases (W&B)

All workers use W&B offline mode in Phase 1:

```python
import wandb

wandb.init(
    project="my-project",
    mode="offline",  # Phase 1
    dir="/workspace/outputs/wandb"
)

wandb.log({"loss": 0.5, "accuracy": 0.95})
```

**Sync offline runs:**
```bash
wandb sync ./outputs/wandb/offline-*
```

### FiftyOne

Access the FiftyOne app at http://localhost:5151

```python
import fiftyone as fo

# Load a dataset
dataset = fo.load_dataset("my-dataset")

# The app is already running as a service
# Just load data and it appears in the UI
```

### Backblaze B2 (Mocked)

Phase 1 uses a mocked client:

```python
from data_transfer import B2Client

client = B2Client()

# Lists from .b2_local_manifest.json
files = client.list_files(prefix="datasets/")

# Copies from local data/raw/
client.download_file("image.png", Path("./output.png"))
```

### ComfyUI

Access at http://localhost:8188

In Phase 1, uses a mock server that generates placeholder images.

---

## Adding New Projects

1. Create project directory:
```bash
mkdir -p projects/my-new-project
```

2. Create Dockerfile:
```dockerfile
ARG BASE_IMAGE=mlops-diffusers:latest
FROM ${BASE_IMAGE}

COPY --chown=mlops:mlops projects/my-new-project/ /workspace/projects/my-new-project/
CMD ["python", "projects/my-new-project/main.py"]
```

3. Add to docker-compose.yml:
```yaml
my-new-project:
  build:
    context: .
    dockerfile: projects/my-new-project/Dockerfile
  profiles:
    - workers
```

4. Run:
```bash
docker-compose run --rm my-new-project
```

---

## Phase 2/3 Roadmap

### Prefect Integration

```yaml
# Uncomment in docker-compose.yml
prefect-server:
  image: prefecthq/prefect:2-python3.11
  ports:
    - "4200:4200"
```

### SkyPilot Integration

```yaml
# sky.yaml for cloud GPU provisioning
resources:
  accelerators: A100:1
  use_spot: true

run: |
  python projects/lora-trainer/train_lora.py --device cuda
```

### Production Checklist

- [ ] Enable W&B online mode with API keys
- [ ] Configure real B2 storage with encryption
- [ ] Set up Prefect for workflow orchestration
- [ ] Deploy GPU workers via SkyPilot
- [ ] Add secrets management
- [ ] Configure monitoring (Prometheus/Grafana)

---

## Common Commands

```bash
# Build all images
docker-compose -f docker-compose.build.yml build

# Start services
docker-compose up -d

# Run a worker job
docker-compose run --rm flux-gen

# View logs
docker-compose logs -f comfyui

# Stop all
docker-compose down

# Clean up volumes
docker-compose down -v

# Rebuild a specific image
docker-compose build --no-cache comfyui

# Shell into a running container
docker-compose exec fiftyone bash
```

---

## Troubleshooting

### Build Failures

```bash
# Build with verbose output
docker-compose -f docker-compose.build.yml build --progress=plain

# Clean build cache
docker builder prune
```

### Port Conflicts

```bash
# Check what's using a port
lsof -i :8188
lsof -i :5151

# Kill the process
kill -9 <PID>
```

### GPU Not Detected

```bash
# Verify NVIDIA Container Toolkit
nvidia-smi
docker run --rm --gpus all nvidia/cuda:12.1-base nvidia-smi
```

---

## Security

- **Non-root containers**: All containers run as `mlops` user
- **No hardcoded secrets**: Credentials via environment variables
- **Rate limiting**: Built into B2 client
- **Network isolation**: Custom bridge network

---

*Phase 1: Local Core - Multi-container architecture for reproducible MLOps*
