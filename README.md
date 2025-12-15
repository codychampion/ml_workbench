# MLOps Workspace

A professional MLOps repository with a **multi-container architecture** focused on reproducibility, experiment tracking, and modular service deployment.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  SERVICES (long-running)              WORKERS (job executors)                │
│  ┌────────────┐ ┌────────────┐ ┌────────────┐  ┌────────────┐ ┌───────────┐│
│  │  ComfyUI   │ │  FiftyOne  │ │ W&B Local  │  │  flux-gen  │ │lora-train ││
│  │   :8188    │ │   :5151    │ │   :8080    │  │            │ │           ││
│  └────────────┘ └────────────┘ └────────────┘  └────────────┘ └───────────┘│
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Quick Start

```bash
# Build all images
docker-compose build

# Start services (ComfyUI, FiftyOne, W&B)
docker-compose up -d

# Run a worker job
docker-compose run --rm flux-gen
docker-compose run --rm lora-trainer

# Dev shell with all tools
docker-compose --profile dev up -d dev
docker-compose exec dev bash

# Stop everything
docker-compose down
```

### Services

| Service | URL | Description |
|---------|-----|-------------|
| ComfyUI | http://localhost:8188 | Image generation UI |
| FiftyOne | http://localhost:5151 | Data visualization |
| W&B Local | http://localhost:8080 | Experiment tracking |

---

## Project Structure

```
mlops-workspace/
├── docker/base/                 # Base Dockerfiles
│   ├── Dockerfile.base          # Python foundation
│   ├── Dockerfile.torch-cpu     # + PyTorch CPU
│   └── Dockerfile.diffusers     # + HuggingFace
│
├── services/                    # Long-running servers
│   ├── comfyui/
│   ├── fiftyone/
│   └── wandb-local/
│
├── projects/                    # Worker containers
│   ├── flux-comfyui-generation/
│   ├── adversarial-patches/
│   └── lora-trainer/
│
├── docker-compose.yml           # Main orchestration
├── docker-compose.gpu.yml       # GPU overlay (Phase 2/3)
└── config.py                    # Central config
```

---

## Running Workers

Workers are defined with profiles so they don't start automatically:

```bash
# Image generation
docker-compose run --rm flux-gen python projects/flux-comfyui-generation/run_generation.py \
    --prompts "A mountain at sunset"

# Adversarial patches
docker-compose run --rm adv-patches python projects/adversarial-patches/generate_patch.py

# LoRA training
docker-compose run --rm lora-trainer python projects/lora-trainer/train_lora.py --epochs 10
```

---

## GPU Support (Phase 2/3)

```bash
# Use GPU overlay
docker-compose -f docker-compose.yml -f docker-compose.gpu.yml up -d
```

---

## Adding New Projects

1. Create `projects/my-project/Dockerfile`:
```dockerfile
FROM mlops-diffusers:latest
COPY projects/my-project/ /workspace/projects/my-project/
CMD ["python", "projects/my-project/main.py"]
```

2. Add to `docker-compose.yml`:
```yaml
my-project:
  build:
    dockerfile: projects/my-project/Dockerfile
  profiles:
    - workers
```

3. Run: `docker-compose run --rm my-project`

---

## Common Commands

```bash
docker-compose build              # Build all
docker-compose up -d              # Start services
docker-compose run --rm flux-gen  # Run worker
docker-compose logs -f comfyui    # View logs
docker-compose down               # Stop all
docker-compose down -v            # Stop + remove volumes
```

---

*Phase 1: CPU-only, W&B offline, mocked B2 storage*
