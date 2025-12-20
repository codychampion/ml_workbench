# MLOps Workbench

A comprehensive, self-hosted ML/AI pipeline workbench with a **pipeline-based architecture** for data collection, annotation, training, evaluation, and inference.

## Architecture

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│  INFRASTRUCTURE SERVICES                                                          │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐  │
│  │   AIM    │ │  Vault   │ │ LiteLLM  │ │ JuiceFS  │ │ MongoDB  │ │  Redis   │  │
│  │  :43800  │ │  :8200   │ │  :4000   │ │(storage) │ │  (meta)  │ │ (cache)  │  │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘ └──────────┘ └──────────┘  │
├──────────────────────────────────────────────────────────────────────────────────┤
│  DATA TOOLS                                                                       │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐               │
│  │ FiftyOne │ │  Label   │ │   CVAT   │ │Spotlight │ │ ComfyUI  │               │
│  │  :5151   │ │  Studio  │ │  :8082   │ │  :8083   │ │  :8188   │               │
│  │  (viz)   │ │  :8081   │ │ (video)  │ │(explore) │ │  (gen)   │               │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘ └──────────┘               │
├──────────────────────────────────────────────────────────────────────────────────┤
│  PIPELINE STAGES                                                                  │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐               │
│  │ 1.collect│→│2.annotate│→│ 3.train  │→│4.evaluate│→│ 5.infer  │               │
│  │  (data)  │ │(caption) │ │  (LoRA)  │ │(metrics) │ │  (gen)   │               │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘ └──────────┘               │
└──────────────────────────────────────────────────────────────────────────────────┘
```

## Services Overview

| Service | Port | Description |
|---------|------|-------------|
| **AIM** | 43800 | Experiment tracking (replaces W&B) |
| **Vault** | 8200 | Secrets management |
| **LiteLLM** | 4000 | LLM API gateway (OpenAI-compatible) |
| **FiftyOne** | 5151 | Dataset visualization |
| **Label Studio** | 8081 | Image/text annotation |
| **CVAT** | 8082 | Video annotation |
| **Spotlight** | 8083 | Data exploration (Renumics) |
| **ComfyUI** | 8188 | Image generation workflows |

---

## Quick Start

```bash
# Discover helpful shortcuts
make help

# Build images and start the lightweight core infrastructure profile
make build
make infra

# (Optional) add data tooling (FiftyOne, Label Studio, Khoj/Obsidian, Zotero)
make tools

# Run pipeline stages (pass extra flags via ARGS)
make collect ARGS="--subreddit earthporn"
make annotate ARGS="--input-dir ./data/collected"
make train ARGS="--dataset ./data/collected"
make evaluate ARGS="--predictions ./outputs"
make infer ARGS="--prompts 'A sunset'"

# Dev shell with the core profile (add PROFILES="core tools" for data UIs)
make dev-shell

# Stop everything
make down
```

The defaults are tuned for a single-machine prototype: only the core services start by default, and heavier data UIs (FiftyOne, Label Studio, Khoj + Obsidian + Zotero) are opt-in.
See the [Developer Guide](./docs/DEVELOPER_GUIDE.md) for profile guidance, pipeline shortcuts, testing, and DVC usage.

---

## Project Structure

```
ml_workbench/
├── pipelines/                      # Pipeline stages
│   ├── collect/                    # Stage 1: Data collection
│   │   ├── Dockerfile
│   │   └── collect.py              # Reddit/gallery-dl scraping
│   │
│   ├── annotate/                   # Stage 2: Data annotation
│   │   ├── Dockerfile
│   │   ├── caption.py              # Auto-captioning
│   │   ├── models.py               # Model configs
│   │   └── create_dataset.py       # FiftyOne dataset creation
│   │
│   ├── train/                      # Stage 3: Model training
│   │   ├── Dockerfile
│   │   ├── finetune.py             # Captioner fine-tuning
│   │   └── train_lora.py           # LoRA training
│   │
│   ├── evaluate/                   # Stage 4: Evaluation
│   │   ├── Dockerfile
│   │   ├── benchmark.py            # Performance benchmarks
│   │   └── metrics.py              # BLEU/ROUGE metrics
│   │
│   └── infer/                      # Stage 5: Inference
│       ├── Dockerfile
│       ├── run_generation.py       # Image generation
│       └── generate_patch.py       # Adversarial patches
│
├── services/                       # Long-running services
│   ├── fiftyone/                   # Dataset visualization
│   ├── spotlight/                  # Data exploration
│   └── comfyui/                    # Image generation UI
│
├── docker/                         # Docker configs
│   ├── base/                       # Base Dockerfiles
│   ├── init-scripts/               # PostgreSQL init
│   ├── litellm/                    # LiteLLM config
│   └── vault/                      # Vault config
│
├── data/                           # Data directories
│   ├── raw/                        # Raw collected data
│   ├── collected/                  # Processed collections
│   └── processed/                  # Annotated data
│
├── models/                         # Trained models
├── outputs/                        # Pipeline outputs
│   ├── aim/                        # AIM experiment logs
│   └── checkpoints/                # Model checkpoints
│
├── docker-compose.yml              # Main orchestration
├── config.py                       # Central configuration
├── dvc.yaml                        # DVC pipeline definition
└── params.yaml                     # Pipeline parameters
```

---

## Pipeline Stages

### 1. Collect - Data Gathering
```bash
# Collect images from Reddit
docker-compose run --rm collect python -m pipelines.collect.collect \
    --subreddit earthporn --limit 100

# Collect from URL
docker-compose run --rm collect python -m pipelines.collect.collect \
    --url "https://reddit.com/r/art/top"
```

### 2. Annotate - Auto-Captioning
```bash
# Caption collected images
docker-compose run --rm annotate python -m pipelines.annotate.caption \
    --input-dir ./data/collected --model blip-base

# Create FiftyOne dataset
docker-compose run --rm annotate python -m pipelines.annotate.create_dataset \
    --input-dir ./data/collected --name my-dataset
```

### 3. Train - Model Fine-Tuning
```bash
# Fine-tune captioner with LoRA
docker-compose run --rm train python -m pipelines.train.finetune \
    --dataset ./data/collected --model blip-base --epochs 3

# Train LoRA for image generation
docker-compose run --rm train python -m pipelines.train.train_lora \
    --epochs 10 --lora-rank 16
```

### 4. Evaluate - Metrics & Benchmarks
```bash
# Run benchmarks
docker-compose run --rm evaluate python -m pipelines.evaluate.benchmark \
    --model ./models/captioner

# Calculate metrics
docker-compose run --rm evaluate python -m pipelines.evaluate.metrics \
    --predictions ./outputs/predictions.json
```

### 5. Infer - Generation
```bash
# Generate images
docker-compose run --rm infer python -m pipelines.infer.run_generation \
    --prompts "A mountain at sunset" "A futuristic city"
```

---

## DVC Pipeline

Run the full pipeline with DVC:

```bash
# Run full pipeline
dvc repro

# Run specific stage
dvc repro train-captioner

# View pipeline graph
dvc dag
```

---

## Self-Hosted Services

All services are self-hosted with no external API dependencies:

### AIM - Experiment Tracking
```bash
# Access UI
open http://localhost:43800

# Python usage
from aim import Run
run = Run(repo="./outputs/aim")
run.track(loss, name="loss", epoch=epoch)
```

### Vault - Secrets Management
```bash
# Access UI (dev token: mlops-dev-token)
open http://localhost:8200

# Store a secret
vault kv put secret/mlops/api-keys openai=sk-xxx
```

### LiteLLM - LLM Gateway
```bash
# OpenAI-compatible API
curl http://localhost:4000/v1/chat/completions \
    -H "Authorization: Bearer sk-mlops-dev-key" \
    -d '{"model": "ollama/mistral", "messages": [{"role": "user", "content": "Hello"}]}'
```

### CVAT - Video Annotation
```bash
# Access UI
open http://localhost:8082

# Default credentials: admin/admin
```

### Label Studio - Data Labeling
```bash
# Access UI
open http://localhost:8081
```

### Renumics Spotlight - Data Exploration
```bash
# Access UI
open http://localhost:8083
```

---

## Configuration

All configuration is managed via environment variables or `config.py`:

```python
from config import get_config, get_secret

config = get_config()

# AIM tracking
print(config.aim.repo)           # ./outputs/aim
print(config.aim.server)         # http://aim:53800

# Vault secrets
api_key = get_secret("mlops/api-keys", "openai")

# LiteLLM
print(config.litellm.api_base)   # http://litellm:4000
```

---

## Common Commands

```bash
# Build
docker-compose build

# Start services
docker-compose up -d

# Run pipeline stage
docker-compose run --rm <stage> <command>

# View logs
docker-compose logs -f aim

# Stop all
docker-compose down

# Stop + remove volumes
docker-compose down -v

# GPU support (when available)
docker-compose -f docker-compose.yml -f docker-compose.gpu.yml up -d
```

---

## Adding New Pipeline Stages

1. Create `pipelines/my-stage/Dockerfile`:
```dockerfile
FROM python:3.13-slim
LABEL pipeline.stage="my-stage"
WORKDIR /workspace
COPY pipelines/my-stage/ /workspace/pipelines/my-stage/
RUN pip install -r pipelines/my-stage/requirements.txt
```

2. Add to `docker-compose.yml`:
```yaml
my-stage:
  build:
    dockerfile: pipelines/my-stage/Dockerfile
  profiles:
    - pipeline
```

3. Run: `docker-compose run --rm my-stage python -m pipelines.my_stage.main`

---

*Self-hosted MLOps: AIM tracking, JuiceFS storage, Vault secrets, LiteLLM gateway*
