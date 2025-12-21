---
type: documentation
category: mlops
title: "MLOps Workbench - AI Assistant Context Guide"
date: 2025-12-21
tags:
  - mlops
  - documentation
  - ai-assistant
  - architecture
  - guide
status: active
---

# MLOps Workbench - AI Assistant Context Guide

> Quick context file for AI assistants (Claude, GPT, etc.) to understand this codebase.

## Project Overview

Self-hosted MLOps workbench for ML pipelines with:
- **No vendor lock-in**: All services self-hosted or S3-compatible
- **Pipeline architecture**: [[#Pipeline Stages|collect → annotate → train → evaluate → infer]]
- **Cloud-portable**: Switch from local MinIO to B2/AWS by changing env vars

## Architecture

```
ml_workbench/
├── conf/                    # Hydra configuration (YAML)
│   ├── config.yaml          # Main entry point
│   ├── infrastructure/      # Services config (AIM, S3, Vault, etc.)
│   ├── pipeline/            # Per-pipeline settings
│   └── experiment/          # Experiment presets
├── data_transfer/           # Storage clients
│   ├── s3_client.py         # S3-compatible client (USE THIS)
│   └── b2_client.py         # Legacy B2 client (deprecated)
├── utils/                   # Utilities
│   ├── hydra_aim.py         # Hydra + AIM integration
│   ├── storage.py           # Storage helpers (uses Vault for creds)
│   ├── vault.py             # HashiCorp Vault integration
│   ├── manifest.py          # [[MANIFESTS|Dataset manifest system]]
│   └── knowledge_sync.py    # Knowledge base sync
├── pipelines/               # ML pipeline stages
│   ├── collect/             # Stage 1: Data gathering
│   ├── annotate/            # Stage 2: Auto-captioning
│   ├── train/               # Stage 3: Model training (LoRA, fine-tuning)
│   ├── evaluate/            # Stage 4: Benchmarks & metrics
│   └── infer/               # Stage 5: Inference & generation
├── knowledge/               # Knowledge base (Obsidian vault)
│   ├── collections/         # Data collection manifests
│   ├── annotations/         # Annotation manifests
│   ├── experiments/         # AIM experiment reports
│   ├── papers/              # Zotero papers
│   └── datasets/            # FiftyOne datasets
└── docker-compose.yml       # Main compose file
```

## Key Technologies

| Category | Technology | Notes |
|----------|------------|-------|
| Config | **[[#Hydra Configuration\|Hydra]]** | Hierarchical YAML config with CLI overrides |
| Workflow Orchestration | **[[#Prefect Workflows\|Prefect]]** | Self-hosted DAG pipelines with UI |
| Data Versioning | **DVC** | Git for data, tracks datasets/models |
| Experiment Tracking | **[[#AIM Integration\|AIM]]** | Self-hosted, replaces W&B |
| Model Registry | **AIM + S3** | Version, store, and promote models |
| Object Storage | **MinIO** → B2/S3 | S3-compatible, cloud-portable |
| Notebooks | **Marimo** | Reactive .py notebooks, NOT Jupyter |
| Secrets | **[[#Vault Secrets\|HashiCorp Vault]]** | ALL secrets go here |
| LLM Gateway | **LiteLLM** | Unified API for multiple LLMs |
| Annotation | **Label Studio** | Images/text labeling, S3-backed |
| Dataset Viz | **FiftyOne** | S3-backed for remote datasets |
| Knowledge Base | **[[#Knowledge Stack\|Khoj + Obsidian + Zotero]]** | AI search, notes, paper management |

## Common Patterns

### Hydra Configuration

```python
import hydra
from omegaconf import DictConfig

@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    # Access config: cfg.train.epochs, cfg.infrastructure.storage.endpoint
    pass
```

**Config files:**
- `conf/config.yaml` - Main entry point
- `conf/infrastructure/default.yaml` - Services configuration
- `conf/pipeline/*.yaml` - Per-pipeline settings
- `conf/experiment/*.yaml` - Experiment presets

### AIM Integration

```python
from utils import init_aim_from_hydra, AimCallback

run = init_aim_from_hydra(cfg)  # Auto-logs Hydra config
callback = AimCallback(run)
callback.on_epoch_end(epoch, {"loss": loss, "accuracy": acc})
run.close()
```

**Access AIM UI:** http://localhost:43800

### S3 Storage

```python
from utils import init_storage_from_hydra, upload_model, download_model

# From Hydra config (auto-fetches creds from Vault)
s3 = init_storage_from_hydra(cfg, bucket_type="models")
s3.upload_file(Path("model.pt"), "models/v1/model.pt")

# Or direct
from data_transfer import S3Client
client = S3Client()  # Uses S3_* env vars
```

**Switch storage providers:**
```bash
# Local (MinIO - default)
S3_ENDPOINT=http://minio:9000

# Production (Backblaze B2)
S3_ENDPOINT=https://s3.us-west-000.backblazeb2.com

# Production (AWS S3)
S3_ENDPOINT=https://s3.us-east-1.amazonaws.com
```

### Prefect Workflows

```python
from prefect import flow, task

@task(name="prepare-data", retries=2)
def prepare_data(config: dict):
    """Tasks handle individual steps with retry logic."""
    pass

@flow(name="training-pipeline", log_prints=True)
def training_flow(config: dict):
    """Flows orchestrate tasks into DAG pipelines."""
    data = prepare_data(config)
    return result

# Run locally
training_flow({"epochs": 10})
```

**Access Prefect UI:** http://localhost:4200

### Model Registry

```python
from utils import ModelRegistry, register_model, load_model

# Register a model after training
registry = ModelRegistry()
model_info = registry.register(
    model_path="./outputs/model.pt",
    name="blip-captioner-lora",
    version="1.0.0",
    metrics={"accuracy": 0.95, "loss": 0.05},
    aim_run=run,  # Link to AIM experiment
    stage="development"
)

# Load a model (auto-downloads from S3)
model_path = load_model("blip-captioner-lora", version="latest")

# Promote to production
registry.promote("blip-captioner-lora", "1.0.0", stage="production")
```

### Vault Secrets

**ALL credentials go in Vault:**

```python
from utils import get_secret, get_api_key, get_s3_credentials

# Get any secret
secret = get_secret("storage/s3")  # Returns dict
api_key = get_api_key("openai")    # Returns string

# S3 credentials (auto-falls back to env vars)
creds = get_s3_credentials()

# Store secrets
from utils import VaultClient
vault = VaultClient()
vault.put_secret("myapp/config", {"api_key": "xxx"})
```

**Vault UI:** http://localhost:8200 (token: `mlops-dev-token`)

### Dataset Manifests

See [[MANIFESTS|Dataset Manifest System]] for full documentation.

```python
from utils import create_collection_manifest, create_annotation_manifest

# Auto-created by pipelines, or create manually:
manifest = create_collection_manifest(
    output_dir="./data/my-dataset",
    source="reddit",
    source_url="https://reddit.com/r/earthporn",
    metadata={"subreddit": "earthporn", "limit": 100}
)

# Multiple annotations on same collection
annotation1 = create_annotation_manifest(
    collection_id=manifest["id"],
    annotation_type="caption",
    model="blip-base",
    output_dir="./data/annotated/blip"
)

annotation2 = create_annotation_manifest(
    collection_id=manifest["id"],
    annotation_type="caption",
    model="blip2-opt-2.7b",
    output_dir="./data/annotated/blip2"
)
```

## Pipeline Stages

### 1. Collect - Data Gathering

```bash
# Collect images from Reddit
docker-compose run --rm collect python -m pipelines.collect.collect \
    --subreddit earthporn --limit 100
```

**Creates:** `collection_manifest.json` with git traceability

### 2. Annotate - Auto-Captioning

```bash
# Caption collected images
docker-compose run --rm annotate python -m pipelines.annotate.caption \
    --input-dir ./data/collected/earthporn --model blip-base

# Create FiftyOne dataset
docker-compose run --rm annotate python -m pipelines.annotate.create_dataset \
    --input-dir ./data/collected/earthporn --name my-dataset
```

**Creates:** `annotation_manifest.json` linked to parent collection

**Available models:**
- `blip-base` - Fast, good quality
- `blip2-opt-2.7b` - Higher quality
- `git-base` - Microsoft GIT
- `vit-gpt2` - ViT + GPT-2

### 3. Train - Model Fine-Tuning

```bash
# Fine-tune captioner with LoRA
docker-compose run --rm train python -m pipelines.train.finetune \
    --dataset ./data/collected/earthporn --model blip-base --epochs 3

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

## Knowledge Stack

### Khoj (AI Search Assistant)

**URL:** http://localhost:42110

Indexes all your knowledge:
- Obsidian vault (this vault!)
- Zotero papers
- AIM experiments
- Dataset manifests

### Obsidian Sync

**CouchDB UI:** http://localhost:5984/_utils

Install "Self-hosted LiveSync" plugin:
```
URI: http://localhost:5984
Username: obsidian
Password: mlops-dev-password
Database: obsidian
```

### Zotero (Paper Management)

**URL:** http://localhost:8085

```python
import requests

# Add paper from arXiv
resp = requests.post("http://localhost:8085/api/papers", json={
    "url": "https://arxiv.org/abs/2303.08774"
})
```

### Knowledge Base Sync

```bash
# Sync all sources to markdown
python -m utils.knowledge_sync

# Sync only collections
python -m utils.knowledge_sync --source collections

# Watch for changes
python -m utils.knowledge_sync --watch --interval 60
```

This syncs:
- [[experiments/|AIM experiments]] → `knowledge/experiments/`
- [[collections/|Collections]] → `knowledge/collections/`
- [[annotations/|Annotations]] → `knowledge/annotations/`
- [[papers/|Papers]] → `knowledge/papers/`
- [[datasets/|Datasets]] → `knowledge/datasets/`

## Web UIs

| Service | URL | Credentials |
|---------|-----|-------------|
| MinIO Console | http://localhost:9001 | mlops-admin / mlops-dev-password |
| AIM | http://localhost:43800 | - |
| Prefect | http://localhost:4200 | - |
| Label Studio | http://localhost:8081 | S3-backed |
| FiftyOne | http://localhost:5151 | S3-backed |
| Khoj | http://localhost:42110 | admin@mlops.local / mlops-dev-password |
| Obsidian CouchDB | http://localhost:5984/_utils | obsidian / mlops-dev-password |
| Zotero | http://localhost:8085 | Paper management API |
| Vault | http://localhost:8200 | Token: mlops-dev-token |
| LiteLLM | http://localhost:4000 | - |

## Important Notes

1. **NO Jupyter** - Use Marimo notebooks (`.py` files, reactive, git-friendly)
2. **NO W&B** - Use AIM for experiment tracking
3. **NO hardcoded credentials** - ALL secrets in Vault
4. **S3Client over B2Client** - B2Client is legacy
5. **Hydra for all config** - Don't use config.py for new code
6. **DVC for data versioning** - Use `dvc.yaml` and `params.yaml`
7. **Prefect for orchestration** - All pipelines have `@flow` decorators
8. **Model Registry** - Register models with `register_model()` after training
9. **Dataset Manifests** - Track collections and annotations with [[MANIFESTS|manifest system]]
10. **Tiny CPU Models** - Use `+experiment=tiny_cpu` for low-resource testing

## Quick Reference

### Docker Commands

```bash
# Start all services
docker-compose up -d

# Run pipeline stage
docker-compose run --rm <stage> python -m pipelines.<stage>.<script>

# View logs
docker-compose logs -f <service>

# GPU support
docker-compose -f docker-compose.yml -f docker-compose.gpu.yml up -d
```

### Configuration Overrides

```bash
# Per-run overrides
python script.py train.epochs=10 train.lr=1e-4

# Multi-run sweeps
python script.py -m train.lr=1e-3,1e-4,1e-5

# Use experiment preset
python script.py +experiment=tiny_cpu
```

### Debugging

```bash
# Check service health
docker-compose ps

# Test S3 connection
docker-compose run --rm train python -c "from data_transfer import S3Client; print(S3Client().list_buckets())"

# View Hydra config
python -m pipelines.train.finetune --cfg job
```

## Related Documentation

- [[MANIFESTS|Dataset Manifest System]] - Track collections and annotations
- Architecture diagram - See README.md
- Testing guide - `tests/run_tests.py --help`

## External Links

- [Hydra Documentation](https://hydra.cc/)
- [Prefect Documentation](https://docs.prefect.io/)
- [AIM Documentation](https://aimstack.io/)
- [MinIO Documentation](https://min.io/docs/)
- [HashiCorp Vault](https://www.vaultproject.io/)

---

*Last updated: 2025-12-21*
*This file is auto-synced to the knowledge base for AI-powered search via Khoj*
