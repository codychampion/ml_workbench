# MLOps Workbench - AI Assistant Context

> Quick context file for AI assistants (Claude, GPT, etc.) to understand this codebase.

## Project Overview

Self-hosted MLOps workbench for ML pipelines with:
- **No vendor lock-in**: All services self-hosted or S3-compatible
- **Pipeline architecture**: collect → annotate → train → evaluate → infer
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
│   └── vault.py             # HashiCorp Vault integration
├── pipelines/               # ML pipeline stages
│   ├── collect/             # Stage 1: Data gathering
│   ├── annotate/            # Stage 2: Auto-captioning
│   ├── train/               # Stage 3: Model training (LoRA, fine-tuning)
│   ├── evaluate/            # Stage 4: Benchmarks & metrics
│   └── infer/               # Stage 5: Inference & generation
├── notebooks/               # Marimo notebooks (NOT Jupyter)
├── services/                # Supporting services
├── docker-compose.yml       # Main compose file
└── docker-compose.gpu.yml   # GPU override
```

## Key Technologies

| Category | Technology | Notes |
|----------|------------|-------|
| Config | **Hydra** | Hierarchical YAML config with CLI overrides |
| Workflow Orchestration | **Prefect** | Self-hosted DAG pipelines with UI |
| Data Versioning | **DVC** | Git for data, tracks datasets/models |
| Experiment Tracking | **AIM** | Self-hosted, replaces W&B |
| Model Registry | **AIM + S3** | Version, store, and promote models |
| Data Quality | **Great Expectations** | Data validation and quality checks |
| Object Storage | **MinIO** → B2/S3 | S3-compatible, cloud-portable |
| Notebooks | **Marimo** | Reactive .py notebooks, NOT Jupyter |
| Distributed Storage | **JuiceFS** | Redis-backed |
| Secrets | **HashiCorp Vault** | ALL secrets go here |
| LLM Gateway | **LiteLLM** | Unified API for multiple LLMs |
| Annotation | **Label Studio**, **CVAT** | Images/text and video, S3-backed |
| Dataset Viz | **FiftyOne**, **Spotlight** | S3-backed for remote datasets |
| Image Generation | **ComfyUI** | Full ComfyUI with S3 model sync |
| Knowledge Base | **Khoj + Obsidian + Zotero** | AI search, notes, paper management |

## Common Patterns

### Configuration (Hydra)
```python
import hydra
from omegaconf import DictConfig

@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    # Access config: cfg.train.epochs, cfg.infrastructure.storage.endpoint
    pass
```

### Experiment Tracking (AIM)
```python
from utils import init_aim_from_hydra, AimCallback

run = init_aim_from_hydra(cfg)  # Auto-logs Hydra config
callback = AimCallback(run)
callback.on_epoch_end(epoch, {"loss": loss, "accuracy": acc})
run.close()
```

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

### Workflow Orchestration (Prefect)
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
    # ... more tasks
    return result

# Run locally
training_flow({"epochs": 10})

# Or deploy to Prefect server
# prefect deployment build ./train.py:training_flow -n "prod"
# prefect deployment apply training_flow-deployment.yaml
```

All pipeline scripts (train_lora.py, run_generation.py, collect.py, etc.) are
decorated with `@flow` and `@task` for Prefect orchestration.

### Model Registry (AIM + S3)
```python
from utils import ModelRegistry, register_model, load_model, list_models

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

# List all models
for model in list_models():
    print(f"{model.name}@{model.latest_version}")
```

### Data Validation (Great Expectations)
```python
# Via REST API (http://localhost:8084)
import requests

# Validate a dataset
response = requests.post("http://localhost:8084/api/validate", json={
    "datasource": "minio_data",
    "asset": "datasets/train.csv",
    "suite": "image_dataset"
})
print(response.json()["success"])  # True/False

# Create expectation suite
requests.post("http://localhost:8084/api/expectations", json={
    "name": "my_dataset",
    "expectations": [
        {"type": "expect_column_to_exist", "column": "filepath"},
        {"type": "expect_column_values_to_not_be_null", "column": "caption"}
    ]
})
```

### Vault Secrets (ALL credentials go here)
```python
from utils import get_secret, get_api_key, get_s3_credentials, VaultClient

# Get any secret
secret = get_secret("storage/s3")  # Returns dict
api_key = get_api_key("openai")    # Returns string

# S3 credentials (auto-falls back to env vars)
creds = get_s3_credentials()
print(creds.endpoint, creds.access_key)

# Store secrets
vault = VaultClient()
vault.put_secret("myapp/config", {"api_key": "xxx", "db_url": "yyy"})

# Initialize default secrets structure
from utils import init_vault_secrets
init_vault_secrets()  # Creates storage/s3, api_keys/*, databases/*
```

### Environment Variables (Storage)
```bash
# Local (MinIO - default)
S3_ENDPOINT=http://minio:9000
S3_ACCESS_KEY=mlops-admin
S3_SECRET_KEY=mlops-dev-password

# Production (Backblaze B2)
S3_ENDPOINT=https://s3.us-west-000.backblazeb2.com
S3_ACCESS_KEY=<key-id>
S3_SECRET_KEY=<app-key>

# Production (AWS S3)
S3_ENDPOINT=https://s3.us-east-1.amazonaws.com
```

## Docker Commands

```bash
# Start all services
docker-compose up -d

# Run pipeline stages
docker-compose run --rm collect python -m pipelines.collect.collect --help
docker-compose run --rm train python -m pipelines.train.train_lora --help
docker-compose run --rm --profile pipeline train <command>

# GPU support
docker-compose -f docker-compose.yml -f docker-compose.gpu.yml up -d
```

## Web UIs

| Service | URL | Credentials |
|---------|-----|-------------|
| MinIO Console | http://localhost:9001 | mlops-admin / mlops-dev-password |
| AIM | http://localhost:43800 | - |
| Prefect | http://localhost:4200 | - |
| Label Studio | http://localhost:8081 | S3-backed |
| CVAT | http://localhost:8082 | S3-backed |
| Spotlight | http://localhost:8083 | - |
| Great Expectations | http://localhost:8084 | - |
| FiftyOne | http://localhost:5151 | S3-backed |
| Khoj | http://localhost:42110 | admin@mlops.local / mlops-dev-password |
| Obsidian CouchDB | http://localhost:5984/_utils | obsidian / mlops-dev-password |
| Zotero | http://localhost:8085 | Paper management API |
| ComfyUI | http://localhost:8188 | S3 model sync |
| Vault | http://localhost:8200 | Token: mlops-dev-token |
| LiteLLM | http://localhost:4000 | - |

## File Naming Conventions

- Dockerfiles: `pipelines/<stage>/Dockerfile`
- Config: `conf/<category>/<name>.yaml`
- Notebooks: `notebooks/<name>.py` (Marimo format)
- Python modules use snake_case

## Important Notes

1. **NO Jupyter** - Use Marimo notebooks (`.py` files, reactive, git-friendly)
2. **NO W&B** - Use AIM for experiment tracking
3. **NO hardcoded credentials** - ALL secrets in Vault (utils/vault.py)
4. **S3Client over B2Client** - B2Client is legacy
5. **Hydra for all config** - Don't use config.py for new code
6. **Knowledge Stack** - Khoj (AI search), Obsidian (notes), Zotero (papers)
7. **DVC for data versioning** - Use `dvc.yaml` and `params.yaml` for data pipelines
8. **Prefect for orchestration** - All pipelines have `@flow` decorators
9. **Model Registry** - Register models with `register_model()` after training
10. **Great Expectations** - Validate datasets before training at http://localhost:8084
11. **Tiny CPU Models** - Use `+experiment=tiny_cpu` for low-resource testing

## S3 Integration

All annotation and visualization tools connect to MinIO:
- **Label Studio**: Import/export datasets from `s3://mlops-data/`
- **CVAT**: Import videos from S3, export annotations to S3
- **FiftyOne**: Load datasets directly from S3 URLs
- **ComfyUI**: Auto-syncs models from `s3://mlops-models/comfyui/`

To use S3 storage in annotation tools, configure cloud storage in their respective UIs pointing to `http://minio:9000`.

## Quick Debugging

```bash
# Check service health
docker-compose ps

# View logs
docker-compose logs -f <service>

# Enter container
docker-compose exec <service> bash

# Test S3 connection
docker-compose run --rm train python -c "from data_transfer import S3Client; print(S3Client().list_buckets())"
```

## Adding New Pipeline Stage

1. Create `pipelines/<stage>/` directory
2. Add Dockerfile based on existing ones (include boto3, hvac, hydra-core, omegaconf)
3. Add service to `docker-compose.yml` with S3_* and VAULT_* env vars
4. Add pipeline config in `conf/pipeline/<stage>.yaml`
5. Copy `conf/` and `utils/` in Dockerfile

## Secrets Management (Vault)

All secrets should be stored in Vault at `secret/mlops/`:

```
secret/mlops/
├── storage/
│   ├── s3          # MinIO/S3 credentials
│   └── b2          # Backblaze B2 credentials
├── api_keys/
│   ├── openai      # OpenAI API key
│   ├── anthropic   # Anthropic API key
│   └── huggingface # HuggingFace token
└── databases/
    ├── mongodb     # MongoDB connection URL
    └── redis       # Redis connection URL
```

Initialize with: `python -c "from utils import init_vault_secrets; init_vault_secrets()"`

## Modifying Configuration

- **Per-run overrides**: `python script.py train.epochs=10 train.lr=1e-4`
- **Multi-run sweeps**: `python script.py -m train.lr=1e-3,1e-4,1e-5`
- **New experiment preset**: Create `conf/experiment/<name>.yaml`

## Knowledge Base (Khoj + Obsidian + Zotero)

The knowledge stack provides AI-powered search across notes, papers, and experiments:

### Khoj (AI Search Assistant)
```python
# Khoj indexes Obsidian vault, Zotero papers, and AIM experiments
# Access at http://localhost:42110

# Configure data sources in Khoj UI:
# - Obsidian: /data/obsidian (read-only mount)
# - Zotero: /data/zotero (paper metadata)
# - Experiments: /data/outputs (AIM exports)
```

### Obsidian Sync
```bash
# Obsidian uses CouchDB for LiveSync across devices
# CouchDB UI: http://localhost:5984/_utils

# In Obsidian app, install "Self-hosted LiveSync" plugin
# Configure with:
#   URI: http://localhost:5984
#   Username: obsidian
#   Password: mlops-dev-password
#   Database: obsidian
```

### Zotero (Paper Management)
```python
import requests

# Add paper from URL (extracts metadata automatically)
resp = requests.post("http://localhost:8085/api/papers", json={
    "url": "https://arxiv.org/abs/2303.08774"
})

# List papers
papers = requests.get("http://localhost:8085/api/papers").json()

# Export as BibTeX
bibtex = requests.get("http://localhost:8085/api/export/bibtex").text

# Export as Markdown (for Obsidian)
md = requests.get("http://localhost:8085/api/export/markdown").text
```

## AIM Report Ingestion

Export AIM experiments to the knowledge base for AI-powered search:

```bash
# Export all experiments to Obsidian-compatible markdown
python -m utils.aim_ingestion

# Export specific run
python -m utils.aim_ingestion --run abc123

# Export last 7 days
python -m utils.aim_ingestion --since 7d

# Watch for new experiments (daemon mode)
python -m utils.aim_ingestion --watch --interval 60
```

Exported experiments appear in `knowledge/experiments/` and are indexed by Khoj.

## Git Hooks (PR Summaries)

Track code changes with AI-generated summaries:

```bash
# Install the post-merge hook
ln -sf ../../hooks/post-merge .git/hooks/post-merge

# Generate AI summary of recent changes
python -m hooks.summarize_pr HEAD~5..HEAD

# Summarize a GitHub PR
python -m hooks.summarize_pr --pr 123

# Summaries are saved to knowledge/git-summaries/
```

## Tiny CPU Models

For testing on low-resource machines (small laptops, CI):

```bash
# Use tiny model preset for all pipelines
python -m pipelines.annotate.auto_caption +experiment=tiny_cpu
python -m pipelines.infer.run_generation +experiment=tiny_cpu
```

### Available Tiny Models

| Task | Model | Params | RAM |
|------|-------|--------|-----|
| Captioning | `tiny-git` (microsoft/git-base) | ~180M | ~800MB |
| Captioning | `tiny-vit-gpt2` | ~240M | ~1GB |
| Generation | `tiny-sd` (small-stable-diffusion-v0) | ~430M | ~2GB |
| Classification | `convnext-tiny` | ~28M | ~120MB |

### Resource Requirements

- Captioning only: ~1-2GB RAM
- Generation only: ~2-4GB RAM
- Full pipeline: ~4-6GB RAM

## Testing

Comprehensive test suite for verifying all services and integrations.

### Quick Start

```bash
# Install test dependencies
pip install -r tests/requirements.txt

# Check service status
python tests/run_tests.py --status

# Run quick health checks
python tests/run_tests.py --quick

# Run all integration tests
python tests/run_tests.py

# Run specific test category
python tests/run_tests.py --category minio
python tests/run_tests.py --category knowledge
python tests/run_tests.py --category mlops
```

### Test Categories

| Category | Tests | Description |
|----------|-------|-------------|
| `health` | Service health checks | Verify all services are running |
| `minio` | S3/MinIO integration | Bucket operations, uploads, downloads |
| `knowledge` | Khoj, CouchDB, Zotero | Knowledge stack functionality |
| `mlops` | AIM, Prefect, Vault, etc. | MLOps service integrations |

### Running with pytest directly

```bash
# All integration tests
pytest tests/integration/ -v -m integration

# Specific test file
pytest tests/integration/test_services_health.py -v

# Skip slow tests
pytest tests/integration/ -v -m "not slow"

# Generate HTML report
pytest tests/integration/ --html=report.html
```

### Environment Variables

```bash
SERVICE_HOST=localhost    # Service host (default: localhost)
```
