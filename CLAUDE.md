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
| Experiment Tracking | **AIM** | Self-hosted, replaces W&B |
| Object Storage | **MinIO** → B2/S3 | S3-compatible, cloud-portable |
| Notebooks | **Marimo** | Reactive .py notebooks, NOT Jupyter |
| Distributed Storage | **JuiceFS** | Redis-backed |
| Secrets | **HashiCorp Vault** | ALL secrets go here |
| LLM Gateway | **LiteLLM** | Unified API for multiple LLMs |
| Annotation | **Label Studio**, **CVAT** | Images/text and video |
| Dataset Viz | **FiftyOne**, **Spotlight** | |
| Knowledge Base | **SiYuan** | Zettelkasten, papers, experiment notes |

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
| Label Studio | http://localhost:8081 | - |
| CVAT | http://localhost:8082 | - |
| Spotlight | http://localhost:8083 | - |
| FiftyOne | http://localhost:5151 | - |
| SiYuan | http://localhost:6806 | Code: mlops-dev |
| ComfyUI | http://localhost:8188 | - |
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
6. **SiYuan for notes** - Use for papers, experiment notes, knowledge base

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
