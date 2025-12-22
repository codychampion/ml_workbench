---
type: guide
title: "MLOps Workbench - AI Assistant Context"
tags: [guide, ai-assistant]
---

# MLOps Workbench - AI Assistant Context

Self-hosted ML pipeline workbench with profile-based architecture.

## Architecture

```
ml_workbench/
ƒ"oƒ"?ƒ"? conf/              # Hydra config
ƒ"oƒ"?ƒ"? data_transfer/     # S3Client
ƒ"oƒ"?ƒ"? knowledge/         # Obsidian vault (open in Obsidian)
ƒ"oƒ"?ƒ"? pipelines/         # collect ƒ+' annotate ƒ+' train ƒ+' evaluate ƒ+' infer
ƒ"oƒ"?ƒ"? scripts/           # Glue scripts
ƒ"oƒ"?ƒ"? services/          # FiftyOne
ƒ""ƒ"?ƒ"? docker-compose.yml # Profile-based services
```

## Key Technologies

| Component | Technology |
|-----------|------------|
| Storage | MinIO (S3-compatible) |
| Config | Hydra |
| Tracking | AIM |
| Knowledge | Obsidian + Khoj |
| CV UI | FiftyOne |
| Labeling | Label Studio |
| Models | Docker images ƒ+' local registry |

## Docker Profiles

```bash
docker compose up -d                          # MinIO only
docker compose --profile chat up -d           # Khoj + Postgres
docker compose --profile cv_ui up -d          # FiftyOne + MongoDB
docker compose --profile labeling up -d       # Label Studio
docker compose --profile tracking up -d       # AIM
docker compose --profile registry up -d       # Docker Registry
docker compose --profile pipeline run --rm train <cmd>  # Pipelines
```

## Common Patterns

### S3 Storage
```python
from utils.storage import get_s3_client
s3 = get_s3_client("mlops-data")
s3.upload_file(Path("model.pt"), "models/v1/model.pt")
```

### AIM Tracking
```python
from aim import Run
run = Run(repo="./outputs/aim", experiment="my-exp")
run.track(loss, name="loss", epoch=epoch)
run.close()
```

### Secrets (env vars first)
```python
from utils.vault import get_s3_credentials, get_api_key
creds = get_s3_credentials()  # Uses S3_* env vars
key = get_api_key("openai")   # Uses OPENAI_API_KEY env var
```

## Web UIs

| Service | URL | Profile |
|---------|-----|---------|
| MinIO Console | http://localhost:9001 | (default) |
| Khoj | http://localhost:42110 | chat |
| FiftyOne | http://localhost:5151 | cv_ui |
| Label Studio | http://localhost:8081 | labeling |
| AIM | http://localhost:43800 | tracking |

## Knowledge Vault

Open `./knowledge/` in Obsidian. Start Khoj for AI chat:
```bash
docker compose --profile chat up -d
# http://localhost:42110
```

Structure:
- `papers/notes/` - Paper notes
- `experiments/plans/` - Experiment plans
- `experiments/runs/` - Run summaries
- `models/registry/` - Model registry

## Testing

```bash
docker compose --profile test run --rm test
```
