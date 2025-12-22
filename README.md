# MLOps Workbench

Self-hosted ML pipeline workbench with profile-based services.

## Quick Start

```bash
# Start core (MinIO only)
docker compose up -d

# Add Khoj chat interface
docker compose --profile chat up -d

# Run a pipeline
docker compose --profile pipeline run --rm train python -m pipelines.train.train_lora
```

## Profiles

| Profile | Command | Services |
|---------|---------|----------|
| (default) | `docker compose up -d` | MinIO (S3 storage) |
| chat | `--profile chat` | Khoj + Postgres |
| cv_ui | `--profile cv_ui` | FiftyOne + MongoDB |
| labeling | `--profile labeling` | Label Studio |
| tracking | `--profile tracking` | AIM |
| registry | `--profile registry` | Docker Registry |
| pipeline | `--profile pipeline run --rm <stage>` | Pipeline containers |
| dev | `--profile dev` | Dev shell |
| test | `--profile test run --rm test` | Tests |

## Web UIs

| Service | URL | Profile |
|---------|-----|---------|
| MinIO Console | http://localhost:9001 | (default) |
| Khoj | http://localhost:42110 | chat |
| FiftyOne | http://localhost:5151 | cv_ui |
| Label Studio | http://localhost:8081 | labeling |
| AIM | http://localhost:43800 | tracking |

## Workflow

```
1. Plan          → knowledge/experiments/plans/
2. Collect       → docker compose --profile pipeline run --rm collect ...
3. Annotate      → docker compose --profile pipeline run --rm annotate ...
4. Train         → docker compose --profile pipeline run --rm train ...
5. Evaluate      → docker compose --profile pipeline run --rm evaluate ...
6. Infer         → docker compose --profile pipeline run --rm infer ...
7. Ingest run    → python scripts/ingest_aim_run.py
8. Register model→ python scripts/register_model_image.py <model_id>
```

## Knowledge Vault

Open `./knowledge/` in Obsidian. Structure:
- `papers/notes/` - Paper notes
- `experiments/plans/` - Experiment plans
- `experiments/runs/` - Run summaries (auto-generated)
- `models/registry/` - Model registry

Khoj indexes the vault for AI chat.

## Configuration

Copy `.env.example` to `.env`. Key vars:
- `S3_ENDPOINT`, `S3_ACCESS_KEY`, `S3_SECRET_KEY` - MinIO/S3
- `OPENAI_API_KEY`, `ANTHROPIC_API_KEY` - For Khoj chat

## Scripts

```bash
python scripts/new_experiment.py "My experiment"    # Create plan
python scripts/ingest_aim_run.py --run <hash>       # Ingest AIM run
python scripts/register_model_image.py <model_id>   # Register model
python scripts/labelstudio_sync.py export --dataset my-ds  # Export to LS
```

## Pipeline Stages

```bash
# Collect
docker compose --profile pipeline run --rm collect \
  python -m pipelines.collect.collect --subreddit earthporn

# Annotate
docker compose --profile pipeline run --rm annotate \
  python -m pipelines.annotate.auto_caption --input ./data/collected

# Train
docker compose --profile pipeline run --rm train \
  python -m pipelines.train.train_lora --epochs 10

# Evaluate
docker compose --profile pipeline run --rm evaluate \
  python -m pipelines.evaluate.benchmark --model ./models/lora

# Infer
docker compose --profile pipeline run --rm infer \
  python -m pipelines.infer.run_generation --prompt "A sunset"
```

## Project Structure

```
ml_workbench/
├── conf/              # Hydra config
├── data_transfer/     # S3 client
├── docker/            # Docker configs
├── knowledge/         # Obsidian vault
├── models/            # Model artifacts
├── outputs/           # Pipeline outputs, AIM logs
├── pipelines/         # Pipeline stages
├── scripts/           # Glue scripts
├── services/          # FiftyOne service
├── tests/             # Integration tests
└── utils/             # Utilities
```
