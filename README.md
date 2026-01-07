# MLOps Workbench

Self-hosted ML pipeline workbench with profile-based services.

## Quick Start

```bash
# Start core (MinIO only)
docker compose up -d

# Start local LLM server (Llama 3.1 70B on 5090)
docker compose --profile llm up -d

# Start Jupyter Lab with GPU
docker compose --profile jupyter up -d

# Add Khoj chat interface
docker compose --profile chat up -d

# Run a pipeline
docker compose --profile pipeline run --rm train python -m pipelines.train.train_lora
```

See [README_LLM.md](README_LLM.md) for detailed LLM server configuration.

## Profiles

| Profile | Command | Services |
|---------|---------|----------|
| (default) | `docker compose up -d` | MinIO (S3 storage) |
| llm | `--profile llm` | vLLM server (70B) |
| jupyter | `--profile jupyter` | JupyterLab + GPU |
| api | `--profile api` | FastAPI server |
| redteam | `--profile redteam` | Red team daemon |
| chat | `--profile chat` | Khoj + Postgres |
| cv_ui | `--profile cv_ui` | FiftyOne + MongoDB |
| labeling | `--profile labeling` | Label Studio |
| tracking | `--profile tracking` | AIM |
| registry | `--profile registry` | Docker Registry |
| pipeline | `--profile pipeline run --rm <stage>` | Pipeline containers |
| dev | `--profile dev` | Dev shell + Jupyter |
| test | `--profile test run --rm test` | Tests |

## Web UIs

| Service | URL | Profile |
|---------|-----|---------|
| MinIO Console | http://localhost:9001 | (default) |
| LLM API | http://localhost:8000 | llm |
| Jupyter Lab | http://localhost:8888 | jupyter, dev |
| API Server | http://localhost:8080 | api |
| Khoj | http://localhost:42110 | chat |
| FiftyOne | http://localhost:5151 | cv_ui |
| Label Studio | http://localhost:8081 | labeling |
| AIM | http://localhost:43800 | tracking |

## Workflow

```
1. Plan          → knowledge/experiments/plans/
2. Collect       → python -m pipelines.collect.collect_{reddit,hf,cuad}
3. Annotate      → python -m pipelines.annotate.caption
4. Train         → python -m pipelines.train.train_lora
5. Evaluate      → python -m pipelines.evaluate.benchmark
6. Infer         → python -m pipelines.infer.run_generation
7. Ingest run    → python scripts/ingest_aim_run.py
8. Register model→ python scripts/register_model_image.py <model_id>
```

## Examples

### Reddit Image Collection
```bash
# Collect 100 images from r/earthporn
python -m pipelines.collect.collect_reddit --subreddit earthporn --limit 100 --sort top --time week

# Multiple subreddits
python -m pipelines.collect.collect_reddit --subreddit "earthporn,cityporn" --limit 50
```

### HuggingFace Dataset Collection
```bash
# Download 50 samples from LAION
python -m pipelines.collect.collect_hf --dataset laion/laion400m --split train --limit 50

# With Hydra config
python -m pipelines.collect.collect_hf --hydra pipeline=collect_hf
```

### CUAD Contract Dataset
```bash
# Download 1000 contract samples
python -m pipelines.collect.collect_cuad --split train --limit 1000

# Full dataset (84,325 samples)
python -m pipelines.collect.collect_cuad --split train --limit -1

# With Hydra config for MAKER experiment
python -m pipelines.collect.collect_cuad --hydra pipeline=collect_cuad
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
# Collect - Reddit
docker compose --profile pipeline run --rm collect \
  python -m pipelines.collect.collect_reddit --subreddit earthporn --limit 100

# Collect - HuggingFace
docker compose --profile pipeline run --rm collect \
  python -m pipelines.collect.collect_hf --dataset laion/laion400m --limit 50

# Collect - CUAD (Legal Contracts)
docker compose --profile pipeline run --rm collect \
  python -m pipelines.collect.collect_cuad --split train --limit 100

# Annotate - Caption images
docker compose --profile pipeline run --rm annotate \
  python -m pipelines.annotate.caption --input ./data/collected --model blip-base

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
