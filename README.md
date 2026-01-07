# MLOps Workbench

Self-hosted ML pipeline workbench with profile-based services.

## Quick Start

```bash
# Start core (MinIO only)
docker compose up -d

# Start local LLM server (Llama 3.1 70B on 5090)
docker compose --profile llm up -d

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
| chat | `--profile chat` | Khoj + Postgres |
| labeling | `--profile labeling` | Label Studio |
| tracking | `--profile tracking` | AIM |
| registry | `--profile registry` | Docker Registry |
| pipeline | `--profile pipeline run --rm <stage>` | Pipeline containers |
| test | `--profile test run --rm test` | Tests |

## Web UIs

| Service | URL | Profile |
|---------|-----|---------|
| MinIO Console | http://localhost:9001 | (default) |
| LLM API | http://localhost:8000 | llm |
| Khoj | http://localhost:42110 | chat |
| Label Studio | http://localhost:8081 | labeling |
| AIM | http://localhost:43800 | tracking |

## Workflow

```
1. Plan          → knowledge/experiments/plans/
2. Collect       → python -m pipelines.collect.collect_{reddit,hf}
3. Train         → python -m pipelines.train.train_lora
4. Evaluate      → python -m pipelines.evaluate.benchmark
5. Infer         → python -m pipelines.infer.run_generation
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
```

## Pipeline Stages

```bash
# Collect - Reddit
docker compose --profile pipeline run --rm collect \
  python -m pipelines.collect.collect_reddit --subreddit earthporn --limit 100

# Collect - HuggingFace
docker compose --profile pipeline run --rm collect \
  python -m pipelines.collect.collect_hf --dataset laion/laion400m --limit 50

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
