# MLOps Workbench

![Status](https://img.shields.io/badge/status-active-16a34a)
![Docker](https://img.shields.io/badge/Docker-compose-2496ed)
![Python](https://img.shields.io/badge/Python-ML%20pipelines-3776ab)
![Local LLM](https://img.shields.io/badge/local%20LLM-vLLM-7c3aed)
![MLOps](https://img.shields.io/badge/MLOps-local--first-111827)

A self-hosted machine-learning workbench for running local AI experiments end to end: data collection, annotation, training, evaluation, inference, model tracking, local LLM serving, and research notes in one Dockerized environment.

This repo is designed for the messy middle between notebooks and production: the place where experiments need repeatable infrastructure, useful observability, durable artifacts, and enough structure that future-you can understand what happened.

## What this gives you

| Capability | Services / tools |
|---|---|
| Object storage | MinIO-compatible S3 storage |
| Local LLM serving | vLLM with OpenAI-compatible API |
| Experiment workbench | JupyterLab, Hydra configs, pipeline containers |
| Dataset inspection | FiftyOne + MongoDB |
| Annotation | Label Studio |
| Tracking | AIM experiment tracking |
| Chat over notes | Khoj + Postgres over the knowledge vault |
| API surface | FastAPI server |
| Registry patterns | Docker registry + model registration scripts |

## Why it matters

Most ML side projects either stay as fragile notebooks or become overbuilt infrastructure. This workbench sits between those extremes. It keeps the workflow local-first, reproducible, and modular while still supporting serious components: GPU-backed LLM serving, dataset labeling, model evaluation, experiment metadata, and a research knowledge vault.

The result is a practical lab environment for testing ideas quickly without losing the operational pieces that make ML work trustworthy.

## Architecture at a glance

```text
Plan → Collect → Annotate → Train → Evaluate → Infer → Track → Register
 │       │          │         │        │        │       │        │
 │       │          │         │        │        │       │        └─ model registry scripts
 │       │          │         │        │        │       └───────── AIM + run summaries
 │       │          │         │        │        └───────────────── generation / inference jobs
 │       │          │         │        └────────────────────────── benchmark pipelines
 │       │          │         └─────────────────────────────────── LoRA training pipelines
 │       │          └───────────────────────────────────────────── BLIP captions + Label Studio
 │       └──────────────────────────────────────────────────────── Reddit / HF / CUAD collectors
 └──────────────────────────────────────────────────────────────── Obsidian knowledge vault
```

## Quick start

```bash
# Start core storage
cp .env.example .env
docker compose up -d

# Start local LLM server
# See README_LLM.md for model and GPU configuration.
docker compose --profile llm up -d

# Start JupyterLab with GPU access
docker compose --profile jupyter up -d

# Add chat over the knowledge vault
docker compose --profile chat up -d

# Run a pipeline stage
docker compose --profile pipeline run --rm train \
  python -m pipelines.train.train_lora
```

See [README_LLM.md](README_LLM.md) for the local LLM server setup and 5090-oriented configuration.

## Compose profiles

| Profile | Command | Services |
|---|---|---|
| default | `docker compose up -d` | MinIO S3 storage |
| `llm` | `docker compose --profile llm up -d` | vLLM local LLM server |
| `jupyter` | `docker compose --profile jupyter up -d` | JupyterLab + GPU |
| `api` | `docker compose --profile api up -d` | FastAPI server |
| `redteam` | `docker compose --profile redteam up -d` | Red-team daemon |
| `chat` | `docker compose --profile chat up -d` | Khoj + Postgres |
| `cv_ui` | `docker compose --profile cv_ui up -d` | FiftyOne + MongoDB |
| `labeling` | `docker compose --profile labeling up -d` | Label Studio |
| `tracking` | `docker compose --profile tracking up -d` | AIM |
| `registry` | `docker compose --profile registry up -d` | Docker registry |
| `pipeline` | `docker compose --profile pipeline run --rm <stage>` | One-off pipeline jobs |
| `dev` | `docker compose --profile dev up -d` | Dev shell + Jupyter |
| `test` | `docker compose --profile test run --rm test` | Test runner |

## Local service URLs

| Service | URL | Profile |
|---|---|---|
| MinIO Console | http://localhost:9001 | default |
| LLM API | http://localhost:8000 | `llm` |
| JupyterLab | http://localhost:8888 | `jupyter`, `dev` |
| API Server | http://localhost:8080 | `api` |
| Khoj | http://localhost:42110 | `chat` |
| FiftyOne | http://localhost:5151 | `cv_ui` |
| Label Studio | http://localhost:8081 | `labeling` |
| AIM | http://localhost:43800 | `tracking` |

## Pipeline examples

### Collect images from Reddit

```bash
python -m pipelines.collect.collect_reddit \
  --subreddit earthporn \
  --limit 100 \
  --sort top \
  --time week
```

### Collect from Hugging Face

```bash
python -m pipelines.collect.collect_hf \
  --dataset laion/laion400m \
  --split train \
  --limit 50
```

### Collect CUAD contract data

```bash
python -m pipelines.collect.collect_cuad \
  --split train \
  --limit 1000
```

### Run full pipeline stages through Docker

```bash
# Collect
docker compose --profile pipeline run --rm collect \
  python -m pipelines.collect.collect_reddit --subreddit earthporn --limit 100

# Annotate
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

## Knowledge vault

Open `./knowledge/` in Obsidian. The vault is organized around experiment planning, paper notes, run summaries, and model registry records.

```text
knowledge/
├── papers/notes/          # Paper notes
├── experiments/plans/     # Experiment plans
├── experiments/runs/      # Run summaries
└── models/registry/       # Model registry records
```

Khoj can index the vault so the workbench has a chat surface over research notes, experimental plans, and run history.

## Configuration

Copy `.env.example` to `.env` and fill in the services you plan to run.

Key variables include:

| Variable | Purpose |
|---|---|
| `S3_ENDPOINT` | MinIO / S3-compatible storage endpoint |
| `S3_ACCESS_KEY` | Object storage access key |
| `S3_SECRET_KEY` | Object storage secret |
| `OPENAI_API_KEY` | Optional external model access for chat workflows |
| `ANTHROPIC_API_KEY` | Optional external model access for chat workflows |
| `VLLM_API_KEY` | Local vLLM API key |
| `HF_TOKEN` | Hugging Face token for gated models |

## Utility scripts

```bash
python scripts/new_experiment.py "My experiment"          # Create an experiment plan
python scripts/ingest_aim_run.py --run <hash>             # Ingest an AIM run
python scripts/register_model_image.py <model_id>         # Register a model artifact
python scripts/labelstudio_sync.py export --dataset my-ds # Export Label Studio data
```

## Project structure

```text
ml_workbench/
├── conf/              # Hydra configuration
├── data_transfer/     # S3 client utilities
├── docker/            # Docker configs
├── knowledge/         # Obsidian knowledge vault
├── models/            # Model artifacts
├── outputs/           # Pipeline outputs and AIM logs
├── pipelines/         # Collection, annotation, training, evaluation, inference
├── scripts/           # Glue scripts and operational utilities
├── services/          # Service-specific code
├── tests/             # Integration tests
└── utils/             # Shared utilities
```

## Good fit for

This workbench is useful for local-first ML experimentation, data and model pipeline prototyping, AI safety or red-team workflows, multimodal dataset curation, retrieval and knowledge-work experiments, and testing the operational shape of an ML idea before turning it into a more formal service.
