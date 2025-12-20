# Developer Guide

This guide is a quick reference for contributors who want a streamlined way to work with the MLOps Workbench without memorizing every Docker or DVC command.

## Prerequisites
- Docker & Docker Compose
- Python 3.10+
- [DVC](https://dvc.org/) installed locally (for `dvc` targets)
- Optional: `black` and `ruff` in your Python environment for formatting and linting

## Compose Profiles at a Glance
- **core** (default): storage, databases, secrets, orchestration (MinIO, MongoDB, Redis, Postgres, Vault, Aim, Prefect, LiteLLM).
- **tools**: visualization and knowledge tooling (FiftyOne, Label Studio, Khoj/Obsidian, Zotero) that add a few extra CPUs/GB of RAM for their UIs and databases.
- **pipeline**: short-lived pipeline stages (collect/annotate/train/evaluate/infer).
- **dev**: the development container with notebooks/CLIs preinstalled.
- **test**: integration test runner.

For a single-machine prototype, the defaults keep only the **core** profile running. Use `make infra` to start core, add tools with `make tools`, or explicitly opt into both via `make up PROFILES="core tools"`.

## Everyday Workflow
1. Discover available commands:
   ```bash
   make help
   ```
2. Build and start infrastructure services (core-only to stay light):
   ```bash
   make build
   make infra          # core infra only
   make tools          # add data tooling (optional)
   ```
3. Open a dev shell with the tooling profile:
   ```bash
   make dev-shell      # automatically enables core infra + dev profile
   ```
4. Stop everything when you're done:
   ```bash
   make down
   ```

## Single-Machine Tips
- Keep memory use down by sticking to the default **core** profile; start `make tools` only when you need data UIs.
- Expect the tools profile to add roughly:
  - FiftyOne: ~2 CPU / 4GB (UI + MongoDB connections)
  - Label Studio: ~2 CPU / 2GB
  - Khoj + Obsidian CouchDB + Zotero helpers: ~4 CPU / 3.5GB combined
- GPU runs on the same host can use `docker-compose.gpu.yml` by appending `-f docker-compose.gpu.yml` to the compose command (e.g., `COMPOSE="docker compose -f docker-compose.yml -f docker-compose.gpu.yml" make train`).
- If you want tools only while debugging, run `make tools` in another terminal and `make down` once you're finished.

## Running Pipeline Stages
Each stage accepts additional arguments via `ARGS` so you don't have to retype the full command.

```bash
# Collect Reddit images
make collect ARGS="--subreddit earthporn --limit 50"

# Auto-caption collected assets
make annotate ARGS="--input-dir ./data/collected --model blip-base"

# Fine-tune captioner
make train ARGS="--dataset ./data/collected --epochs 3"

# Calculate metrics
make evaluate ARGS="--predictions ./outputs/predictions.json"

# Generate images
make infer ARGS="--prompts 'A sunset over mountains'"
```

Common argument sets you can copy/paste:
- Collect: `ARGS="--subreddit earthporn --limit 50 --sort top"`
- Annotate: `ARGS="--input-dir ./data/collected --model blip-base --batch-size 8"`
- Train: `ARGS="--dataset ./data/collected --epochs 3 --learning-rate 5e-5"`

## DVC Shortcuts
- Run the entire pipeline: `make dvc-repro`
- Run a single stage: `make dvc-stage STAGE=train-captioner`
- Visualize the graph: `make dvc-dag`

## Testing
- Unit tests: `make tests`
- Integration & health checks:
  ```bash
  # Preflight: ensure core infra is up (`make infra`) and MINIO/MONGODB/VAULT URLs
  # match the defaults exposed by docker-compose

  make integration-tests ARGS="--quick"
  make integration-tests ARGS="--category minio"
  ```

## Code Quality
- Format with Black: `make format`
- Lint with Ruff: `make lint`

## Cleanup
Remove cached Python artifacts:
```bash
make clean
```
