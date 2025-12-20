COMPOSE ?= docker-compose
PYTHON ?= python
# Default to the lightest single-machine footprint; opt into tools with PROFILES="core tools"
PROFILES ?= core
PROFILE_FLAGS := $(addprefix --profile ,$(PROFILES))
PIPELINE_FLAGS := --profile core --profile pipeline

.PHONY: help build infra up tools down logs dev-up dev-shell collect annotate train evaluate infer dvc-repro dvc-stage dvc-dag tests integration-tests format lint clean

help: ## Show this help message
	@awk 'BEGIN {FS = ":.*##"; printf "Available targets:\n"} /^[a-zA-Z0-9_-]+:.*##/ { printf "  %-18s %s\n", $$1, $$2 }' $(MAKEFILE_LIST)

build: ## Build all Docker images
	$(COMPOSE) build

infra: ## Start only the core infrastructure profile (storage, db, secrets, orchestration)
	$(COMPOSE) --profile core up -d

up: ## Start services for the selected profiles (defaults to core only; PROFILES="core tools" to add data tools)
	$(COMPOSE) $(PROFILE_FLAGS) up -d

tools: ## Start data tooling services (FiftyOne, Label Studio, Khoj/Obsidian, Zotero)
	$(COMPOSE) --profile core --profile tools up -d

down: ## Stop all services and remove containers
	$(COMPOSE) down

logs: ## Tail compose logs (set SERVICES="service1 service2" to filter)
	$(COMPOSE) $(PROFILE_FLAGS) logs -f --tail=200 $(SERVICES)

dev-up: ## Start the dev container with core infrastructure
	$(COMPOSE) --profile core --profile dev up -d dev

dev-shell: dev-up ## Open a shell inside the dev container
	$(COMPOSE) exec dev bash

collect: ## Run the data collection stage (set ARGS="--subreddit earthporn")
	$(COMPOSE) $(PIPELINE_FLAGS) run --rm collect python -m pipelines.collect.collect $(ARGS)

annotate: ## Run the annotation stage (set ARGS="--input-dir ./data/collected")
	$(COMPOSE) $(PIPELINE_FLAGS) run --rm annotate python -m pipelines.annotate.caption $(ARGS)

train: ## Run training stage (set ARGS="--dataset ./data/collected")
	$(COMPOSE) $(PIPELINE_FLAGS) run --rm train python -m pipelines.train.finetune $(ARGS)

evaluate: ## Run evaluation stage (set ARGS="--predictions ./outputs/predictions.json")
	$(COMPOSE) $(PIPELINE_FLAGS) run --rm evaluate python -m pipelines.evaluate.metrics $(ARGS)

infer: ## Run inference stage (set ARGS="--prompts 'A sunset'")
	$(COMPOSE) $(PIPELINE_FLAGS) run --rm infer python -m pipelines.infer.run_generation $(ARGS)

dvc-repro: ## Run the full DVC pipeline
	dvc repro

dvc-stage: ## Run a specific DVC stage (set STAGE="train-captioner")
	dvc repro $(STAGE)

dvc-dag: ## Show the DVC pipeline graph
	dvc dag

tests: ## Run Python unit tests locally
	$(PYTHON) -m pytest -q

integration-tests: ## Run service health and integration checks (set ARGS="--quick")
	$(PYTHON) tests/run_tests.py $(ARGS)

format: ## Format Python code with black (requires dependency)
	$(PYTHON) -m black .

lint: ## Lint Python code with ruff (requires dependency)
	ruff .

clean: ## Remove Python cache and local artifacts
	find . -type d -name "__pycache__" -prune -exec rm -rf {} +
	rm -rf .pytest_cache .ruff_cache
