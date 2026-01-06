.PHONY: help setup start stop restart logs status clean test

# Colors
RED=\033[0;31m
GREEN=\033[0;32m
YELLOW=\033[1;33m
NC=\033[0m # No Color

help:
	@echo "$(GREEN)ML Workbench - Docker Services$(NC)"
	@echo "================================"
	@echo ""
	@echo "$(YELLOW)Core Services:$(NC)"
	@echo "  make start          - Start LLM server only"
	@echo "  make start-all      - Start all infrastructure (LLM + Redis + Postgres)"
	@echo "  make start-dev      - Start development environment (+ Jupyter)"
	@echo "  make start-full     - Start everything (multi-model + vision)"
	@echo ""
	@echo "$(YELLOW)Management:$(NC)"
	@echo "  make stop           - Stop all services"
	@echo "  make restart        - Restart all services"
	@echo "  make logs           - View LLM server logs"
	@echo "  make logs-all       - View all service logs"
	@echo "  make status         - Check service status"
	@echo ""
	@echo "$(YELLOW)Testing:$(NC)"
	@echo "  make test           - Test LLM API"
	@echo "  make test-gpu       - Check GPU availability"
	@echo ""
	@echo "$(YELLOW)Cleanup:$(NC)"
	@echo "  make clean          - Stop and remove containers"
	@echo "  make clean-all      - Remove containers and volumes"
	@echo ""
	@echo "$(YELLOW)Monitoring:$(NC)"
	@echo "  make gpu            - Monitor GPU usage"
	@echo "  make top            - Show container resource usage"

setup:
	@echo "$(GREEN)Setting up ML Workbench...$(NC)"
	@if [ ! -f .env ]; then \
		cp .env.example .env; \
		echo "$(YELLOW)Created .env from .env.example$(NC)"; \
		echo "$(RED)⚠️  Please edit .env and set your API keys!$(NC)"; \
	fi
	@mkdir -p outputs data models aim_repo
	@echo "$(GREEN)✓ Setup complete$(NC)"

start:
	@echo "$(GREEN)Starting LLM server...$(NC)"
	docker compose up -d llm
	@echo "$(YELLOW)Server starting on http://localhost:8000$(NC)"
	@echo "Run 'make logs' to monitor startup"

start-all:
	@echo "$(GREEN)Starting all infrastructure services...$(NC)"
	docker compose --profile infrastructure up -d
	@echo "$(GREEN)✓ Services started:$(NC)"
	@echo "  - LLM Server: http://localhost:8000"
	@echo "  - Redis: localhost:6379"
	@echo "  - PostgreSQL: localhost:5432"

start-dev:
	@echo "$(GREEN)Starting development environment...$(NC)"
	docker compose --profile dev --profile infrastructure up -d
	@echo "$(GREEN)✓ Development environment ready:$(NC)"
	@echo "  - LLM Server: http://localhost:8000"
	@echo "  - Jupyter Lab: http://localhost:8888"
	@docker compose logs jupyter | grep "http://127.0.0.1:8888/lab?token=" || true

start-full:
	@echo "$(GREEN)Starting full environment...$(NC)"
	docker compose --profile multi-model --profile vision --profile infrastructure up -d
	@echo "$(GREEN)✓ All services started:$(NC)"
	@echo "  - Main LLM (70B): http://localhost:8000"
	@echo "  - Fast LLM (32B): http://localhost:8001"
	@echo "  - Vision LLM: http://localhost:8002"

stop:
	@echo "$(YELLOW)Stopping all services...$(NC)"
	docker compose down
	@echo "$(GREEN)✓ All services stopped$(NC)"

restart:
	@echo "$(YELLOW)Restarting services...$(NC)"
	docker compose restart
	@echo "$(GREEN)✓ Services restarted$(NC)"

logs:
	docker compose logs -f llm

logs-all:
	docker compose logs -f

status:
	@echo "$(GREEN)=== Docker Containers ===$(NC)"
	@docker compose ps
	@echo ""
	@echo "$(GREEN)=== GPU Status ===$(NC)"
	@nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv
	@echo ""
	@echo "$(GREEN)=== LLM Server Health ===$(NC)"
	@curl -s http://localhost:8000/health 2>/dev/null && echo "$(GREEN)✓ Healthy$(NC)" || echo "$(RED)✗ Not responding$(NC)"

test:
	@echo "$(GREEN)Testing LLM API...$(NC)"
	@echo ""
	@echo "$(YELLOW)Available models:$(NC)"
	@curl -s http://localhost:8000/v1/models | python3 -m json.tool | grep '"id"' || echo "$(RED)Server not responding$(NC)"
	@echo ""
	@echo "$(YELLOW)Test completion:$(NC)"
	@curl -s http://localhost:8000/v1/chat/completions \
		-H "Content-Type: application/json" \
		-H "Authorization: Bearer ml-workbench-key" \
		-d '{"model":"llm","messages":[{"role":"user","content":"Say hello"}],"max_tokens":20}' \
		| python3 -m json.tool || echo "$(RED)Request failed$(NC)"

test-gpu:
	@echo "$(GREEN)Checking GPU availability...$(NC)"
	docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi

gpu:
	@watch -n 1 nvidia-smi

top:
	docker stats

clean:
	@echo "$(YELLOW)Stopping and removing containers...$(NC)"
	docker compose down
	@echo "$(GREEN)✓ Cleanup complete$(NC)"

clean-all:
	@echo "$(RED)⚠️  This will delete all data (Redis, PostgreSQL, etc.)$(NC)"
	@read -p "Are you sure? [y/N] " -n 1 -r; \
	echo; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		docker compose down -v; \
		echo "$(GREEN)✓ All containers and volumes removed$(NC)"; \
	else \
		echo "$(YELLOW)Cancelled$(NC)"; \
	fi

# Quick shortcuts
llm: start
dev: start-dev
full: start-full
