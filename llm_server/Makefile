.PHONY: help start stop restart logs models status test ui

help:
	@echo "LLM Server Management"
	@echo "====================="
	@echo "make start       - Start main LLM server (70B)"
	@echo "make start-fast  - Start fast model (32B)"
	@echo "make start-all   - Start all models"
	@echo "make stop        - Stop all servers"
	@echo "make restart     - Restart servers"
	@echo "make logs        - View server logs"
	@echo "make status      - Check server status"
	@echo "make test        - Test API"
	@echo "make ui          - Start with web UI"
	@echo "make models      - List available models"

start:
	docker-compose up -d vllm
	@echo "Server starting on http://localhost:8000"
	@echo "Run 'make logs' to monitor startup"

start-fast:
	docker-compose --profile multi-model up -d vllm-fast
	@echo "Fast server starting on http://localhost:8001"

start-vision:
	docker-compose --profile vision up -d vllm-vision
	@echo "Vision server starting on http://localhost:8002"

start-all:
	docker-compose --profile multi-model --profile vision up -d
	@echo "All servers starting"

stop:
	docker-compose down

restart:
	docker-compose restart

logs:
	docker-compose logs -f vllm

status:
	@echo "=== Docker Containers ==="
	@docker-compose ps
	@echo ""
	@echo "=== GPU Usage ==="
	@nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv
	@echo ""
	@echo "=== Server Health ==="
	@curl -s http://localhost:8000/health || echo "Main server not responding"

test:
	@echo "Testing API..."
	@curl -s http://localhost:8000/v1/models | python3 -m json.tool
	@echo ""
	@curl -s http://localhost:8000/v1/chat/completions \
		-H "Content-Type: application/json" \
		-H "Authorization: Bearer your-secret-key-here" \
		-d '{"model":"meta-llama/Llama-3.1-70B-Instruct","messages":[{"role":"user","content":"Hi"}]}' \
		| python3 -m json.tool

ui:
	docker-compose --profile ui up -d
	@echo "Web UI available at http://localhost:3000"

models:
	@echo "Available models:"
	@curl -s http://localhost:8000/v1/models 2>/dev/null | python3 -c "import sys,json; print('\n'.join([m['id'] for m in json.load(sys.stdin)['data']]))" || echo "Server not running"

clean:
	docker-compose down -v
	@echo "Cleared all data. Model cache preserved."

gpu:
	watch -n 1 nvidia-smi
