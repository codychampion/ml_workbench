# Local LLM Server (5090 Optimized)

## Quick Start

```bash
# Start LLM server (Llama 3.1 70B)
docker compose --profile llm up -d

# Check logs
docker compose logs -f llm

# Test API
curl http://localhost:8000/v1/models
```

## Usage

### Python (OpenAI SDK)

```python
from openai import OpenAI

llm = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="ml-workbench-key"  # From .env VLLM_API_KEY
)

response = llm.chat.completions.create(
    model="llm",
    messages=[{"role": "user", "content": "Explain adversarial attacks"}]
)
print(response.choices[0].message.content)
```

### From Inside Docker Network

Other services can access LLM at `http://llm:8000/v1`:

```python
# In pipeline containers
llm = OpenAI(
    base_url="http://llm:8000/v1",
    api_key=os.getenv("VLLM_API_KEY")
)
```

## Profiles

**Single Model (Default):**
```bash
docker compose --profile llm up -d
# Llama 3.1 70B on port 8000
```

**Multi-Model:**
```bash
docker compose --profile llm --profile llm-multi up -d
# 70B on port 8000
# Qwen 32B on port 8001
```

**With Vision:**
```bash
docker compose --profile llm --profile llm-vision up -d
# 70B on port 8000
# Qwen2-VL 7B on port 8002
```

## Configuration (.env)

```bash
# Model selection
LLM_MODEL=meta-llama/Llama-3.1-70B-Instruct  # or Qwen/Qwen2.5-72B-Instruct

# API key
VLLM_API_KEY=ml-workbench-key

# Performance
MAX_CONTEXT=8192      # Context window
GPU_MEM_UTIL=0.9      # GPU memory utilization (0.7-0.95)

# Cache
HF_CACHE_DIR=~/.cache/huggingface
HF_TOKEN=             # For gated models
```

## Model Options for 5090

| Model | VRAM | Speed | Context | Profile |
|-------|------|-------|---------|---------|
| Llama 3.1 70B | 28GB | 40 tok/s | 8K | `llm` |
| Qwen 2.5 72B | 30GB | 45 tok/s | 8K | `llm` |
| Qwen 2.5 32B | 14GB | 100 tok/s | 32K | `llm-multi` |
| Qwen2-VL 7B | 10GB | 120 tok/s | 4K | `llm-vision` |

## Integration Examples

### Generate Adversarial Prompts

```python
from openai import OpenAI

llm = OpenAI(base_url="http://localhost:8000/v1", api_key="ml-workbench-key")

def generate_attack_prompts(model_type: str):
    response = llm.chat.completions.create(
        model="llm",
        messages=[{
            "role": "user",
            "content": f"Generate 10 adversarial test cases for {model_type} model"
        }]
    )
    return response.choices[0].message.content
```

### Analyze Vulnerability Report

```python
def analyze_vuln_report(report: dict):
    response = llm.chat.completions.create(
        model="llm",
        messages=[{
            "role": "user",
            "content": f"Analyze this security report: {json.dumps(report)}"
        }]
    )
    return response.choices[0].message.content
```

### Caption Generation

```python
def generate_caption(image_description: str):
    response = llm.chat.completions.create(
        model="llm",
        messages=[{
            "role": "user",
            "content": f"Generate a caption for: {image_description}"
        }],
        max_tokens=100
    )
    return response.choices[0].message.content
```

## Performance Tips

**Maximize throughput:**
```bash
GPU_MEM_UTIL=0.95
MAX_CONTEXT=4096  # Shorter context = more batch slots
```

**Low latency:**
```bash
GPU_MEM_UTIL=0.8
MAX_CONTEXT=8192
```

**Long documents:**
```bash
LLM_MODEL=Qwen/Qwen2.5-32B-Instruct
MAX_CONTEXT=32768
```

## Monitoring

```bash
# Check health
curl http://localhost:8000/health

# View metrics
docker compose exec llm curl localhost:8000/metrics

# GPU usage
watch nvidia-smi
```

## Cost Comparison

**vs OpenAI GPT-4:**
- OpenAI: $30 per 1M tokens
- Local: $0 unlimited
- Power: ~$0.08/hour @ full load

**vs Cloud GPU:**
- AWS g5.xlarge: $1.01/hour
- Local 5090: $0/hour

**Break-even:** ~1M tokens (~1 week of moderate use)

## Troubleshooting

**Out of memory:**
```bash
# Lower GPU utilization
GPU_MEM_UTIL=0.8

# Shorter context
MAX_CONTEXT=4096

# Smaller model
LLM_MODEL=Qwen/Qwen2.5-32B-Instruct
```

**Slow first request:**
- Model downloads first time (~140GB for 70B)
- Wait 5-15 minutes for download + loading
- Check logs: `docker compose logs -f llm`

**Port conflict:**
```bash
# Change port in docker-compose.yml
ports: ["8080:8000"]  # Use 8080 instead
```

## Advanced: Custom Models

Edit `docker-compose.yml`:

```yaml
llm:
  command: >
    --model deepseek-ai/deepseek-coder-33b-instruct
    --max-model-len 16384
    --gpu-memory-utilization 0.85
```

## Next Steps

1. Start server: `docker compose --profile llm up -d`
2. Test API: `curl http://localhost:8000/v1/models`
3. Use in Python: See examples above
4. Integrate with pipelines: Set `OPENAI_API_BASE=http://llm:8000/v1`

See `llm_server/` directory for standalone deployment docs.
