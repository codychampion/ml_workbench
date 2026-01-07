# Local LLM Server (RTX 5090 Optimized)

High-performance local LLM server using vLLM with OpenAI-compatible API.

## Quick Start

```bash
# 1. Start main server (Llama 3.1 70B)
docker-compose up -d vllm

# 2. Wait for model download (first time: ~140GB)
docker-compose logs -f vllm

# 3. Test
curl http://localhost:8000/v1/models
```

## Model Configurations

### Default: Llama 3.1 70B Instruct
- **VRAM**: ~28GB (fits perfectly on 5090)
- **Context**: 8K tokens
- **Speed**: ~30-50 tokens/sec
- **Best for**: General purpose, coding, reasoning

```bash
docker-compose up -d vllm
```

### Fast: Qwen 2.5 32B Instruct
- **VRAM**: ~14GB (leaves room for other tasks)
- **Context**: 32K tokens
- **Speed**: ~80-120 tokens/sec
- **Best for**: Fast responses, long context

```bash
docker-compose --profile multi-model up -d vllm-fast
```

### Vision: Qwen2-VL 7B
- **VRAM**: ~10GB
- **Context**: 4K tokens
- **Speed**: ~100+ tokens/sec
- **Best for**: Image understanding, multimodal

```bash
docker-compose --profile vision up -d vllm-vision
```

### All Models (Multi-Server)
Run multiple models simultaneously (32GB total):
```bash
# 70B (28GB) + 7B Vision (10GB) = 38GB (needs offloading)
# Better: 32B (14GB) + 7B Vision (10GB) = 24GB ✓
docker-compose --profile multi-model --profile vision up -d
```

## Recommended Model Combinations for 5090

**Option A: Single Large Model**
- Llama 3.1 70B @ 28GB - Maximum intelligence

**Option B: Balanced Dual**
- Qwen 2.5 32B @ 14GB - Fast general purpose
- Qwen2-VL 7B @ 10GB - Vision tasks
- Total: 24GB, 8GB free for other work

**Option C: Speed Optimized**
- Qwen 2.5 32B @ 14GB - Primary
- Llama 3.1 8B @ 5GB - Fast assistant
- Total: 19GB, 13GB free

## API Usage

### Python (OpenAI SDK)

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="your-secret-key-here"
)

# Chat completion
response = client.chat.completions.create(
    model="meta-llama/Llama-3.1-70B-Instruct",
    messages=[
        {"role": "user", "content": "Explain adversarial ML attacks"}
    ],
    temperature=0.7,
    max_tokens=500
)

print(response.choices[0].message.content)
```

### Vision Model

```python
# Using Qwen2-VL (port 8002)
client = OpenAI(
    base_url="http://localhost:8002/v1",
    api_key="your-secret-key-here"
)

response = client.chat.completions.create(
    model="Qwen/Qwen2-VL-7B-Instruct",
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": "What's in this image?"},
            {"type": "image_url", "image_url": {"url": "file:///path/to/image.jpg"}}
        ]
    }]
)
```

### Streaming

```python
stream = client.chat.completions.create(
    model="meta-llama/Llama-3.1-70B-Instruct",
    messages=[{"role": "user", "content": "Write a poem"}],
    stream=True
)

for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")
```

### curl

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-secret-key-here" \
  -d '{
    "model": "meta-llama/Llama-3.1-70B-Instruct",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

## Integration with ML Workbench

```python
# Use in your red-teaming workflows
from openai import OpenAI

llm = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="your-secret-key-here"
)

def generate_adversarial_prompts(target_model_info: str) -> list:
    """Use LLM to generate creative attack prompts."""
    response = llm.chat.completions.create(
        model="meta-llama/Llama-3.1-70B-Instruct",
        messages=[{
            "role": "user",
            "content": f"Generate 10 adversarial test cases for: {target_model_info}"
        }]
    )
    return response.choices[0].message.content

def analyze_vulnerability_report(report: dict) -> str:
    """LLM-powered vulnerability analysis."""
    response = llm.chat.completions.create(
        model="meta-llama/Llama-3.1-70B-Instruct",
        messages=[{
            "role": "user",
            "content": f"Analyze this ML security report and suggest fixes:\n{report}"
        }]
    )
    return response.choices[0].message.content
```

## Web UI

Access ChatGPT-like interface:

```bash
# Start with UI
docker-compose --profile ui up -d

# Open browser
open http://localhost:3000
```

## Performance Tuning

### Max Throughput (Llama 70B)
```yaml
command: >
  --model meta-llama/Llama-3.1-70B-Instruct
  --tensor-parallel-size 1
  --max-model-len 8192
  --gpu-memory-utilization 0.95
  --max-num-seqs 256          # High batch size
  --enable-prefix-caching      # Cache system prompts
```

### Low Latency
```yaml
command: >
  --model meta-llama/Llama-3.1-70B-Instruct
  --max-model-len 4096         # Shorter context
  --gpu-memory-utilization 0.8
  --max-num-seqs 32            # Lower batch
```

### Long Context (32K+)
```yaml
command: >
  --model Qwen/Qwen2.5-32B-Instruct
  --max-model-len 32768
  --gpu-memory-utilization 0.9
  --enable-chunked-prefill     # Handle long prompts
```

## Alternative Models for 5090

**Largest Possible:**
- `meta-llama/Llama-3.1-70B-Instruct` (28GB) ✓ Fits
- `Qwen/Qwen2.5-72B-Instruct` (30GB) ✓ Fits
- `mistralai/Mixtral-8x22B-Instruct-v0.1` (needs quantization)

**Quantized Large Models:**
- `TheBloke/Llama-2-70B-AWQ` (16GB, faster)
- `TheBloke/Mixtral-8x7B-AWQ` (12GB)

**Best Value:**
- `Qwen/Qwen2.5-32B-Instruct` (14GB, 32K context)
- `meta-llama/Llama-3.1-8B-Instruct` (5GB, very fast)

**Specialized:**
- `Qwen/Qwen2.5-Coder-32B-Instruct` (14GB, coding)
- `deepseek-ai/deepseek-coder-33b-instruct` (16GB, coding)

## Monitoring

```bash
# Watch GPU usage
watch -n 1 nvidia-smi

# View logs
docker-compose logs -f vllm

# Check API health
curl http://localhost:8000/health
```

## Troubleshooting

**Out of memory:**
```yaml
# Reduce memory usage
--gpu-memory-utilization 0.8  # Lower from 0.9
--max-model-len 4096          # Shorter context
```

**Slow downloads:**
```bash
# Set HuggingFace mirror (if in China/slow region)
export HF_ENDPOINT=https://hf-mirror.com
```

**Model not found:**
```bash
# Pre-download model
huggingface-cli download meta-llama/Llama-3.1-70B-Instruct
```

## Cost Savings

**vs OpenAI:**
- GPT-4: $30/1M input tokens
- Your server: $0/unlimited tokens
- Break-even: ~1M tokens (~750K words)

**vs Cloud GPU:**
- AWS g5.xlarge: $1.01/hour
- Your 5090: $0/hour (already owned)

**Power cost:**
- 5090 TDP: 575W
- Full load: ~$0.08/hour (@ $0.15/kWh)
- 24/7 month: ~$58/month
- Still cheaper than cloud!

## Next Steps

1. **Start server**: `docker-compose up -d vllm`
2. **Test API**: `curl http://localhost:8000/v1/models`
3. **Integrate**: Use OpenAI SDK with `base_url="http://localhost:8000/v1"`
4. **Profit**: Free unlimited LLM access
