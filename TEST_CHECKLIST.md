# ML Workbench GPU Testing Checklist

Run through these tests on your GPU machine and record any issues.

## Environment Setup

### 1. Prerequisites Check
```bash
# Check Docker
docker --version
# Expected: Docker version 20+

# Check Docker Compose
docker compose version
# Expected: Docker Compose version v2+

# Check NVIDIA GPU
nvidia-smi
# Expected: Should show your RTX 5090 with 32GB VRAM

# Check NVIDIA Docker runtime
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
# Expected: Should show GPU info inside container
```

**Issues:**
```
[Record any issues here]
```

---

## Core Services

### 2. Start MinIO (Core Storage)
```bash
cd /path/to/ml_workbench
docker compose up -d
docker compose ps
```

**Expected:** MinIO container running and healthy
**Issues:**
```
[Record any issues here]
```

### 3. Test MinIO Access
```bash
# Open browser to http://localhost:9001
# Login: mlops-admin / mlops-dev-password
# Check: Buckets mlops-data, mlops-models, mlops-outputs exist
```

**Issues:**
```
[Record any issues here]
```

### 4. Start LLM Server (Optional - if you want to test)
```bash
docker compose --profile llm up -d
# Wait for model download (first time: ~140GB for Llama 3.1 70B)
docker compose logs -f llm

# Test API when ready
curl http://localhost:8000/v1/models
```

**Expected:** Model loaded, API responds
**Issues:**
```
[Record any issues here]
```

---

## Pipeline: Collect

### 5. Build Collect Pipeline
```bash
docker compose build collect
```

**Expected:** Build succeeds with pyarrow and datasets installed
**Issues:**
```
[Record any issues here]
```

### 6. Test Reddit Collection
```bash
docker compose --profile pipeline run --rm collect \
  python -m pipelines.collect.collect_reddit --subreddit earthporn --limit 5
```

**Expected:** Downloads 5 images to ./data/collected/reddit/
**Issues:**
```
[Record any issues here]
```

### 7. Test HuggingFace Collection
```bash
docker compose --profile pipeline run --rm collect \
  python -m pipelines.collect.collect_hf --dataset "nlphuji/flickr30k" --limit 5
```

**Expected:** Downloads 5 samples to ./data/collected/hf/
**Issues:**
```
[Record any issues here]
```

---

## Pipeline: Train

### 8. Build Train Pipeline
```bash
docker compose build train
```

**Expected:** Build succeeds
**Issues:**
```
[Record any issues here]
```

### 9. Test Training (Quick Test)
```bash
docker compose --profile pipeline run --rm train \
  python -m pipelines.train.train_lora --help
```

**Expected:** Shows help message, no errors
**Issues:**
```
[Record any issues here]
```

### 10. GPU Access in Train Container
```bash
docker compose --profile pipeline run --rm train nvidia-smi
```

**Expected:** Shows GPU info (RTX 5090)
**Issues:**
```
[Record any issues here]
```

---

## Pipeline: Evaluate

### 11. Build Evaluate Pipeline
```bash
docker compose build evaluate
```

**Expected:** Build succeeds
**Issues:**
```
[Record any issues here]
```

### 12. Test Evaluation
```bash
docker compose --profile pipeline run --rm evaluate \
  python -m pipelines.evaluate.benchmark --help
```

**Expected:** Shows help message
**Issues:**
```
[Record any issues here]
```

---

## Pipeline: Infer

### 13. Build Infer Pipeline
```bash
docker compose build infer
```

**Expected:** Build succeeds
**Issues:**
```
[Record any issues here]
```

### 14. Test Inference
```bash
docker compose --profile pipeline run --rm infer \
  python -m pipelines.infer.run_generation --help
```

**Expected:** Shows help message
**Issues:**
```
[Record any issues here]
```

---

## Adversarial Pipeline (IMPORTANT!)

### 15. Test Adversarial Patch Generation
```bash
# Check adversarial pipeline exists
ls -la pipelines/adversarial/

# Test main script
docker compose --profile pipeline run --rm infer \
  python -m pipelines.adversarial.generate_adv_patch --help
```

**Expected:** Shows help for adversarial patch generation
**Issues:**
```
[Record any issues here]
```

### 16. Test Physical Transforms
```bash
docker compose --profile pipeline run --rm infer \
  python -c "from pipelines.adversarial.physical_transforms import *; print('Transforms loaded OK')"
```

**Expected:** Prints "Transforms loaded OK"
**Issues:**
```
[Record any issues here]
```

### 17. Test Target Models
```bash
docker compose --profile pipeline run --rm infer \
  python -c "from pipelines.adversarial.target_models import *; print('Target models loaded OK')"
```

**Expected:** Prints "Target models loaded OK"
**Issues:**
```
[Record any issues here]
```

### 18. Quick Adversarial Patch Test (if models available)
```bash
# Only if you have a target model ready
docker compose --profile pipeline run --rm infer \
  python -m pipelines.adversarial.generate_adv_patch \
    --target-model yolov5 \
    --target-class person \
    --epochs 2 \
    --output ./outputs/test_patch.png
```

**Expected:** Generates a test adversarial patch
**Issues:**
```
[Record any issues here]
```

---

## Optional Services

### 19. Khoj Chat (Optional)
```bash
docker compose --profile chat up -d
# Wait for startup
docker compose logs -f khoj

# Open browser to http://localhost:42110
# Login: admin@mlops.local / mlops-dev-password
```

**Expected:** Khoj UI loads
**Issues:**
```
[Record any issues here]
```

### 20. AIM Tracking (Optional)
```bash
docker compose --profile tracking up -d

# Open browser to http://localhost:43800
```

**Expected:** AIM UI loads
**Issues:**
```
[Record any issues here]
```

---

## Integration Tests

### 21. Full Pipeline Test (Collect → Train → Infer)
```bash
# 1. Collect data
docker compose --profile pipeline run --rm collect \
  python -m pipelines.collect.collect_reddit --subreddit earthporn --limit 10

# 2. Check data exists
ls -la ./data/collected/reddit/

# 3. Try a simple training run (may take time depending on model)
# [Skip if no training setup yet]
```

**Issues:**
```
[Record any issues here]
```

---

## Cleanup

### 22. Stop All Services
```bash
docker compose --profile llm --profile chat --profile tracking down
docker compose down
```

**Expected:** All containers stop cleanly
**Issues:**
```
[Record any issues here]
```

---

## Summary

**Total Issues Found:** ___
**Critical Issues:** ___
**Minor Issues:** ___

**Most Important Issues to Fix:**
1.
2.
3.

**System Info:**
- OS: _______________
- Docker Version: _______________
- GPU: _______________
- CUDA Version: _______________

**Notes:**
```
[Any additional observations or comments]
```

---

## Quick Issue Report Format

When reporting issues back, use this format:

```
ISSUE: [Step number and name]
ERROR: [Exact error message]
EXPECTED: [What should have happened]
LOGS: [Relevant logs if any]
```

Example:
```
ISSUE: Step 6 - Test Reddit Collection
ERROR: ModuleNotFoundError: No module named 'gallery_dl'
EXPECTED: Should download 5 images
LOGS: [paste relevant logs]
```
