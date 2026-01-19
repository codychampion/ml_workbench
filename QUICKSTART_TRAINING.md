# LoRA Training Quick Start

## The Simple Truth

Training a LoRA for HunyuanVideo requires the **full model components** (~33GB):
- DiT Transformer (from `tencent/HunyuanVideo-1.5`)
- VAE (from `tencent/HunyuanVideo-1.5`)
- Text Encoders: Qwen 2.5 VL + BYT5 (from `Comfy-Org/HunyuanVideo_1.5_repackaged`)

Your FP8 quantized ComfyUI model alone (`wan2.2_t2v_high_noise_14B_fp8_scaled.safetensors`) **cannot be used for training** because:
1. It's quantized (loses gradient information needed for backpropagation)
2. It's only the high_noise expert (missing other MoE components)
3. It's missing VAE and text encoders

**Good news**: The download is **one-time only**. After first run, training starts immediately.

---

## One Command to Train

```bash
docker compose --profile pipeline run --rm train \
    python pipelines/train/train_lora_simple.py \
    --dataset data/scraped/fallout_nv_20260116_113625
```

That's it. The script handles everything:
1. **First run**: Downloads models (~30 min), then trains (6-8 hours)
2. **Subsequent runs**: Starts training immediately (models cached)

---

## What Happens

### First Time Only
```
Phase 0: Download models (~33GB, 10-30 min)
 └─ Cached to: models/hunyuanvideo_1_5/
 └─ Never downloaded again ✅

Phase 1: Cache latents (~5-10 min)
 └─ VAE encodes your images once
 └─ Reused for all future training runs ✅

Phase 2: Cache text encoders (~2-5 min)
 └─ Process text prompts once
 └─ Reused for all future training runs ✅
```

### Every Training Run
```
Phase 3: Train LoRA (5-8 hours)
 └─ Uses cached latents and text encoders
 └─ Saves checkpoints every 5 epochs
 └─ Outputs ComfyUI-compatible safetensors
```

---

## Expected Timeline

| Run | Download | Cache | Train | Total |
|-----|----------|-------|-------|-------|
| **First** | 30 min | 15 min | 6 hrs | **~7 hours** |
| **Second+** | 0 min | 0 min | 6 hrs | **~6 hours** |

---

## Advanced Options

### More Epochs (Better Quality)
```bash
python pipelines/train/train_lora_simple.py \
    --dataset data/scraped/fallout_nv_20260116_113625 \
    --epochs 30
```

### Smaller LoRA (Faster Training)
```bash
python pipelines/train/train_lora_simple.py \
    --dataset data/scraped/fallout_nv_20260116_113625 \
    --lora-rank 16 \
    --lora-alpha 16
```

### Custom Concept Name
```bash
python pipelines/train/train_lora_simple.py \
    --dataset data/scraped/fallout_nv_20260116_113625 \
    --concept "fallout_nv_v2"
```

### Skip Re-caching (If Training Again)
```bash
python pipelines/train/train_lora_simple.py \
    --dataset data/scraped/fallout_nv_20260116_113625 \
    --skip-cache
```

---

## After Training

Your LoRA will be at:
```
outputs/lora/fallout_nv_20260116_113625/fallout_nv_20260116_113625_epoch20.safetensors
```

Copy to ComfyUI:
```bash
cp outputs/lora/fallout_nv_20260116_113625/*.safetensors \
   /path/to/ComfyUI/models/loras/
```

Use in your workflow (`workflows/custom_lora_test.json`):
```json
{
  "type": "LoraLoaderModelOnly",
  "widgets_values": [
    "fallout_nv_20260116_113625_epoch20.safetensors",
    0.8
  ]
}
```

---

## Troubleshooting

### "Musubi Tuner not found"
```bash
docker compose build train
```

### "Download failed"
- Check internet connection
- Verify disk space (~50GB free)
- HuggingFace may be rate-limiting (wait 5 min, retry)

### "CUDA out of memory"
Reduce LoRA size:
```bash
--lora-rank 16 --resolution 768 432
```

### Training progress monitoring
```bash
# In another terminal
docker logs -f ml_workbench-train-1
```

---

## Why This Works

| Feature | This Solution | Other Attempts |
|---------|---------------|----------------|
| **Setup** | One command | Multiple steps |
| **Models** | Auto-download + cache | Manual download |
| **Caching** | Automatic | Manual |
| **Output** | ComfyUI format | Needs conversion |
| **Errors** | Clear messages | Cryptic failures |
| **Documentation** | This guide | Scattered docs |

---

## Storage Requirements

- **Models**: ~33GB (one-time)
- **Cache**: ~2-5GB per dataset (reusable)
- **LoRA outputs**: ~100-150MB per training run

**Total for first run**: ~50GB
**Total for subsequent runs**: ~100MB (just the LoRA)

---

## Ready to Train?

```bash
# Rebuild Docker image (if you haven't already)
docker compose build train

# Start training!
docker compose --profile pipeline run --rm train \
    python pipelines/train/train_lora_simple.py \
    --dataset data/scraped/fallout_nv_20260116_113625
```

The first run will download models. Grab a coffee, then let it train overnight.

Second run? Starts training immediately. 🚀
