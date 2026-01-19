# HunyuanVideo LoRA Training with Musubi Tuner - Corrected Guide

## Based on Official Musubi Tuner Documentation

This guide follows the **actual** Musubi Tuner documentation:
- [HunyuanVideo 1.5 Docs](https://github.com/kohya-ss/musubi-tuner/blob/main/docs/hunyuan_video_1_5.md)
- [Dataset Config](https://github.com/kohya-ss/musubi-tuner/blob/main/docs/dataset_config.md)

## Prerequisites

- Docker with GPU support
- RTX 5090 (24GB+ VRAM required)
- Your dataset of images with captions (Fallout NV images ready!)
- ComfyUI workflow (verified compatibility with `LoraLoaderModelOnly`)

## Step 1: Rebuild Docker Image

```bash
docker compose build train
```

This installs:
- Musubi Tuner from GitHub
- All required dependencies
- accelerate, safetensors, etc.

**Time**: ~10-15 minutes

## Step 2: Run Training

```bash
docker compose --profile pipeline run --rm train \
    python pipelines/train/train_hunyuan_musubi.py \
    --dataset data/scraped/fallout_nv_20260116_113625 \
    --concept "fallout_nv" \
    --epochs 20 \
    --lora-rank 32 \
    --lora-alpha 32 \
    --resolution 960 544
```

### First Run - Model Download

The script will automatically download ~33GB of models from `Comfy-Org/HunyuanVideo_repackaged`:
- DiT transformer (720p T2V)
- VAE
- Qwen 2.5 VL text encoder
- BYT5 tokenizer

**Download time**: 10-30 minutes (one-time, then cached)

To skip download on subsequent runs:
```bash
--skip-download
```

### Training Process

**Phase 1: Dataset Preparation**
- Creates TOML configuration file
- Generates empty caption .txt files (unconditional training)
- Sets up cache directories

**Phase 2: Latent Caching** (~5-10 min)
- Encodes images with VAE
- Uses tiling to fit in VRAM
- Saves to disk for reuse

**Phase 3: Text Encoder Caching** (~2-5 min)
- Processes text prompts (empty for unconditional)
- Uses FP8 precision for efficiency
- Saves embeddings

**Phase 4: LoRA Training** (~5-7 hours)
- Trains with bf16 mixed precision
- adamw8bit optimizer
- Gradient checkpointing enabled
- Saves checkpoints every 5 epochs

**Phase 5: ComfyUI Conversion**
- Converts Musubi format → ComfyUI format
- Output: `fallout_nv_epoch20.safetensors`

## Arguments Explained

- `--dataset` - Your images directory
- `--concept` - LoRA name (used in output filename)
- `--epochs` - Training iterations (20 = good for 34 images)
- `--lora-rank` - Default 32 (recommended for video)
- `--lora-alpha` - Default 32 (matches rank)
- `--resolution` - Width Height (default: 960 544)
- `--learning-rate` - Default 1e-4 (stable for video)
- `--batch-size` - Default 1 (max for 24GB)
- `--skip-cache` - Reuse cached latents/embeddings
- `--skip-download` - Reuse downloaded models

## Output Structure

```
outputs/lora/fallout_nv/
├── dataset_config.toml       # Generated TOML config
├── cache/                    # Cached latents
├── lora-000020.safetensors   # Training checkpoint
└── fallout_nv_epoch20.safetensors  # ComfyUI-ready LoRA ✅
```

## Using in ComfyUI

### Step 1: Copy LoRA

```bash
cp outputs/lora/fallout_nv/fallout_nv_epoch20.safetensors \
   /path/to/ComfyUI/models/loras/
```

### Step 2: Load in Workflow

Your workflow (`workflows/custom_lora_test.json`) already has the right structure!

Update the LoRA filename:
```json
{
  "type": "LoraLoaderModelOnly",
  "widgets_values": [
    "fallout_nv_epoch20.safetensors",
    0.8
  ]
}
```

### Strength Recommendations

- **1.0** - Maximum effect
- **0.8** - Strong (recommended start)
- **0.6** - Moderate
- **0.4** - Subtle

## Troubleshooting

### Error: "Musubi Tuner not found"
```bash
docker compose build train
```

### Error: Model download fails
Check internet connection and try again. Models are from Comfy-Org's official repackaged repo.

### Error: CUDA out of memory
Already optimized for 24GB:
- Uses gradient checkpointing
- VAE tiling enabled
- FP8 text encoder
- Batch size 1

If still OOM, try reducing resolution:
```bash
--resolution 768 432
```

### Training stops early
Check:
1. Disk space (need ~50GB for models + cache + outputs)
2. GPU temperature (<85°C)
3. Check logs in outputs/lora/fallout_nv/logs/

## Workflow Integration

Your `LoraLoaderModelOnly` workflow node expects:
1. LoRA file in `ComfyUI/models/loras/`
2. Safetensors format ✅
3. Compatible with Wan 2.2 models ✅

The Musubi → ComfyUI conversion ensures full compatibility!

## Re-running with Different Settings

Cache latents/embeddings once, then experiment:

```bash
# First run - full workflow
docker compose --profile pipeline run --rm train \
    python pipelines/train/train_hunyuan_musubi.py \
    --dataset data/scraped/fallout_nv_20260116_113625 \
    --concept "fallout_nv_v1" \
    --epochs 20

# Second run - skip caching, try more epochs
docker compose --profile pipeline run --rm train \
    python pipelines/train/train_hunyuan_musubi.py \
    --dataset data/scraped/fallout_nv_20260116_113625 \
    --concept "fallout_nv_v2" \
    --epochs 30 \
    --skip-cache \
    --skip-download
```

## Expected Results

After training on 34 images for 20 epochs:
- **Training time**: 6-8 hours (RTX 5090)
- **LoRA size**: ~100-150MB (rank 32)
- **Quality**: Should capture Fallout NV style
- **ComfyUI**: Works with LoraLoaderModelOnly at strength 0.6-1.0

## References

- [Musubi Tuner GitHub](https://github.com/kohya-ss/musubi-tuner)
- [HunyuanVideo 1.5 Docs](https://github.com/kohya-ss/musubi-tuner/blob/main/docs/hunyuan_video_1_5.md)
- [Dataset Config Guide](https://github.com/kohya-ss/musubi-tuner/blob/main/docs/dataset_config.md)
- Your workflow: `workflows/custom_lora_test.json`
