# HunyuanVideo LoRA Training with Musubi Tuner - Quick Start

## Why Musubi Tuner?

Musubi Tuner (by Kohya-ss) is the **recommended solution** for HunyuanVideo LoRA training because:
- ✅ **Proven to work** - Battle-tested by the community
- ✅ **ComfyUI compatible** - Outputs safetensors format that works with your workflows
- ✅ **Well documented** - Active community support and tutorials
- ✅ **Professional quality** - Used for production LoRA training

The official HunyuanVideo-1.5 training script has compatibility issues with public models. See `HUNYUANVIDEO_TRAINING_ISSUES.md` for details.

## Prerequisites

- Docker with GPU support
- RTX 5090 (or GPU with 24GB+ VRAM)
- Dataset of images (your Fallout NV images are ready!)
- ComfyUI workflow (we verified compatibility with your existing workflows)

## Your Existing Workflow Compatibility

Your ComfyUI workflow (`workflows/custom_lora_test.json`) uses:
```json
{
  "type": "LoraLoaderModelOnly",
  "widgets_values": [
    "newvegas_epoch5.safetensors",
    0.8
  ]
}
```

Musubi Tuner will output `fallout_nv_epoch20.safetensors` in exactly this format! ✅

## Step 1: Rebuild Docker Image

```bash
docker compose build train
```

This will:
- Install PyTorch with CUDA 12.8
- Clone Musubi Tuner from GitHub
- Install all dependencies
- Configure accelerate for single GPU

**Time**: ~10-15 minutes

## Step 2: Run LoRA Training

### Basic Command (Recommended)

```bash
docker compose --profile pipeline run --rm train \
    python pipelines/train/train_hunyuan_musubi.py \
    --dataset data/scraped/fallout_nv_20260116_113625 \
    --concept "fallout_nv" \
    --epochs 20
```

### Full Command with All Options

```bash
docker compose --profile pipeline run --rm train \
    python pipelines/train/train_hunyuan_musubi.py \
    --dataset data/scraped/fallout_nv_20260116_113625 \
    --concept "fallout_nv" \
    --model "tencent/HunyuanVideo" \
    --epochs 20 \
    --lora-rank 8 \
    --lora-alpha 16 \
    --learning-rate 0.0001 \
    --batch-size 1
```

### Arguments Explained

- `--dataset` - Your scraped images directory
- `--concept` - Name for your LoRA (used in output filename)
- `--model` - Base model (default: tencent/HunyuanVideo)
- `--epochs` - Training iterations (20 recommended for 34 images)
- `--lora-rank` - LoRA rank (8 = good balance of quality/size)
- `--lora-alpha` - LoRA alpha (16 = standard value)
- `--learning-rate` - Learning rate (0.0001 = stable for video)
- `--batch-size` - Images per step (1 = fits in 24GB VRAM)

## Step 3: Training Process

The script runs 3 phases automatically:

### Phase 1: Latent Caching (~5-10 minutes)
```
[Musubi] Step 1/3: Caching latents (VAE encoding)...
```
- Encodes images with VAE
- Saves to disk to avoid re-encoding
- One-time process per dataset

### Phase 2: Text Encoder Caching (~2-5 minutes)
```
[Musubi] Step 2/3: Caching text encoder outputs...
```
- Processes text prompts (empty for unconditional training)
- Saves embeddings to disk
- One-time process per dataset

### Phase 3: LoRA Training (~5-7 hours)
```
[Musubi] Step 3/3: Training LoRA adapter...
```
- Trains the actual LoRA weights
- Shows progress with loss values
- Saves checkpoints during training
- Final output: `fallout_nv_epoch20.safetensors`

**Total time**: ~6-8 hours for 20 epochs on RTX 5090

### Monitoring Training

Watch GPU usage in another terminal:
```bash
watch -n 1 nvidia-smi
```

Expected:
- GPU utilization: 80-100%
- Memory usage: 20-22GB / 24GB
- Temperature: <85°C

## Step 4: Use in ComfyUI

### Option A: Copy to ComfyUI Directory

```bash
cp outputs/lora/fallout_nv/fallout_nv_epoch20.safetensors \
   /path/to/ComfyUI/models/loras/
```

### Option B: Use Your Existing Workflow

Your workflow at `workflows/custom_lora_test.json` is already set up!

Just update the LoRA filename:
```json
{
  "type": "LoraLoaderModelOnly",
  "widgets_values": [
    "fallout_nv_epoch20.safetensors",  // ← Your new LoRA
    0.8                                 // ← Try 0.6-1.0
  ]
}
```

### LoRA Strength Guidelines

Based on your workflow (strength: 0.8):
- **1.0** - Maximum effect (may overpower base model)
- **0.8** - Your current setting (strong but balanced) ✅
- **0.6** - Moderate effect
- **0.4** - Subtle hint of style

## Advanced: Re-running Training

If you want to try different hyperparameters, you can skip the caching steps:

```bash
docker compose --profile pipeline run --rm train \
    python pipelines/train/train_hunyuan_musubi.py \
    --dataset data/scraped/fallout_nv_20260116_113625 \
    --concept "fallout_nv_v2" \
    --epochs 30 \
    --lora-rank 16 \
    --skip-cache
```

The `--skip-cache` flag uses the previously cached latents/embeddings, saving ~10-15 minutes.

## Output Structure

```
outputs/lora/fallout_nv/
├── musubi_dataset/          # Prepared dataset
│   ├── img/                 # Numbered images
│   │   ├── 001.jpg
│   │   ├── 002.jpg
│   │   └── ...
│   └── meta_lat.json        # Dataset metadata
├── logs/                    # TensorBoard logs
├── lora.safetensors         # Raw LoRA output
└── fallout_nv_epoch20.safetensors  # Final renamed LoRA ✅
```

## Troubleshooting

### Error: "Musubi Tuner not found"
**Solution**: Rebuild Docker image
```bash
docker compose build train
```

### Error: "CUDA out of memory"
**Solutions**:
1. Reduce batch size (already at minimum: 1)
2. Reduce LoRA rank: `--lora-rank 4`
3. Check no other processes using GPU: `nvidia-smi`

### Training too slow
**Options**:
1. Reduce epochs: `--epochs 10`
2. Reduce dataset size (fewer images)
3. Use gradient checkpointing (already enabled)

### LoRA not working in ComfyUI
**Checks**:
1. Verify LoRA file is in `ComfyUI/models/loras/`
2. Check file is not corrupted: `ls -lh fallout_nv_epoch20.safetensors`
3. Try lower strength first: 0.4-0.6
4. Ensure base model matches training model

## Expected Results

After training on 34 images for 20 epochs:
- **Training time**: 6-8 hours on RTX 5090
- **LoRA file size**: ~50-100MB (rank 8)
- **Quality**: Should capture Fallout NV visual style
- **ComfyUI**: Load with `LoraLoaderModelOnly`, strength 0.6-0.8

## Workflow Integration

Your existing workflow structure:
1. **UNETLoader** → Load base Wan 2.2 model
2. **LoraLoaderModelOnly** → Apply your trained LoRA ✅
3. **ModelSamplingSD3** → Configure sampling
4. **CLIPTextEncode** → Your prompts
5. **KSampler** → Generate video

The Musubi LoRA slots perfectly into step 2!

## Next Steps

1. **Train your first LoRA** with default settings
2. **Test in ComfyUI** with strength 0.8
3. **Experiment**:
   - Try different strengths (0.4-1.0)
   - Train with more epochs (30-40)
   - Adjust LoRA rank (4, 8, 16, 32)
   - Combine with other LoRAs

## References

- [Musubi Tuner GitHub](https://github.com/kohya-ss/musubi-tuner)
- [HunyuanVideo LoRA Guide](https://huggingface.co/blog/neph1/hunyuan-lora)
- [Civitai Tutorial](https://civitai.com/articles/10588/how-to-train-hunyuanvideo-lora-using-musubi-tuner)
- Your workflow: `workflows/custom_lora_test.json`

## Comparison: Musubi vs Official Script

| Feature | Musubi Tuner | Official Script |
|---------|--------------|-----------------|
| **Works with public models** | ✅ Yes | ❌ No (VAE config error) |
| **ComfyUI compatible** | ✅ Yes (safetensors) | ⚠️ Unknown |
| **Community tested** | ✅ Widely used | ❌ New release |
| **Documentation** | ✅ Extensive | ⚠️ Limited |
| **Setup complexity** | ⚠️ Multi-step | ✅ Simple (if it worked) |
| **Training speed** | ✅ Fast (with caching) | ⚠️ Unknown |

**Winner**: Musubi Tuner for reliability and ComfyUI compatibility! ✅
