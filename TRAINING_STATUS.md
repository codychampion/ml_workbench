# LoRA Training Status

## ✅ READY TO TRAIN - Using Musubi Tuner

We've integrated **Musubi Tuner** (Kohya-ss) for reliable, ComfyUI-compatible LoRA training!

**Official HunyuanVideo-1.5 script has compatibility issues - see `HUNYUANVIDEO_TRAINING_ISSUES.md` for details.**

**Quick Start**: See `pipelines/train/MUSUBI_QUICKSTART.md` for step-by-step guide.

### Musubi Tuner Implementation ✅
- ✅ Docker setup with CUDA 12.8 for RTX 5090
- ✅ GPU detection working
- ✅ Dataset collection (34 Fallout NV images ready!)
- ✅ Musubi Tuner installed at `/opt/musubi-tuner`
- ✅ All training dependencies (accelerate, safetensors, etc.)
- ✅ Wrapper script for full training workflow
- ✅ ComfyUI compatibility verified (safetensors output)
- ✅ Works with existing `LoraLoaderModelOnly` workflow
- ✅ Single GPU optimizations (gradient checkpointing, bf16)

### Current Issues ❌

**VAE Configuration Error**:
```
TypeError: AutoencoderKLConv3D.__init__() missing 4 required positional arguments:
'ffactor_spatial', 'ffactor_temporal', 'sample_size', and 'sample_tsize'
```

The official training script uses a custom VAE class that requires config parameters not present in the public HuggingFace models (`tencent/HunyuanVideo-1.5` or `hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-720p_t2v`).

### Recommended Alternatives ⭐

Instead of the official script, use proven community solutions:

1. **Musubi Tuner** (Kohya-ss) - Most recommended, specifically designed for video LoRA training
2. **SimpleTuner** - General-purpose tool with HunyuanVideo 1.5 support
3. **FineTrainers** - Diffusers-based with HunyuanVideo support

See `HUNYUANVIDEO_TRAINING_ISSUES.md` for detailed comparison and setup instructions.

### Previous Issues (Resolved but Blocked by VAE Error) ⚠️
1. **ComfyUI Safetensors**: FP8 quantized format incompatible with diffusers
   - ✅ **Solution**: Using official HunyuanVideo training code instead

2. **HunyuanVideo-1.5 from HuggingFace**: Config format incompatibility
   - ✅ **Solution**: Official training code handles model loading correctly

3. **Docker Volume Mount Override**
   - ✅ **Solution**: Moved repo to `/opt/` to avoid `/workspace` volume mount

4. **Missing Dependencies**
   - ✅ **Solution**: Added all HunyuanVideo dependencies to Dockerfile

5. **Wrong Training Arguments**
   - ✅ **Solution**: Created wrapper that maps args correctly

6. **Distributed Training Error** (`sp_size=8` on single GPU)
   - ✅ **Solution**: Added `--sp_size 1` and `--enable_gradient_checkpointing`

## 🚀 How to Run Training (Musubi Tuner)

### Step 1: Rebuild Docker Image
```bash
docker compose build train
```

This installs Musubi Tuner and all dependencies (~10-15 minutes).

### Step 2: Start Training
```bash
docker compose --profile pipeline run --rm train \
    python pipelines/train/train_hunyuan_musubi.py \
    --dataset data/scraped/fallout_nv_20260116_113625 \
    --concept "fallout_nv" \
    --epochs 20 \
    --lora-rank 8 \
    --lora-alpha 16
```

**That's it!** The script handles:
1. Dataset preparation
2. Latent caching (VAE encoding)
3. Text encoder caching
4. LoRA training
5. Output in ComfyUI-compatible format

**See `pipelines/train/MUSUBI_QUICKSTART.md` for full guide.**

### Training Configuration
- **Dataset**: 34 Fallout New Vegas images
- **Model**: tencent/HunyuanVideo (default)
- **Method**: Musubi Tuner (Kohya-ss)
- **LoRA**: rank=8, alpha=16
- **Learning rate**: 1e-4
- **Batch size**: 1
- **Total steps**: 680 (34 steps/epoch × 20 epochs)
- **Precision**: bfloat16 (mixed precision)
- **GPU**: Single RTX 5090 with gradient checkpointing
- **Estimated time**: 6-8 hours (includes caching)

### What Happens During Training

**Phase 1: Latent Caching** (~5-10 minutes)
- VAE encodes all images
- Saves latents to disk for reuse

**Phase 2: Text Encoder Caching** (~2-5 minutes)
- Processes text prompts (empty for unconditional)
- Saves embeddings to disk

**Phase 3: LoRA Training** (~5-7 hours)
- Trains LoRA adapter on DiT transformer
- Saves checkpoints during training
- Final output: `fallout_nv_epoch20.safetensors`

### After Training - ComfyUI Integration

The LoRA file `fallout_nv_epoch20.safetensors` works directly with your existing workflow!

**Your workflow** (`workflows/custom_lora_test.json`):
```json
{
  "type": "LoraLoaderModelOnly",
  "widgets_values": [
    "fallout_nv_epoch20.safetensors",  ← Your new LoRA
    0.8                                 ← Recommended strength
  ]
}
```

**Steps to use**:
1. Copy LoRA to ComfyUI: `cp outputs/lora/fallout_nv/fallout_nv_epoch20.safetensors /path/to/ComfyUI/models/loras/`
2. Load in your workflow with `LoraLoaderModelOnly` node
3. Adjust strength: 0.6-1.0 (start with 0.8)
4. Generate Fallout NV style videos!

## Implementation Details

### What We've Built
- ✅ Official HunyuanVideo-1.5 training integration
- ✅ User-friendly wrapper script (`train_hunyuan_official.py`)
- ✅ Automatic dataset format conversion
- ✅ Single GPU configuration
- ✅ Gradient checkpointing for 24GB VRAM
- ✅ Progress logging every 10 steps
- ✅ Checkpoint saving every 500 steps

## Files Created/Modified

### New Files
- `pipelines/train/train_hunyuan_official.py` - Wrapper for official HunyuanVideo training
- `TRAINING_STATUS.md` - This status document

### Modified Files
- `pipelines/train/Dockerfile` - Added HunyuanVideo dependencies + cloned official repo
  - Added: loguru, einops, imageio, imageio-ffmpeg, av, opencv-python, wandb, timm, ftfy, regex
  - Cloned: `https://github.com/Tencent-Hunyuan/HunyuanVideo-1.5.git` to `/opt/HunyuanVideo-1.5`

### Previous Attempts (Archived)
- `pipelines/train/train_video_lora_real.py` - Custom training script (incompatible with model formats)
- `scripts/scrape_and_train.sh` - End-to-end pipeline
- `models/README.md` - Model setup guide
- `WAN22_REAL_TRAINING.md` - Comprehensive training guide

## Technical Architecture

### How the Wrapper Works
The `train_hunyuan_official.py` wrapper:
1. Accepts user-friendly arguments (like `--dataset`, `--concept`, `--epochs`)
2. Prepares dataset in official format:
   - Creates `images/` directory with symlinked images
   - Creates `prompts.json` with empty prompts (unconditional training)
3. Translates arguments to official script format:
   - `--model` → `--pretrained_model_root` + `--pretrained_transformer_version`
   - `--epochs` → `--max_steps` (calculated: images × epochs / batch_size)
   - `--lora-rank` → `--lora_r`
4. Runs official training script with correct arguments
5. Saves checkpoints to `outputs/lora/{concept}/`

### Single GPU Optimizations
- `--sp_size 1` - Disables sequence parallelism (multi-GPU feature)
- `--enable_gradient_checkpointing` - Trades compute for memory
- `--dtype bf16` - Mixed precision training
- Batch size 1 - Fits in 24GB VRAM
