# LoRA Training Status

## ⚠️ BLOCKED - Official Training Script Has Compatibility Issues

We've encountered fundamental compatibility issues with the official HunyuanVideo-1.5 training script.

**See `HUNYUANVIDEO_TRAINING_ISSUES.md` for full technical analysis and alternative solutions.**

### Implementation Complete ✅
- ✅ Docker setup with CUDA 12.8 for RTX 5090
- ✅ GPU detection working
- ✅ Dataset collection (34 Fallout NV images)
- ✅ Official HunyuanVideo-1.5 repo cloned to `/opt/HunyuanVideo-1.5`
- ✅ All training dependencies installed (loguru, einops, imageio, etc.)
- ✅ Wrapper script that translates our args to official format
- ✅ Single GPU configuration (`--sp_size 1`)
- ✅ Gradient checkpointing for memory efficiency
- ✅ Dataset preparation pipeline

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

## 🚀 How to Run Training

### Step 1: Rebuild Docker Image
```bash
docker compose build train
```

### Step 2: Start Training
```bash
docker compose --profile pipeline run --rm train \
    python pipelines/train/train_hunyuan_official.py \
    --dataset data/scraped/fallout_nv_20260116_113625 \
    --concept "fallout_nv" \
    --model "hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-720p_t2v" \
    --epochs 20 \
    --lora-rank 8 \
    --lora-alpha 16
```

**Important**: Use the diffusers-formatted model from `hunyuanvideo-community`, not `tencent/HunyuanVideo-1.5`!

### Training Configuration
- **Dataset**: 34 Fallout New Vegas images
- **Model**: hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-720p_t2v
- **LoRA**: rank=8, alpha=16
- **Learning rate**: 1e-4
- **Batch size**: 1
- **Total steps**: 680 (34 steps/epoch × 20 epochs)
- **Precision**: bfloat16
- **GPU**: Single RTX 5090 with gradient checkpointing
- **Estimated time**: 5-7 hours

### What Happens During Training
1. Downloads HunyuanVideo-1.5 model (~33GB, one-time)
2. Prepares dataset in official format (images/ + prompts.json)
3. Trains LoRA adapter on transformer blocks
4. Saves checkpoints every 500 steps to `outputs/lora/fallout_nv/`
5. Final LoRA weights in safetensors format

### After Training
The LoRA adapter will be in `outputs/lora/fallout_nv/` and can be:
- Loaded in ComfyUI with the LoRA loader node
- Applied to your existing Wan 2.2 / HunyuanVideo models
- Used for Fallout New Vegas style video generation

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
