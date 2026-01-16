# HunyuanVideo LoRA Training - Quick Start

## Prerequisites
- Docker with GPU support
- RTX 5090 (or other CUDA-capable GPU with 24GB+ VRAM)
- Dataset of images scraped with the collect pipeline

## Step 1: Rebuild Docker Image
```bash
docker compose build train
```

This will:
- Install PyTorch with CUDA 12.8
- Install all HunyuanVideo dependencies
- Clone official HunyuanVideo-1.5 repo to `/opt/HunyuanVideo-1.5`

## Step 2: Run Training

### Basic Usage
```bash
docker compose --profile pipeline run --rm train \
    python pipelines/train/train_hunyuan_official.py \
    --dataset data/scraped/your_dataset \
    --concept "your_concept_name" \
    --model "tencent/HunyuanVideo-1.5" \
    --epochs 20
```

### Full Example (Fallout New Vegas)
```bash
docker compose --profile pipeline run --rm train \
    python pipelines/train/train_hunyuan_official.py \
    --dataset data/scraped/fallout_nv_20260116_113625 \
    --concept "fallout_nv" \
    --model "tencent/HunyuanVideo-1.5" \
    --epochs 20 \
    --lora-rank 8 \
    --lora-alpha 16
```

### All Available Arguments
```bash
python pipelines/train/train_hunyuan_official.py --help
```

Options:
- `--dataset, -d` - Path to dataset directory (required)
- `--concept, -c` - Concept name for output directory (required)
- `--model` - Model path or HuggingFace repo (required)
- `--output, -o` - Output directory (default: `./outputs/lora/{concept}`)
- `--epochs` - Number of training epochs (default: 5)
- `--batch-size` - Training batch size (default: 1)
- `--learning-rate` - Learning rate (default: 1e-4)
- `--lora-rank` - LoRA rank (default: 8)
- `--lora-alpha` - LoRA alpha (default: 16)
- `--resolution` - Training resolution (default: 512)

## What Happens During Training

### Phase 1: Setup (~2 minutes)
1. Checks for official HunyuanVideo-1.5 repo in `/opt/HunyuanVideo-1.5`
2. Prepares dataset in official format:
   - Creates symlinks in `images/` directory
   - Generates `prompts.json` with empty prompts (unconditional training)
3. Calculates training steps: `(num_images / batch_size) × epochs`

### Phase 2: Model Download (one-time, ~10 minutes)
1. Downloads HunyuanVideo-1.5 model from HuggingFace (~33GB)
2. Cached to `/workspace/.cache/huggingface/` (mounted as Docker volume)
3. Subsequent runs skip this step

### Phase 3: Training (~5-7 hours for 20 epochs)
1. Initializes LoRA layers on transformer blocks
2. Trains with gradient checkpointing
3. Logs progress every 10 steps
4. Saves checkpoints every 500 steps
5. Uses bfloat16 precision for memory efficiency

### Phase 4: Completion
1. Saves final LoRA weights to `outputs/lora/{concept}/`
2. Format: safetensors (ComfyUI compatible)

## Output Structure
```
outputs/lora/your_concept/
├── prepared_dataset/       # Dataset in official format
│   ├── images/            # Symlinked training images
│   └── prompts.json       # Empty prompts
├── checkpoint-500/        # Intermediate checkpoint
│   └── adapter_model.safetensors
├── checkpoint-1000/       # Another checkpoint
│   └── adapter_model.safetensors
└── adapter_model.safetensors  # Final trained LoRA
```

## Using the Trained LoRA

### In ComfyUI
1. Copy LoRA file to ComfyUI models directory:
   ```bash
   cp outputs/lora/your_concept/adapter_model.safetensors \
      models/loras/your_concept.safetensors
   ```

2. In ComfyUI workflow:
   - Add "Load LoRA" node
   - Select `your_concept.safetensors`
   - Set strength (start with 0.8-1.0)
   - Connect to your HunyuanVideo model

### LoRA Strength Guidelines
- **1.0** - Full effect (may be too strong)
- **0.8** - Strong but balanced (recommended starting point)
- **0.6** - Moderate effect
- **0.4** - Subtle effect

## Monitoring Training

### Check GPU Usage
```bash
# In another terminal
watch -n 1 nvidia-smi
```

You should see:
- GPU utilization: 80-100%
- Memory usage: 20-22GB / 24GB
- Temperature: Below 85°C

### Check Training Logs
Training outputs to stdout. Look for:
- `[Training] Estimated steps: 680 (34 steps/epoch × 20 epochs)`
- Loss values decreasing over time
- Checkpoint saves every 500 steps

### Signs of Problems
- **GPU not utilized**: Check CUDA setup with `nvidia-smi`
- **OOM (out of memory)**: Reduce batch size or enable gradient checkpointing
- **Loss not decreasing**: May need to adjust learning rate
- **Loss goes to NaN**: Learning rate too high, reduce it

## Troubleshooting

### Error: "Official HunyuanVideo-1.5 repo not found"
**Solution**: Rebuild Docker image
```bash
docker compose build train
```

### Error: "ModuleNotFoundError: No module named 'loguru'"
**Solution**: Rebuild Docker image (dependencies missing)
```bash
docker compose build train
```

### Error: "sp_size (8) cannot be greater than world_size (1)"
**Solution**: Already fixed in latest version. Update code and rebuild:
```bash
git pull
docker compose build train
```

### Error: "CUDA out of memory"
**Solution**: Reduce batch size
```bash
--batch-size 1  # Already the default minimum
```

If still OOM, your GPU may have insufficient VRAM for this model.

### Training Too Slow
**Options**:
1. Reduce epochs: `--epochs 10`
2. Reduce dataset size (fewer images)
3. Use faster GPU
4. Disable gradient checkpointing (requires more VRAM)

## Expected Results

After training on 34 images for 20 epochs:
- Training time: 5-7 hours on RTX 5090
- Final LoRA size: ~50-100MB
- Should capture style/content of your training images
- Can be combined with other LoRAs in ComfyUI

## Next Steps

1. Test the LoRA in ComfyUI
2. Experiment with different LoRA strengths
3. Try training with more images for better results
4. Combine with other LoRAs for unique styles
