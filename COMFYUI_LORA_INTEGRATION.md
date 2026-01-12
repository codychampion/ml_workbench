# ComfyUI LoRA Integration Guide

## Your Current Setup

Your ComfyUI workflow already uses LoRAs via `LoraLoaderModelOnly` nodes (nodes #83 and #85):

```
Current LoRAs in your workflow:
- wan2.2_t2v_lightx2v_4steps_lora_v1.1_high_noise.safetensors (node 83)
- wan2.2_t2v_lightx2v_4steps_lora_v1.1_low_noise.safetensors (node 85)
```

These load on top of your base models:
- `wan2.2_t2v_high_noise_14B_fp8_scaled.safetensors` (node 75)
- `wan2.2_t2v_low_noise_14B_fp8_scaled.safetensors` (node 76)

## How to Add Your Custom Trained LoRAs

### Option 1: Automatic (Recommended)

Since `./outputs` is already shared between your training pipeline and ComfyUI:

```bash
# 1. Train your LoRA
python pipelines/train/train_video_lora.py \
    --dataset ./data/scraped/fallout \
    --concept fallout \
    --epochs 5

# 2. LoRA is saved to: ./outputs/lora/fallout/fallout_epoch5.safetensors

# 3. ComfyUI sees it automatically because of docker-compose.yml:
#    - ./outputs:/app/outputs:rw  (shared volume)
```

### Option 2: Manual Copy

If you want LoRAs in ComfyUI's standard location:

```bash
# Copy trained LoRA to ComfyUI loras folder
cp ./outputs/lora/fallout/fallout_epoch5.safetensors \
   ./models/loras/fallout_custom.safetensors
```

## Using Your Custom LoRA in ComfyUI

### Method 1: Replace Existing LoRA Node

In your workflow, change node #83 or #85:

**Before:**
```json
{
  "id": 83,
  "type": "LoraLoaderModelOnly",
  "widgets_values": [
    "wan2.2_t2v_lightx2v_4steps_lora_v1.1_high_noise.safetensors",
    1.0
  ]
}
```

**After:**
```json
{
  "id": 83,
  "type": "LoraLoaderModelOnly",
  "widgets_values": [
    "fallout_custom.safetensors",  // Your trained LoRA
    0.8  // Adjust strength (0.0-1.0)
  ]
}
```

### Method 2: Add New LoRA Node (Stack Multiple)

You can chain multiple LoRAs:

```
UNETLoader → LoraLoader (official) → LoraLoader (your custom) → ModelSampling
```

This lets you combine:
- Official 4-step speed LoRA
- Your custom Fallout-themed LoRA

## Training LoRAs for Your Workflow

### Quick Training Command

```bash
# Train Fallout-themed LoRA
python pipelines/train/train_video_lora.py \
    --dataset ./data/scraped/fallout \
    --concept fallout \
    --model /app/models/unet/wan2.2_t2v_high_noise_14B_fp8_scaled.safetensors \
    --epochs 5 \
    --lora-rank 8 \
    --batch-size 1
```

### Full Pipeline (Scrape + Train)

```bash
# Scrape Fallout images from Reddit and train LoRA
./scripts/scrape_and_train.sh fo4 fallout 100 5
```

## LoRA Compatibility

Your Wan 2.2 models support LoRAs because they're based on DiT (Diffusion Transformer) architecture.

**Compatible base models in your workflow:**
- ✅ `wan2.2_t2v_high_noise_14B_fp8_scaled.safetensors`
- ✅ `wan2.2_t2v_low_noise_14B_fp8_scaled.safetensors`

**LoRA training targets:**
- Attention layers (Q, K, V projections)
- Feed-forward layers
- Typically rank 4-16 (8 is default)

## Workflow Examples

### Example 1: Speed + Custom Theme

```
1. Load base model: wan2.2_t2v_high_noise_14B_fp8_scaled
2. Apply official 4-step LoRA (strength: 1.0)
3. Apply your Fallout LoRA (strength: 0.7)
4. Generate with prompt: "A power-armored soldier walking through nuclear wasteland"
```

### Example 2: Pure Custom LoRA

```
1. Load base model: wan2.2_t2v_high_noise_14B_fp8_scaled
2. Apply ONLY your Fallout LoRA (strength: 1.0)
3. Generate with prompt: "Vault dweller emerging from underground bunker"
```

## Testing Your LoRA

After training, test in ComfyUI:

1. **Open your workflow** in ComfyUI
2. **Click on LoraLoaderModelOnly node** (#83)
3. **Select your trained LoRA** from dropdown
4. **Adjust strength slider** (start with 0.5-0.8)
5. **Generate** with Fallout-themed prompts

Expected results:
- Stronger Fallout aesthetic (Pip-Boy UI, vault suits, power armor)
- Better adherence to Fallout art style
- More accurate environmental details (wasteland, retro-futurism)

## Troubleshooting

### LoRA Not Showing in ComfyUI

**Problem:** Your trained LoRA doesn't appear in the dropdown

**Solution:**
```bash
# Check if LoRA was created
ls -la ./outputs/lora/fallout/

# Check if ComfyUI can see it
ls -la ./models/loras/

# Restart ComfyUI to refresh model list
docker compose --profile comfyui restart
```

### LoRA Has No Effect

**Problem:** LoRA loads but doesn't change output

**Possible causes:**
1. Strength too low (try 0.8-1.0)
2. LoRA undertrained (train more epochs)
3. Base model mismatch (check you're using the right model)

**Solution:**
```bash
# Retrain with more epochs
python pipelines/train/train_video_lora.py \
    --dataset ./data/scraped/fallout \
    --concept fallout \
    --epochs 10  # More epochs
```

### Memory Issues

**Problem:** Training runs out of VRAM

**Solution:**
```bash
# Reduce batch size
python pipelines/train/train_video_lora.py \
    --batch-size 1 \
    --gradient-accumulation-steps 4  # Simulate batch_size=4
```

## Next Steps

1. **Complete the `train_video_lora.py` implementation** (currently symbolic)
   - Load actual Wan 2.2 model
   - Apply PEFT LoRA adapters
   - Implement proper training loop
   - Save as `.safetensors`

2. **Train your first LoRA**
   ```bash
   ./scripts/scrape_and_train.sh fnv newvegas 50 5
   ```

3. **Test in ComfyUI**
   - Load your trained LoRA
   - Generate Fallout-themed videos
   - Iterate on training if needed

---

**Bottom line:** Your ComfyUI workflow is LoRA-ready. We just need to train compatible LoRAs using your native pipeline.
