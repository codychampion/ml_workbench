# HunyuanVideo LoRA Training - Issues and Solutions

## Issue Summary

We attempted to use the official Tencent HunyuanVideo-1.5 training script for LoRA training but encountered compatibility issues with the public HuggingFace model releases.

## Technical Issues Encountered

### 1. Model Format Incompatibility

**Issue**: The official training script expects a specific checkpoint structure that doesn't match the HuggingFace model format.

**What We Tried**:
- `hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-720p_t2v` - Diffusers format
- `tencent/HunyuanVideo-1.5` - Official release

**Error**:
```
TypeError: AutoencoderKLConv3D.__init__() missing 4 required positional arguments:
'ffactor_spatial', 'ffactor_temporal', 'sample_size', and 'sample_tsize'
```

**Root Cause**: The training script uses a custom `AutoencoderKLConv3D` VAE class that requires configuration parameters not present in the standard HuggingFace model configs.

### 2. Directory Structure Expectations

**Issue**: The training script looks for subdirectories (like `720p_t2v/`) within the model root, but HuggingFace repos don't have this structure.

**Error**:
```
Could not find 720p_t2v in /workspace/.cache/huggingface/hub/models--hunyuanvideo-community--HunyuanVideo-1.5-Diffusers-720p_t2v/snapshots/...
```

**Workaround**: Removing the `--pretrained_transformer_version` argument, but this triggers the VAE config error.

## Working Alternatives

Based on community research, these are proven solutions for HunyuanVideo LoRA training:

### 1. Musubi Tuner (Kohya-ss) ⭐ Recommended

**Pros**:
- Specifically designed for video LoRA training
- Supports HunyuanVideo and HunyuanVideo-1.5
- Active community support
- Well-documented

**Cons**:
- More complex setup (requires pre-caching steps)
- Different workflow than official script

**Setup**:
```bash
git clone https://github.com/kohya-ss/musubi-tuner
cd musubi-tuner
pip install -e .
accelerate config
```

**Documentation**: https://github.com/kohya-ss/musubi-tuner

**Tutorial**: https://civitai.com/articles/10588/how-to-train-hunyuanvideo-lora-using-musubi-tuner

### 2. SimpleTuner

**Pros**:
- General-purpose fine-tuning kit
- Supports HunyuanVideo 1.5 as of recent updates
- Works with Diffusers models

**Cons**:
- Less documentation specifically for HunyuanVideo
- May require more configuration

**Setup**:
```bash
git clone https://github.com/bghira/SimpleTuner
cd SimpleTuner
pip install -r requirements.txt
```

**Documentation**: https://github.com/bghira/SimpleTuner

### 3. FineTrainers

**Pros**:
- Built on Diffusers library
- Added HunyuanVideo support in Dec 2024

**Cons**:
- Work-in-progress
- May have limited features

**Setup**:
```bash
git clone https://github.com/a-r-r-o-w/finetrainers
cd finetrainers
pip install -e .
```

### 4. Diffusion-Pipe

**Pros**:
- Optimized specifically for HunyuanVideo
- Pipeline-parallel architecture for efficiency

**Cons**:
- Less widespread adoption
- Limited documentation

## Why The Official Script Doesn't Work (Yet)

The official HunyuanVideo-1.5 training script was released on December 5, 2025, and appears to be designed for:

1. **Internal Tencent checkpoint format** - Not the public HuggingFace releases
2. **Specific directory structure** - Expects checkpoint subdirectories that don't exist in HF repos
3. **Custom VAE configs** - Requires parameters not in standard diffusers configs

### Potential Solutions for Official Script

1. **Wait for updates**: Tencent may release training-compatible model checkpoints
2. **Download full checkpoints**: Use `huggingface-cli download` to get all files locally
3. **Modify training script**: Patch the VAE loading code to handle missing configs
4. **Contact Tencent**: Report the compatibility issue on GitHub

## Recommended Approach

### For Quick Results: Use Musubi Tuner

Musubi Tuner is the most battle-tested solution with active community support and tutorials.

**Steps**:
1. Install Musubi Tuner in our Docker environment
2. Cache latents and text embeddings for your dataset
3. Run LoRA training with proven configs
4. Export ComfyUI-compatible LoRA weights

### For Long-term: Monitor Official Script

Keep an eye on the official HunyuanVideo-1.5 repository for:
- Training-specific model releases
- Config file updates
- Community solutions and patches

## Implementation Plan

### Option A: Add Musubi Tuner to our pipeline

**Changes needed**:
1. Update `pipelines/train/Dockerfile`:
   - Clone musubi-tuner repo
   - Install dependencies
   - Configure accelerate

2. Create wrapper script `pipelines/train/train_hunyuan_musubi.py`:
   - Handle dataset preparation
   - Run latent caching
   - Execute training
   - Convert output to ComfyUI format

3. Update documentation with Musubi Tuner workflow

**Estimated time**: 2-3 hours to set up and test

### Option B: Wait for Official Script Fix

**Changes needed**:
1. Monitor HunyuanVideo-1.5 repo for updates
2. Test with any new model releases or patches
3. Update our wrapper when compatibility improves

**Estimated time**: Unknown - depends on Tencent's release schedule

## References

- [Official HunyuanVideo-1.5 Training](https://github.com/Tencent-Hunyuan/HunyuanVideo-1.5)
- [Musubi Tuner](https://github.com/kohya-ss/musubi-tuner)
- [SimpleTuner](https://github.com/bghira/SimpleTuner)
- [HunyuanVideo LoRA Training Blog](https://huggingface.co/blog/neph1/hunyuan-lora)
- [Civitai Training Tutorial](https://civitai.com/articles/10588/how-to-train-hunyuanvideo-lora-using-musubi-tuner)
- [Civitai Windows GUI Tutorial](https://civitai.com/articles/10335/hunyuan-video-lora-trainning-with-gui-in-windows)

## What We Built

Despite the compatibility issues, we created:

- ✅ Docker environment with CUDA 12.8 for RTX 5090
- ✅ Dataset scraping and preparation pipeline
- ✅ Integration with official HunyuanVideo-1.5 repo
- ✅ Wrapper script for translating arguments
- ✅ Single GPU optimizations (gradient checkpointing, sp_size=1)

This infrastructure can be reused with any of the alternative training frameworks.

## Next Steps

**Immediate**:
1. Decide whether to implement Musubi Tuner or wait for official script fixes
2. If Musubi: Integrate into Docker pipeline
3. If wait: Monitor GitHub for updates

**Testing**:
Once training works, test the generated LoRA with:
1. ComfyUI LoRA loader
2. Different strength values (0.4, 0.6, 0.8, 1.0)
3. Combination with other LoRAs
4. Video generation quality
