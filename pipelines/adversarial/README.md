# Adversarial Patch Generation Pipeline

Generate adversarial patches that fool ML models while maintaining aesthetic constraints.

## Overview

This pipeline implements style-constrained adversarial patch generation:
- **Adversarial**: Fool target models (person detectors, classifiers)
- **Aesthetic**: Look like specific styles (e.g., punk band patches)
- **Robust**: Work in physical world (EoT training)
- **Prominent**: Adversarial noise is VISIBLE, not hidden - embraced as part of the design

### Adversarial Noise as Aesthetic

**This implementation makes adversarial perturbations prominent and visible.**

Unlike traditional adversarial patches that try to hide perturbations, this pipeline:
- **Celebrates the glitch**: Adversarial noise becomes part of the punk/DIY aesthetic
- **Embraces corruption**: Digital artifacts are features, not bugs
- **Visible perturbations**: The adversarial patterns are clearly visible in the final patch
- **Glitchy aesthetic**: Think glitch art, digital punk, cyberpunk corruption

Control this with `--adversarial-prominence`:
- `0.5` = Subtle (smooth, hidden adversarial noise)
- `1.0` = Balanced (some visible artifacts)
- `2.0` = Prominent (default - clear adversarial patterns)
- `3.0+` = Maximum glitch (very visible digital corruption)

**Philosophy**: The adversarial noise itself becomes the punk aesthetic - raw, digital, unpolished.

## Quick Start

### 1. Generate Patch (CLI)

**Prominent adversarial noise (default):**
```bash
python -m pipelines.adversarial.generate_adv_patch \
  --target-model yolov8 \
  --attack evasion \
  --style "punk band patch, DIY aesthetic, black and white" \
  --test-images ./data/test/coco_persons \
  --iterations 500 \
  --adversarial-prominence 2.0 \
  --output ./outputs/adv_patches/punk_patch.png
```

**Maximum glitch aesthetic:**
```bash
python -m pipelines.adversarial.generate_adv_patch \
  --target-model yolov8 \
  --attack evasion \
  --style "glitch art punk patch, digital corruption, cyberpunk" \
  --test-images ./data/test/coco_persons \
  --adversarial-prominence 3.0 \
  --lambda-adv 0.7 \
  --output ./outputs/adv_patches/glitchy_punk.png
```

### 2. Generate Patch (Hydra Config)

```bash
python -m pipelines.adversarial.generate_adv_patch --hydra \
  pipeline=adversarial_patch
```

## Architecture

### Core Components

1. **Target Models** (`target_models.py`)
   - `YOLOv8Wrapper`: Person detection evasion
   - `CLIPClassifierWrapper`: Image classification attacks
   - Unified interface for different models

2. **Style Constraints** (`style_constraints.py`)
   - `CLIPStyleConstraint`: CLIP-based style matching
   - `TotalVariationLoss`: Smoothness regularization
   - `PrintabilityLoss`: Printable colors
   - `StyleConstraintCombined`: All constraints together

3. **Physical Transforms** (`physical_transforms.py`)
   - `PhysicalTransformPipeline`: Lighting, rotation, scale, noise
   - `PatchApplicator`: Apply patch to images
   - `EOTWrapper`: Expectation over Transformation training

4. **Patch Optimizer** (`patch_optimizer.py`)
   - `AdversarialPatchOptimizer`: Main optimization loop
   - Combines adversarial + style + regularization losses
   - Adam optimizer with gradient clipping

### Loss Function

```
total_loss = λ_adv * adversarial_loss
           + λ_style * (clip_loss + tv_loss + print_loss)

where:
- adversarial_loss: Fool target model
- clip_loss: Match style prompt (negative CLIP similarity)
- tv_loss: Smoothness (total variation)
- print_loss: Printable colors
```

## Configuration

Edit `conf/pipeline/adversarial_patch.yaml`:

```yaml
adversarial_patch:
  target_model:
    type: "yolov8"
    attack_objective: "evasion"

  style:
    prompt: "punk band patch, DIY aesthetic"

  optimization:
    lambda_adv: 0.5  # Adversarial weight
    lambda_style: 0.4  # Style weight
    use_eot: true
    eot_samples: 10

  output:
    path: "${paths.outputs}/adv_patches/patch.png"
```

## Dependencies

```bash
pip install torch torchvision
pip install ultralytics  # For YOLOv8
pip install git+https://github.com/openai/CLIP.git
```

## Example Workflows

### 1. Glitchy Punk Patch (Maximum Adversarial Prominence)

**Embracing adversarial noise as the aesthetic - glitch art punk style:**

```bash
python -m pipelines.adversarial.generate_adv_patch \
  --target-model yolov8 \
  --attack evasion \
  --style "glitch art punk patch, digital corruption, distorted, cyberpunk, data moshing" \
  --test-images ./data/test/coco_persons \
  --lambda-adv 0.7 \
  --lambda-style 0.3 \
  --adversarial-prominence 3.0 \
  --iterations 600 \
  --output ./outputs/adv_patches/glitchy_punk.png
```

**Result**: Highly visible adversarial patterns that look like intentional glitch art.

### 2. Prominent DIY Punk Patch (Balanced)

**Visible adversarial noise integrated with punk aesthetic:**

```bash
python -m pipelines.adversarial.generate_adv_patch \
  --target-model yolov8 \
  --attack evasion \
  --style "punk rock band patch, screen printed, DIY, high contrast, distressed" \
  --test-images ./data/test/coco_persons \
  --lambda-adv 0.5 \
  --lambda-style 0.4 \
  --adversarial-prominence 2.0 \
  --iterations 800 \
  --output ./outputs/adv_patches/prominent_punk.png
```

**Result**: Clear adversarial patterns that complement the DIY punk aesthetic.

### 3. Maximum Evasion (Minimal Style, Visible Noise)

**Prioritize attack effectiveness while keeping noise visible:**

```bash
python -m pipelines.adversarial.generate_adv_patch \
  --target-model yolov8 \
  --attack evasion \
  --style "abstract digital patch" \
  --test-images ./data/test/coco_persons \
  --lambda-adv 0.8 \
  --lambda-style 0.2 \
  --adversarial-prominence 2.5 \
  --iterations 500 \
  --output ./outputs/adv_patches/max_evasion_prominent.png
```

**Result**: Strong attack with visible digital corruption patterns.

### 4. Physical Robustness Test

```bash
python -m pipelines.adversarial.generate_adv_patch \
  --target-model yolov8 \
  --attack evasion \
  --style "punk band patch" \
  --test-images ./data/test/coco_persons \
  --eot-samples 20 \
  --iterations 800 \
  --output ./outputs/adv_patches/robust_patch.png
```

## Output

Each run produces:
- **Patch image**: `patch.png`
- **Metadata**: `patch.json` with:
  - Final losses
  - Style similarity score
  - Optimization parameters
  - Timestamp

## Metrics

- **Adversarial Loss**: Lower = better attack (model more fooled)
- **Style Similarity**: Higher = better match to style (0-1 scale)
- **Combined Success**: Balance both objectives

**Target**:
- Evasion rate > 80%
- Style similarity > 0.7

## Ethical Considerations

⚠️ **This is defensive security research.**

**Approved uses:**
- Testing model robustness
- Privacy protection research
- Academic studies
- Art projects on surveillance

**NOT approved:**
- Evading law enforcement
- Malicious surveillance evasion
- Harassment or stalking

See `knowledge/experiments/plans/adversarial_punk_patches_plan.md` for full ethical discussion.

## Physical Fabrication

### Making Wearable Patches

Convert digital patches to physical wearable items:

```bash
# Prepare patch for printing
python -m pipelines.adversarial.prepare_for_print \
  --patch ./outputs/adv_patches/punk_patch.png \
  --output ./outputs/print_ready/punk_patch_print.png \
  --size-inches 4 4 \
  --dpi 300 \
  --add-crop-marks \
  --color-profile sRGB
```

### Printing Methods

**Option 1: Fabric Transfer Paper (Easiest)**
1. Print on fabric transfer paper at 300 DPI
2. Follow manufacturer instructions for heat press/iron
3. Apply to dark fabric (black works best)
4. Recommended: Avery T-Shirt Transfers or similar

**Option 2: Screen Printing (Most Durable)**
1. Print transparency at 1200 DPI
2. Burn screen with emulsion
3. Print with fabric ink
4. Heat cure according to ink specs
5. Best for high-contrast designs

**Option 3: Sublimation (Polyester Only)**
1. Use sublimation printer and paper
2. Heat press at 400°F for 60 seconds
3. Only works on polyester/polymer-coated surfaces

**Option 4: Direct-to-Garment (Professional)**
1. Send print-ready PNG to DTG service
2. Specify fabric type and placement
3. Most expensive but highest quality

### Materials Needed

- **Fabric:** Black cotton, denim jacket, or canvas (for transfers/screen)
- **Patch backing:** Iron-on adhesive backing (optional for removable patches)
- **Sewing kit:** If making traditional sewn patches
- **Protective coating:** Clear acrylic sealer spray (optional for durability)

### Application Instructions

**Iron-On Application:**
1. Preheat fabric (cotton setting, no steam)
2. Position patch face-up on garment
3. Cover with parchment paper
4. Press firmly for 30-60 seconds
5. Let cool completely before peeling backing

**Sewn Application:**
1. Position patch on garment
2. Pin in place
3. Hand-stitch or machine-stitch around edges
4. Use black thread for punk aesthetic

### Physical Testing Protocol

After fabricating:

```bash
# Test physical patch effectiveness
python -m pipelines.evaluate.eval_physical_patch \
  --test-images ./data/physical_test/ \
  --patch-location shoulder \
  --lighting-conditions "indoor,outdoor,low-light" \
  --angles "-30,0,30" \
  --distances "1m,3m,5m"
```

### Durability & Care

- **Washing:** Turn inside out, cold water, gentle cycle
- **Drying:** Air dry or low heat (high heat may crack transfers)
- **Storage:** Lay flat or hang to avoid creasing
- **Re-testing:** Test adversarial effectiveness after 5, 10, 20 washes

### Expected Physical Performance

Based on EoT training:
- **Print-photograph cycle:** 70-80% evasion rate maintained
- **Viewing angle:** ±30° effective range
- **Lighting variation:** Robust to indoor/outdoor/low-light
- **Distance:** Effective 1-5 meters from camera

### Size Recommendations

- **Shoulder/chest patch:** 4" x 4" (100mm x 100mm)
- **Back patch:** 8" x 10" (200mm x 250mm)
- **Sleeve patch:** 3" x 3" (75mm x 75mm)

Default patch size (300x300 pixels) scales to 4"x4" at 300 DPI.

## Related

- **Experiment Plan**: `knowledge/experiments/plans/adversarial_punk_patches_plan.md`
- **Dataset Card**: `knowledge/datasets/cards/punk_patches_dataset_card.md`
- **Research**: Adversarial Patch (Brown et al. 2017), Robust Physical Perturbations
- **Fabrication Guide**: This section above

## Troubleshooting

**CUDA out of memory**:
- Reduce `--eot-samples` (default: 10 → 5)
- Reduce `--num-test-images` (default: 10 → 5)
- Use smaller patch size `--patch-size 200 200`

**Low style similarity**:
- Increase `--lambda-style` (0.4 → 0.6)
- More iterations `--iterations` (500 → 800)
- Better style prompt

**Low evasion rate**:
- Increase `--lambda-adv` (0.5 → 0.7)
- More iterations
- More EoT samples for robustness
