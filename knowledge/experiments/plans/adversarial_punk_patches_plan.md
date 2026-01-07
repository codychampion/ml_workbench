---
type: experiment-plan
exp_id: "adv-punk-001"
created: "2026-01-01"
status: draft
tags: [adversarial-ml, patches, punk, security, evasion, style-transfer]
target_models: [yolo, faster-rcnn, clip]
---

# Experiment: Adversarial Punk Band Patches

## Hypothesis

It's possible to generate adversarial patches that:
1. Cause misclassification/suppression in object detection and image classification models
2. Visually resemble authentic punk band patches (high aesthetic quality)
3. Maintain adversarial effectiveness when printed and photographed (physical robustness)

This explores the intersection of adversarial ML and style constraints - can we make adversarial perturbations that are both functional and aesthetically designed?

## Motivation

Traditional adversarial patches are often:
- Visually nonsensical (random noise patterns)
- Immediately suspicious to humans
- Not practical for real-world scenarios

**Goal:** Create adversarial patches that:
- Look like legitimate punk band patches (DIY aesthetic)
- Could plausibly be worn on clothing
- Still fool ML models (person detectors, age classifiers, etc.)

**Applications:**
- Privacy protection (evade surveillance)
- Security research (test model robustness)
- Art project (adversarial fashion)
- Academic research on constrained adversarial examples

## Dataset

### Punk Patch Reference Dataset
- **Source:** Mixed collection of punk band patches
- **Collection Strategy:**
  - Reddit: r/BattleJackets, r/punk, r/crustpunk
  - Band merchandise sites (scraped with permission)
  - DIY punk aesthetics references
- **Target Size:** 300-500 patch images
- **Path:** `./data/collected/punk_patches/`

### Target Model Test Dataset
- **COCO Detection:** Subset with person class
- **CelebA:** Age/gender classification
- **ImageNet:** General classification
- **Path:** `./data/collected/target_test_images/`

## Method

### Phase 1: Style Learning
Learn the visual distribution of punk patches using:

**Option A - StyleGAN/GAN approach:**
- Train StyleGAN2 on punk patch dataset
- Use latent space for style constraints

**Option B - Diffusion approach:**
- Fine-tune Stable Diffusion on punk patches
- Use diffusion prior as style constraint

**Option C - Direct optimization with CLIP:**
- Use CLIP embeddings to constrain patch to "punk band patch" style
- Simpler, no training required

### Phase 2: Adversarial Patch Generation

**Attack Objectives:**
1. **Person Detection Evasion:** Make person detector fail to detect humans
2. **Age Misclassification:** Cause age classifier to predict wrong age bracket
3. **Class Confusion:** Cause classifier to predict wrong object class

**Optimization Framework:**
```python
# Pseudo-code for adversarial patch generation
def generate_adversarial_punk_patch(
    target_model,
    style_constraint_model,  # CLIP or diffusion
    attack_objective="evasion"
):
    patch = initialize_patch()  # Start from noise or punk patch

    for iteration in range(max_iterations):
        # Adversarial loss: fool target model
        adv_loss = adversarial_objective(patch, target_model)

        # Style loss: look like punk patch
        style_loss = style_constraint(patch, "punk band patch DIY aesthetic")

        # Regularization: smoothness, printability
        reg_loss = total_variation(patch) + printability(patch)

        # Combined loss
        total_loss = λ_adv * adv_loss + λ_style * style_loss + λ_reg * reg_loss

        patch = update(patch, total_loss)

    return patch
```

**Key Parameters:**
- `λ_adv`: Weight for adversarial effectiveness (0.4-0.6)
- `λ_style`: Weight for punk aesthetic (0.3-0.5)
- `λ_reg`: Weight for printability (0.1-0.2)
- Patch size: 300x300 pixels (printable at ~4"x4")
- Color space: RGB (printable colors)

### Phase 3: Physical Robustness

Test patches under:
- Different lighting conditions
- Various viewing angles (±30°)
- Print-photograph cycle
- Different backgrounds
- Camera perspectives (smartphone cameras)

**Expectation on Transformation (EoT) training:**
```python
transformations = [
    random_brightness,
    random_contrast,
    random_rotation(±15°),
    random_scale(0.8-1.2),
    add_gaussian_noise,
    perspective_transform
]
```

### Phase 4: Punk Aesthetic Design

**Style Constraints:**
- Black/white high contrast (classic punk)
- Band logo aesthetic (DIY, photocopied look)
- Grunge textures and rough edges
- Punk typography (stencil fonts, handwritten)
- Common motifs: skulls, safety pins, anarchy symbols, band names

**CLIP Guidance Prompts:**
- "punk rock band patch, black and white, DIY aesthetic"
- "hardcore punk logo, screen printed patch, rough edges"
- "crusty punk battle jacket patch, photocopied zine aesthetic"

## Implementation Pipeline

### New Pipeline Script: `pipelines/adversarial/generate_adv_patch.py`

```python
#!/usr/bin/env python3
"""
Adversarial Patch Generator with Style Constraints
===================================================
Generate adversarial patches that fool ML models while maintaining
aesthetic constraints (punk band patch style).

Usage:
    python -m pipelines.adversarial.generate_adv_patch \
        --target-model yolov8 \
        --attack evasion \
        --style "punk band patch" \
        --output ./outputs/adv_patches/
"""
```

**Key Components:**
1. `TargetModelWrapper` - Unified interface for different victim models
2. `StyleConstraint` - CLIP-based or diffusion-based style guidance
3. `PatchOptimizer` - PGD/Adam optimization with EoT
4. `PhysicalTransform` - Simulate real-world conditions
5. `PatchEvaluator` - Test effectiveness and style quality

### Config: `conf/pipeline/generate_adv_patch.yaml`

```yaml
adversarial_patch:
  target_model:
    type: "yolov8"  # yolov8, faster-rcnn, clip-classifier
    checkpoint: "yolov8n.pt"
    attack_objective: "evasion"  # evasion, misclassification, confusion

  style:
    method: "clip"  # clip, diffusion, gan
    prompt: "punk rock band patch, DIY aesthetic, black and white"
    reference_images: "${paths.data.collected}/punk_patches"

  patch:
    size: [300, 300]
    init_method: "random"  # random, noise, style_sample
    colorspace: "rgb"
    printable_colors: true

  optimization:
    iterations: 500
    learning_rate: 0.01
    optimizer: "adam"
    lambda_adv: 0.5
    lambda_style: 0.4
    lambda_reg: 0.1

  robustness:
    eot_samples: 20
    transformations:
      - brightness: [0.8, 1.2]
      - contrast: [0.8, 1.2]
      - rotation: [-15, 15]
      - scale: [0.8, 1.2]
      - noise: 0.01

  evaluation:
    test_images: "${paths.data.test}/coco_persons"
    success_threshold: 0.8  # 80% evasion rate
    style_similarity_threshold: 0.7  # CLIP similarity to "punk patch"
```

## Success Criteria

### Adversarial Effectiveness
- **Evasion Rate:** > 80% on person detector (YOLO/Faster R-CNN)
- **Physical Robustness:** > 70% evasion rate after print-photo cycle
- **Angle Invariance:** Works at ±30° viewing angle

### Aesthetic Quality
- **Style Similarity:** CLIP similarity > 0.7 to "punk band patch"
- **Human Eval:** > 75% of people identify as "plausible punk patch"
- **Printability:** Colors within CMYK gamut, smooth gradients

### Combined Metric
**Adversarial Aesthetic Score (AAS):**
```
AAS = (evasion_rate * style_similarity) / (1 + human_suspicion_rate)
Target: AAS > 0.6
```

## Ethical Considerations

⚠️ **Important:** This research has dual-use implications.

**Defensive Applications:**
- Testing model robustness
- Privacy protection research
- Identifying vulnerabilities in surveillance systems

**Responsible Disclosure:**
- Document vulnerabilities found in target models
- Share findings with model developers
- Publish countermeasures alongside attacks

**Not for:**
- Evading law enforcement (illegal in many jurisdictions)
- Harassment or stalking
- Bypassing security systems maliciously

**Research Context:**
- Academic study of adversarial ML
- Art/fashion project on AI surveillance
- Security research with proper authorization

## Runs

| Run ID | Target | Style λ | Attack Success | Style Score | Notes |
|--------|--------|---------|----------------|-------------|-------|
| ap-001 | YOLO | 0.3 | - | - | Baseline, low style weight |
| ap-002 | YOLO | 0.5 | - | - | Balanced weights |
| ap-003 | YOLO | 0.7 | - | - | High style constraint |
| ap-004 | CLIP | 0.5 | - | - | Different target model |
| ap-005 | YOLO | 0.5 | - | - | Physical test (printed) |

## Deployment & Testing

### Digital Testing
```bash
# Generate patch
python -m pipelines.adversarial.generate_adv_patch \
    --target-model yolov8 \
    --style "punk band patch" \
    --output ./outputs/patches/punk_evasion_v1.png

# Evaluate on test set
python -m pipelines.evaluate.eval_adv_patch \
    --patch ./outputs/patches/punk_evasion_v1.png \
    --test-images ./data/test/coco_persons \
    --metrics evasion,style,robustness
```

### Physical Testing
1. Print patch at 4"x4" on fabric transfer paper
2. Apply to black t-shirt/jacket
3. Photograph with smartphone at various angles/lighting
4. Run through target model
5. Measure evasion rate

## Conclusions
*To be filled after experiment.*

## Next Steps
- [ ] Collect 300-500 punk patch reference images
- [ ] Implement basic adversarial patch generator (no style constraints)
- [ ] Add CLIP-based style guidance
- [ ] Test on YOLO person detector
- [ ] Experiment with different λ_style weights
- [ ] Physical robustness testing (EoT training)
- [ ] Print and photograph tests
- [ ] Human aesthetic evaluation survey
- [ ] Document findings and countermeasures

## Related Work
- [[adversarial_patch_paper]] - Original adversarial patch work (Brown et al. 2017)
- [[physical_adversarial_examples]] - Physical world attacks
- [[clip_paper]] - CLIP for style guidance
- [[robust_physical_perturbations]] - Expectation over Transformation
- Prior art: "Adversarial T-shirt" project
- Style-constrained adversarial examples research

## References
- Adversarial Patch (2017): https://arxiv.org/abs/1712.09665
- Robust Physical Perturbations: https://arxiv.org/abs/1707.08945
- CLIP: https://arxiv.org/abs/2103.00020
