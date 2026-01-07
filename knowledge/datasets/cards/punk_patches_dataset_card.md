---
type: dataset
name: "Punk Band Patches"
source: reddit
collection_status: planned
used_in_experiments: ["[[adversarial_punk_patches_plan]]"]
tags: [punk, patches, DIY-aesthetic, style-reference, adversarial-ml]
---

# Dataset: Punk Band Patches Reference

## Overview
Collection of punk band patches and DIY patch aesthetics for use as style constraints in adversarial patch generation. This dataset serves as a visual reference to ensure adversarial patches maintain authentic punk aesthetic.

## Aesthetic Characteristics
- **High contrast:** Primarily black and white, bold graphics
- **DIY aesthetic:** Screen-printed, photocopied, hand-drawn look
- **Punk typography:** Stencil fonts, handwritten style, all-caps
- **Common motifs:**
  - Band logos and names
  - Skulls and crossbones
  - Anarchy symbols
  - Safety pins, studs
  - Anti-establishment imagery
- **Texture:** Rough edges, distressed look, fabric grain
- **Production feel:** Screen print, iron-on, sewn patches

## Collection Strategy

### Primary Sources
- **Reddit Communities:**
  - r/BattleJackets - Battle jackets/vests with patches
  - r/punk - General punk culture
  - r/crustpunk - Crust punk specific
  - r/DIYpunk - DIY punk projects
- **Filter:** Focus on patch close-ups, high-quality images

### Target Composition
- Band logo patches: ~40%
- Political/statement patches: ~20%
- Generic punk symbols: ~20%
- DIY/handmade patches: ~15%
- Vintage patches: ~5%

### Quality Criteria
- Clear, focused images of patches
- Good lighting (shows details)
- Minimal background distractions
- Authentic DIY aesthetic (not mass-produced modern patches)
- Resolution: 512x512 minimum

## Collection Command

```bash
python -m pipelines.collect.collect_reddit \
  --subreddit "battlejackets,punk,crustpunk,DIYpunk" \
  --limit 300 \
  --sort top \
  --time all \
  --images-only \
  --output-dir ./data/collected/punk_patches
```

## Target Statistics
- **Size:** 300-500 images
- **Resolution:** 512x512 to 1024x1024
- **Format:** PNG/JPEG
- **Storage:** ~500 MB - 1 GB

## Usage in Adversarial Patch Generation

### Style Constraint Methods

**Method 1: CLIP Embeddings**
```python
# Use CLIP to ensure patch looks like punk aesthetic
style_prompt = "punk rock band patch, DIY aesthetic, black and white, screen printed"
clip_similarity = clip_score(patch, style_prompt)
```

**Method 2: Reference Image Matching**
```python
# Match visual features to reference punk patches
reference_features = extract_features(punk_patch_dataset)
patch_features = extract_features(generated_patch)
style_loss = feature_distance(patch_features, reference_features)
```

**Method 3: Diffusion Prior**
```python
# Use diffusion model trained on punk patches as prior
diffusion_model = load_punk_patch_diffusion()
prior_loss = diffusion_model.score(patch)
```

## Preprocessing

### Caption Generation
Not needed for adversarial patch generation (used as visual reference only), but could be useful for:
- CLIP similarity scoring
- Style description extraction
- Automatic categorization

### Feature Extraction
For style matching:
- Extract CLIP embeddings
- Extract VGG/ResNet features
- Color histogram analysis
- Edge detection patterns (for black outlines)

## Experiments Using This Dataset
- [[adversarial_punk_patches_plan]] - Style-constrained adversarial patches

## Related Datasets
- [[borderlands_dataset_card]] - Another style-focused collection
- General adversarial patch test sets

## Metadata
- **Collection Date:** TBD
- **Last Updated:** 2026-01-01
- **License:** Mixed (Reddit content, fair use for research)
- **Path:** `./data/collected/punk_patches/`

## Ethical Considerations

This dataset supports research with dual-use implications:

**Defensive Research:**
- Testing model robustness against adversarial attacks
- Understanding limitations of vision systems
- Improving security of ML models

**Privacy Applications:**
- Researching privacy-preserving clothing/accessories
- Academic study of surveillance evasion
- Art projects on AI and surveillance

**Important:** Use responsibly. See [[adversarial_punk_patches_plan]] for full ethical discussion.

## Sample Characteristics

### Typical Punk Patch Elements
- Band name in bold typography
- High contrast (black background, white text/graphics)
- Rough, photocopied aesthetic
- DIY production quality
- Subcultural symbolism

### Color Distribution
- Black: ~60-70% (backgrounds, outlines)
- White: ~20-30% (text, highlights)
- Limited color: ~10% (some patches use red, yellow)

## Known Issues & Considerations
- Modern commercial patches look different (too clean)
- Need to filter for authentic DIY aesthetic
- Some images may show whole jackets (need patch cropping)
- Lighting variations may affect style transfer
- Fabric texture visible in photos (feature or bug?)

## Future Enhancements
- Collect physical patches for scanning
- 3D fabric texture dataset
- Temporal evolution of punk patch aesthetics
- Regional variations (UK punk vs. US hardcore)
