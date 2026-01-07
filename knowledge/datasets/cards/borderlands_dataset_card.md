---
type: dataset
name: "Borderlands Style Art"
source: reddit
collection_status: planned
used_in_experiments: ["[[borderlands_lora_plan]]"]
tags: [style-transfer, cell-shaded, comic-art, borderlands, game-art]
---

# Dataset: Borderlands Style Art

## Overview
Collection of images showcasing the distinctive Borderlands visual style for training LoRA adapters on Stable Diffusion XL.

## Style Characteristics
- **Cell-shaded rendering:** Thick black outlines separating color regions
- **Comic book aesthetic:** Hand-painted texture look, graphic novel style
- **High contrast colors:** Vibrant, saturated color palette
- **Borderlands-specific:** Character designs, environments, UI elements from the game franchise

## Collection Strategy

### Primary Sources
- **Reddit Subreddits:**
  - r/borderlands - Official franchise subreddit
  - r/borderlands2 - Borderlands 2 specific
  - r/borderlands3 - Borderlands 3 specific
  - r/borderlandsart - Fan art and creative works
- **Filter:** Top/best posts, images only, exclude pure gameplay screenshots

### Target Composition
- Official promotional art: ~20%
- High-quality fan art: ~40%
- Game screenshots (artistic scenes): ~30%
- Character designs and concept art: ~10%

### Quality Criteria
- Minimum resolution: 1024x1024 (SDXL native)
- Clear cell-shading visible
- Good representation of Borderlands aesthetic
- No UI overlays or watermarks (when possible)

## Collection Command

```bash
python -m pipelines.collect.collect_reddit \
  --subreddit "borderlands,borderlands2,borderlands3,borderlandsart" \
  --limit 500 \
  --sort top \
  --time all \
  --images-only \
  --output-dir ./data/collected/borderlands
```

## Target Statistics
- **Size:** 500-1000 images
- **Resolution:** 1024x1024 minimum, up to 2048x2048
- **Format:** PNG/JPEG
- **Storage:** ~2-4 GB

## Preprocessing

### Captioning Strategy
1. Auto-caption with BLIP-Large
2. Prepend style descriptor: "borderlands style, cell-shaded, comic book art"
3. Add specific tags:
   - Character names (if identifiable): "Maya", "Lilith", "Handsome Jack"
   - Environment types: "Pandora wasteland", "vault", "bandit camp"
   - Color themes: "neon purple", "desert orange", "vault teal"

### Example Caption
```
borderlands style, cell-shaded comic book art, psycho bandit with mask,
neon purple lighting, pandora wasteland background, high contrast colors
```

## Experiments Using This Dataset
- [[borderlands_lora_plan]] - LoRA fine-tuning for style transfer

## Related Datasets
- [[punk_patches_dataset_card]] - Similar style-focused collection
- General Stable Diffusion training sets (for comparison)

## Metadata
- **Collection Date:** TBD
- **Last Updated:** 2026-01-01
- **License:** Mixed (Reddit content, fair use for research/training)
- **Path:** `./data/collected/borderlands/`

## Known Issues & Considerations
- Game screenshots may have UI elements (filter during curation)
- Fan art quality varies (manual review recommended)
- Some images may be too gameplay-focused vs. artistic
- Copyright considerations for commercial use of trained model

## Sample Images
*To be added during collection*
