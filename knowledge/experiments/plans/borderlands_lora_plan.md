---
type: experiment-plan
exp_id: "borderlands-lora-001"
created: "2026-01-01"
status: draft
tags: [lora, style-transfer, borderlands, image-generation, fine-tuning]
dataset: "[[borderlands_dataset]]"
model_base: stable-diffusion-xl
---

# Experiment: Borderlands-Style Art Generation with LoRA

## Hypothesis

Fine-tuning Stable Diffusion XL with LoRA adapters on Borderlands game artwork will enable generation of images with the distinctive cell-shaded, comic book art style characteristic of the Borderlands franchise.

Key characteristics to capture:
- Thick black outlines (cel-shading)
- High contrast colors
- Hand-painted texture aesthetic
- Comic book/graphic novel style
- Borderlands-specific character and environment design patterns

## Dataset

- **Name:** Borderlands Style Dataset
- **Source:** Mixed (Reddit scraping, official artwork, fan art, game screenshots)
- **Target Size:** 500-1000 high-quality images
- **Collection Strategy:**
  - Reddit: r/borderlands, r/borderlands2, r/borderlands3, r/borderlandsart
  - Official Gearbox promotional art
  - Game screenshots (curated for art style, not gameplay)
- **Path:** `./data/collected/borderlands_style/`
- **Preprocessing:**
  - Filter for artistic style (exclude pure gameplay screenshots)
  - Minimum resolution: 1024x1024
  - Caption with descriptors: "borderlands style", "cell-shaded", "comic book art"

## Method

### Model Architecture
- **Base Model:** Stable Diffusion XL 1.0
- **Adapter:** LoRA (Low-Rank Adaptation)
- **LoRA Rank:** 32-64 (experiment with both)
- **Target Modules:** All cross-attention layers

### Training Strategy
1. **Data Collection:**
   ```bash
   python -m pipelines.collect.collect_reddit \
     --subreddit "borderlands,borderlands2,borderlands3,borderlandsart" \
     --limit 500 --sort top --time all --images-only
   ```

2. **Captioning:**
   ```bash
   python -m pipelines.annotate.caption \
     --input ./data/collected/borderlands/ \
     --model blip-large \
     --prompt "Describe this Borderlands-style artwork"
   ```

3. **Caption Enhancement:**
   - Prepend "borderlands style, cell-shaded, comic book art" to all captions
   - Add specific tags: character names, environments, color schemes

4. **LoRA Training:**
   ```bash
   python -m pipelines.train.train_lora \
     --base-model stabilityai/stable-diffusion-xl-base-1.0 \
     --dataset ./data/collected/borderlands/ \
     --lora-rank 32 \
     --epochs 20 \
     --batch-size 4 \
     --learning-rate 1e-4 \
     --output ./models/borderlands-lora-r32
   ```

### Hyperparameters
- Learning rate: 1e-4 (with cosine decay)
- Batch size: 4 (gradient accumulation if needed)
- Epochs: 15-20
- LoRA alpha: 32
- Mixed precision: fp16
- Gradient checkpointing: enabled

## Success Criteria

### Quantitative
- **CLIP Score:** > 0.75 on held-out Borderlands test set
- **FID (Fréchet Inception Distance):** < 50 vs real Borderlands artwork
- **Style Consistency:** > 85% human evaluators identify as Borderlands style

### Qualitative
- [ ] Generates images with characteristic thick black outlines
- [ ] Maintains high contrast color palette
- [ ] Captures hand-painted texture aesthetic
- [ ] Style works across diverse subjects (characters, environments, objects)
- [ ] No mode collapse - generates variety within style

### Inference Tests
Generate images with prompts like:
- "a bandit in borderlands style"
- "borderlands style psycho mask"
- "pandora wasteland in borderlands style, cell-shaded"
- "vault hunter in borderlands style comic book art"

## Pipeline Integration

### Config File
Create `conf/pipeline/train_borderlands_lora.yaml`:
```yaml
train_lora:
  base_model: "stabilityai/stable-diffusion-xl-base-1.0"
  dataset_path: "${paths.data.collected}/borderlands"
  output_dir: "${paths.models}/borderlands-lora"

  lora:
    rank: 32
    alpha: 32
    target_modules: ["to_q", "to_k", "to_v", "to_out"]

  training:
    epochs: 20
    batch_size: 4
    learning_rate: 1e-4
    lr_scheduler: "cosine"
    warmup_steps: 100

  evaluation:
    validation_prompts:
      - "borderlands style bandit with mask"
      - "borderlands style vault hunter"
      - "pandora landscape in borderlands style"
    eval_every_n_epochs: 2
```

## Runs

| Run ID | LoRA Rank | Epochs | LR | Status | CLIP Score | Notes |
|--------|-----------|--------|-----|--------|------------|-------|
| bl-001 | 32 | 15 | 1e-4 | - | - | Baseline run |
| bl-002 | 64 | 15 | 1e-4 | - | - | Higher rank test |
| bl-003 | 32 | 20 | 5e-5 | - | - | Lower LR, more epochs |

## Deployment

Once trained, the LoRA can be used with:
```python
from diffusers import StableDiffusionXLPipeline
import torch

pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16
)
pipe.load_lora_weights("./models/borderlands-lora-r32")
pipe.to("cuda")

image = pipe(
    "a psycho bandit in borderlands style, cell-shaded comic book art",
    num_inference_steps=30
).images[0]
```

## Conclusions
*To be filled after experiment.*

## Next Steps
- [ ] Collect 500+ Borderlands-style images
- [ ] Caption dataset with style descriptors
- [ ] Run baseline LoRA training (rank=32, 15 epochs)
- [ ] Evaluate generated samples against success criteria
- [ ] Experiment with different LoRA ranks
- [ ] Fine-tune caption prepending strategy
- [ ] Register best checkpoint in model registry
- [ ] Document inference examples in knowledge base

## Related Work
- [[stable_diffusion_xl_paper]] - Base model architecture
- [[lora_paper]] - LoRA adaptation technique
- [[dreambooth_paper]] - Alternative fine-tuning approach
- Style transfer LoRAs in the community (reference implementations)
