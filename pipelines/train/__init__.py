"""
Pipeline Stage 3: Model Training
================================
Fine-tuning and training models with LoRA/PEFT.

Trains:
- Captioner models (BLIP, GIT) -> used by annotate/
- LoRA adapters (SDXL, Flux) -> used by infer/

Inputs: data/annotated/
Outputs: models/
"""
