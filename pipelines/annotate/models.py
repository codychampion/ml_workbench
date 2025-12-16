#!/usr/bin/env python3
"""
Model Registry - Supported Captioning Models
=============================================
Registry of available captioning models with their configurations.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelConfig:
    """Configuration for a captioning model."""
    name: str
    model_id: str
    processor_id: Optional[str] = None
    model_type: str = "blip"  # blip, blip2, git, custom
    supports_finetuning: bool = True
    max_length: int = 50
    num_beams: int = 4
    description: str = ""


# Available models registry
MODELS = {
    # BLIP Models
    "blip-base": ModelConfig(
        name="blip-base",
        model_id="Salesforce/blip-image-captioning-base",
        model_type="blip",
        supports_finetuning=True,
        description="BLIP base model - fast, good quality"
    ),
    "blip-large": ModelConfig(
        name="blip-large",
        model_id="Salesforce/blip-image-captioning-large",
        model_type="blip",
        supports_finetuning=True,
        description="BLIP large model - better quality, slower"
    ),

    # BLIP-2 Models
    "blip2-opt": ModelConfig(
        name="blip2-opt",
        model_id="Salesforce/blip2-opt-2.7b",
        model_type="blip2",
        supports_finetuning=True,
        max_length=100,
        description="BLIP-2 with OPT-2.7B - high quality, resource intensive"
    ),
    "blip2-flan-t5": ModelConfig(
        name="blip2-flan-t5",
        model_id="Salesforce/blip2-flan-t5-xl",
        model_type="blip2",
        supports_finetuning=True,
        max_length=100,
        description="BLIP-2 with Flan-T5 XL - instruction-following capable"
    ),

    # GIT Models
    "git-base": ModelConfig(
        name="git-base",
        model_id="microsoft/git-base",
        model_type="git",
        supports_finetuning=True,
        description="Microsoft GIT base - efficient, good for fine-tuning"
    ),
    "git-large": ModelConfig(
        name="git-large",
        model_id="microsoft/git-large",
        model_type="git",
        supports_finetuning=True,
        description="Microsoft GIT large - better quality"
    ),
    "git-large-coco": ModelConfig(
        name="git-large-coco",
        model_id="microsoft/git-large-coco",
        model_type="git",
        supports_finetuning=True,
        description="GIT large fine-tuned on COCO - general purpose"
    ),

    # Specialized Models
    "vit-gpt2": ModelConfig(
        name="vit-gpt2",
        model_id="nlpconnect/vit-gpt2-image-captioning",
        model_type="vit-gpt2",
        supports_finetuning=True,
        description="ViT + GPT-2 - lightweight, fast inference"
    ),
}


def get_model_config(model_name: str) -> ModelConfig:
    """Get model configuration by name."""
    # Handle aliases
    aliases = {
        "blip": "blip-base",
        "blip2": "blip2-opt",
        "git": "git-base",
    }
    model_name = aliases.get(model_name, model_name)

    if model_name not in MODELS:
        available = ", ".join(MODELS.keys())
        raise ValueError(f"Unknown model: {model_name}. Available: {available}")

    return MODELS[model_name]


def list_models() -> None:
    """Print available models."""
    print("\nAvailable Captioning Models:")
    print("=" * 70)
    for name, config in MODELS.items():
        ft = "[finetune]" if config.supports_finetuning else ""
        print(f"  {name:20} {ft:12} {config.description}")
    print("=" * 70)
