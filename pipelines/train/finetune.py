#!/usr/bin/env python3
"""
Fine-tune Captioning Models
===========================
Fine-tune image captioning models on custom datasets using LoRA/PEFT.

Usage:
    python finetune.py --dataset ./data/annotated --model blip-base --output ./models/my-captioner
    python finetune.py --dataset ./data/annotated --model git-base --lora-rank 16
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import torch
from PIL import Image
from tqdm import tqdm

# Add pipelines directory to path for cross-stage imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from annotate.models import get_model_config, list_models


class CaptionDataset(torch.utils.data.Dataset):
    """Dataset for image-caption pairs."""

    def __init__(
        self,
        data_dir: Path,
        processor,
        max_length: int = 50,
        caption_ext: str = ".txt"
    ):
        self.data_dir = Path(data_dir)
        self.processor = processor
        self.max_length = max_length

        # Find all images with captions
        image_extensions = {".jpg", ".jpeg", ".png", ".webp"}
        self.samples = []

        for img_path in self.data_dir.rglob("*"):
            if img_path.suffix.lower() in image_extensions:
                caption_path = img_path.with_suffix(caption_ext)
                if caption_path.exists():
                    self.samples.append((img_path, caption_path))

        print(f"[Dataset] Found {len(self.samples)} image-caption pairs")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, caption_path = self.samples[idx]

        image = Image.open(img_path).convert("RGB")
        caption = caption_path.read_text().strip()

        # Process image and text
        encoding = self.processor(
            images=image,
            text=caption,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

        # Remove batch dimension
        encoding = {k: v.squeeze(0) for k, v in encoding.items()}

        # Add labels for language modeling
        encoding["labels"] = encoding["input_ids"].clone()

        return encoding


class CaptionFineTuner:
    """Fine-tune captioning models with LoRA."""

    def __init__(
        self,
        model_name: str = "blip-base",
        device: str = "auto",
        use_lora: bool = True,
        lora_rank: int = 8,
        lora_alpha: int = 16
    ):
        self.device = self._get_device(device)
        self.config = get_model_config(model_name)
        self.use_lora = use_lora
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha

        print(f"[FineTuner] Loading {self.config.name}: {self.config.model_id}")
        self._load_model()
        print(f"[FineTuner] Using device: {self.device}")

    def _get_device(self, device: str) -> str:
        if device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
            return "cpu"
        return device

    def _load_model(self):
        """Load model and optionally apply LoRA."""
        from transformers import (
            BlipProcessor, BlipForConditionalGeneration,
            AutoProcessor, AutoModelForCausalLM
        )

        model_type = self.config.model_type

        if model_type == "blip":
            self.processor = BlipProcessor.from_pretrained(self.config.model_id)
            self.model = BlipForConditionalGeneration.from_pretrained(
                self.config.model_id,
                torch_dtype=torch.float32
            )
        elif model_type == "git":
            self.processor = AutoProcessor.from_pretrained(self.config.model_id)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_id,
                torch_dtype=torch.float32
            )
        else:
            raise ValueError(f"Fine-tuning not yet supported for {model_type}")

        if self.use_lora:
            self._apply_lora()

        self.model = self.model.to(self.device)

    def _apply_lora(self):
        """Apply LoRA adapters to the model."""
        from peft import LoraConfig, get_peft_model, TaskType

        print(f"[FineTuner] Applying LoRA (rank={self.lora_rank}, alpha={self.lora_alpha})")

        # Configure LoRA for the text decoder
        if self.config.model_type == "blip":
            target_modules = ["q_proj", "v_proj", "k_proj", "out_proj"]
        else:
            target_modules = ["q_proj", "v_proj"]

        lora_config = LoraConfig(
            r=self.lora_rank,
            lora_alpha=self.lora_alpha,
            target_modules=target_modules,
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.CAUSAL_LM
        )

        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()

    def train(
        self,
        train_dataset: CaptionDataset,
        output_dir: Path,
        epochs: int = 3,
        batch_size: int = 4,
        learning_rate: float = 1e-4,
        warmup_steps: int = 100,
        save_steps: int = 500,
        eval_dataset: Optional[CaptionDataset] = None
    ):
        """Train the model."""
        from transformers import TrainingArguments, Trainer

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        training_args = TrainingArguments(
            output_dir=str(output_dir),
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            warmup_steps=warmup_steps,
            learning_rate=learning_rate,
            weight_decay=0.01,
            logging_dir=str(output_dir / "logs"),
            logging_steps=10,
            save_steps=save_steps,
            save_total_limit=3,
            evaluation_strategy="steps" if eval_dataset else "no",
            eval_steps=save_steps if eval_dataset else None,
            load_best_model_at_end=True if eval_dataset else False,
            fp16=self.device == "cuda",
            dataloader_num_workers=0,
            remove_unused_columns=False,
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
        )

        print(f"\n[FineTuner] Starting training...")
        print(f"  Epochs: {epochs}")
        print(f"  Batch size: {batch_size}")
        print(f"  Learning rate: {learning_rate}")
        print(f"  Output: {output_dir}")

        trainer.train()

        # Save final model
        print(f"\n[FineTuner] Saving model to {output_dir}")
        if self.use_lora:
            self.model.save_pretrained(str(output_dir))
        else:
            trainer.save_model(str(output_dir))

        self.processor.save_pretrained(str(output_dir))

        # Save training config
        config_file = output_dir / "training_config.json"
        config_file.write_text(json.dumps({
            "base_model": self.config.model_id,
            "model_type": self.config.model_type,
            "use_lora": self.use_lora,
            "lora_rank": self.lora_rank,
            "lora_alpha": self.lora_alpha,
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "trained_at": datetime.now().isoformat()
        }, indent=2))

        print(f"[FineTuner] Training complete!")
        return output_dir


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune image captioning models"
    )

    parser.add_argument(
        "--dataset", "-d",
        type=Path,
        required=True,
        help="Directory with images and .txt caption files"
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="blip-base",
        help="Base model to fine-tune (default: blip-base)"
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=Path("./outputs/captioner"),
        help="Output directory for fine-tuned model"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs (default: 3)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Training batch size (default: 4)"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
        help="Learning rate (default: 1e-4)"
    )
    parser.add_argument(
        "--no-lora",
        action="store_true",
        help="Disable LoRA, do full fine-tuning"
    )
    parser.add_argument(
        "--lora-rank",
        type=int,
        default=8,
        help="LoRA rank (default: 8)"
    )
    parser.add_argument(
        "--lora-alpha",
        type=int,
        default=16,
        help="LoRA alpha (default: 16)"
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda", "mps"],
        default="auto",
        help="Device to use (default: auto)"
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List available models and exit"
    )

    args = parser.parse_args()

    if args.list_models:
        list_models()
        return

    if not args.dataset.exists():
        print(f"Error: Dataset directory does not exist: {args.dataset}")
        sys.exit(1)

    # Initialize fine-tuner
    finetuner = CaptionFineTuner(
        model_name=args.model,
        device=args.device,
        use_lora=not args.no_lora,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha
    )

    # Create dataset
    train_dataset = CaptionDataset(
        data_dir=args.dataset,
        processor=finetuner.processor,
        max_length=finetuner.config.max_length
    )

    if len(train_dataset) == 0:
        print("Error: No image-caption pairs found in dataset")
        print("Expected: images with matching .txt caption files")
        sys.exit(1)

    # Train
    finetuner.train(
        train_dataset=train_dataset,
        output_dir=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate
    )

    print(f"\n{'='*60}")
    print("Fine-tuning Complete!")
    print(f"{'='*60}")
    print(f"Model saved to: {args.output}")
    print(f"\nTo use your fine-tuned model:")
    print(f"  python caption.py --model {args.output} --input-dir ./data/images")


if __name__ == "__main__":
    main()
