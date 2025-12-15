#!/usr/bin/env python3
"""
Image Captioning Pipeline
=========================
Auto-caption collected images using BLIP or BLIP-2 models.
Generates captions suitable for training image generation models.

Usage:
    python caption.py --input-dir ./data/collected --model blip
    python caption.py --input-dir ./data/collected --model blip2 --batch-size 4
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict

from PIL import Image
from tqdm import tqdm
import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config import get_config


class ImageCaptioner:
    """Generate captions for images using vision-language models."""

    SUPPORTED_MODELS = {
        "blip": "Salesforce/blip-image-captioning-base",
        "blip-large": "Salesforce/blip-image-captioning-large",
        "blip2": "Salesforce/blip2-opt-2.7b",
        "git": "microsoft/git-base-coco",
    }

    def __init__(
        self,
        model_name: str = "blip",
        device: str = "cpu",
        batch_size: int = 1
    ):
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size

        self.model = None
        self.processor = None

        print(f"[Captioner] Initializing {model_name} on {device}")

    def load_model(self):
        """Load the captioning model."""
        from transformers import AutoProcessor, AutoModelForVision2Seq

        model_id = self.SUPPORTED_MODELS.get(self.model_name, self.model_name)
        print(f"[Captioner] Loading model: {model_id}")

        try:
            if "blip2" in self.model_name.lower():
                from transformers import Blip2Processor, Blip2ForConditionalGeneration
                self.processor = Blip2Processor.from_pretrained(model_id)
                self.model = Blip2ForConditionalGeneration.from_pretrained(
                    model_id,
                    torch_dtype=torch.float32
                )
            elif "blip" in self.model_name.lower():
                from transformers import BlipProcessor, BlipForConditionalGeneration
                self.processor = BlipProcessor.from_pretrained(model_id)
                self.model = BlipForConditionalGeneration.from_pretrained(model_id)
            elif "git" in self.model_name.lower():
                from transformers import AutoProcessor, AutoModelForCausalLM
                self.processor = AutoProcessor.from_pretrained(model_id)
                self.model = AutoModelForCausalLM.from_pretrained(model_id)
            else:
                # Generic loading
                self.processor = AutoProcessor.from_pretrained(model_id)
                self.model = AutoModelForVision2Seq.from_pretrained(model_id)

            self.model = self.model.to(self.device)
            self.model.eval()
            print(f"[Captioner] Model loaded successfully")

        except Exception as e:
            print(f"[Captioner] Error loading model: {e}")
            raise

    def caption_image(self, image_path: Path, prompt: Optional[str] = None) -> str:
        """Generate caption for a single image."""
        try:
            image = Image.open(image_path).convert("RGB")

            # Prepare inputs
            if prompt and "blip2" in self.model_name.lower():
                inputs = self.processor(image, text=prompt, return_tensors="pt")
            else:
                inputs = self.processor(image, return_tensors="pt")

            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Generate caption
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=50,
                    num_beams=4,
                    early_stopping=True
                )

            caption = self.processor.decode(outputs[0], skip_special_tokens=True)

            # Clean up caption
            caption = caption.strip()
            if caption.startswith("a photo of"):
                caption = caption[len("a photo of"):].strip()

            return caption

        except Exception as e:
            print(f"[Captioner] Error captioning {image_path}: {e}")
            return ""

    def caption_directory(
        self,
        input_dir: Path,
        output_dir: Optional[Path] = None,
        prompt: Optional[str] = None,
        save_format: str = "txt"
    ) -> Dict[str, str]:
        """
        Caption all images in a directory.

        Args:
            input_dir: Directory containing images
            output_dir: Where to save captions (default: same as input)
            prompt: Optional prompt for conditional captioning
            save_format: Format for saving captions (txt, json, both)

        Returns:
            Dictionary mapping image paths to captions
        """
        input_dir = Path(input_dir)
        output_dir = Path(output_dir) if output_dir else input_dir

        # Find all images
        image_extensions = {".jpg", ".jpeg", ".png", ".webp", ".gif"}
        images = [
            f for f in input_dir.rglob("*")
            if f.suffix.lower() in image_extensions
        ]

        if not images:
            print(f"[Captioner] No images found in {input_dir}")
            return {}

        print(f"\n{'='*60}")
        print(f"Captioning {len(images)} images")
        print(f"{'='*60}")
        print(f"Model: {self.model_name}")
        print(f"Device: {self.device}")
        if prompt:
            print(f"Prompt: {prompt}")
        print("-" * 60)

        # Load model if not already loaded
        if self.model is None:
            self.load_model()

        captions = {}

        for image_path in tqdm(images, desc="Captioning"):
            caption = self.caption_image(image_path, prompt)

            if caption:
                captions[str(image_path)] = caption

                # Save individual caption file
                if save_format in ["txt", "both"]:
                    caption_file = image_path.with_suffix(".txt")
                    caption_file.write_text(caption)

        # Save all captions to JSON
        if save_format in ["json", "both"]:
            json_path = output_dir / "captions.json"
            with open(json_path, 'w') as f:
                json.dump(captions, f, indent=2)
            print(f"\nCaptions saved to: {json_path}")

        print(f"\n{'='*60}")
        print(f"Captioning complete: {len(captions)}/{len(images)} images")
        print(f"{'='*60}")

        return captions


class PromptTemplates:
    """Common prompt templates for different use cases."""

    TRAINING = {
        "default": None,  # Let model generate freely
        "detailed": "Describe this image in detail:",
        "style": "Describe the artistic style and composition of this image:",
        "subject": "What is the main subject of this image?",
        "aesthetic": "Describe the aesthetic qualities of this image:",
    }

    @classmethod
    def get(cls, name: str) -> Optional[str]:
        return cls.TRAINING.get(name)


def main():
    parser = argparse.ArgumentParser(
        description="Auto-caption images for ML training datasets"
    )

    parser.add_argument(
        "--input-dir", "-i",
        type=Path,
        required=True,
        help="Directory containing images to caption"
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=Path,
        help="Output directory for captions (default: same as input)"
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="blip",
        choices=list(ImageCaptioner.SUPPORTED_MODELS.keys()),
        help="Captioning model to use (default: blip)"
    )
    parser.add_argument(
        "--prompt", "-p",
        type=str,
        help="Prompt template name or custom prompt"
    )
    parser.add_argument(
        "--format", "-f",
        choices=["txt", "json", "both"],
        default="both",
        help="Output format (default: both)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to run model on (default: cpu)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for processing (default: 1)"
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List available models and exit"
    )
    parser.add_argument(
        "--list-prompts",
        action="store_true",
        help="List prompt templates and exit"
    )

    args = parser.parse_args()

    if args.list_models:
        print("Available captioning models:")
        for name, model_id in ImageCaptioner.SUPPORTED_MODELS.items():
            print(f"  {name}: {model_id}")
        return

    if args.list_prompts:
        print("Available prompt templates:")
        for name, prompt in PromptTemplates.TRAINING.items():
            print(f"  {name}: {prompt or '(no prompt - free generation)'}")
        return

    # Get prompt
    prompt = None
    if args.prompt:
        prompt = PromptTemplates.get(args.prompt) or args.prompt

    # Initialize captioner
    captioner = ImageCaptioner(
        model_name=args.model,
        device=args.device,
        batch_size=args.batch_size
    )

    # Run captioning
    captions = captioner.caption_directory(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        prompt=prompt,
        save_format=args.format
    )

    print("\nNext steps:")
    print("  1. Review captions in .txt files alongside images")
    print("  2. Create FiftyOne dataset: python projects/data-collector/create_dataset.py")
    print("  3. Use for training with the caption files")


if __name__ == "__main__":
    main()
