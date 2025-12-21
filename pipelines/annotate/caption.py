#!/usr/bin/env python3
"""
Image Captioner - Generate Captions for Images
===============================================
Generates captions for images using various pre-trained or fine-tuned models.

Usage:
    python caption.py --input-dir ./data/collected --model blip-base
    python caption.py --input-dir ./data/collected --model ./models/my-finetuned
    python caption.py --input-dir ./data/collected --output-format jsonl
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Generator

from PIL import Image
from tqdm import tqdm
import torch

# Support both direct execution and module execution (-m)
sys.path.insert(0, str(Path(__file__).parent))
from models import get_model_config, list_models, MODELS
from utils.manifest import record_annotation_manifest, find_parent_collection_id


class ImageCaptioner:
    """Generate captions for images using various models."""

    def __init__(
        self,
        model_name: str = "blip-base",
        device: str = "auto",
        custom_model_path: Optional[Path] = None
    ):
        self.device = self._get_device(device)
        self.custom_model_path = custom_model_path

        if custom_model_path and custom_model_path.exists():
            print(f"[Captioner] Loading custom model from: {custom_model_path}")
            self._load_custom_model(custom_model_path)
        else:
            self.config = get_model_config(model_name)
            print(f"[Captioner] Loading {self.config.name}: {self.config.model_id}")
            self._load_model()

        print(f"[Captioner] Using device: {self.device}")

    def _get_device(self, device: str) -> str:
        if device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
            return "cpu"
        return device

    def _load_model(self):
        """Load pre-trained model based on config."""
        from transformers import (
            BlipProcessor, BlipForConditionalGeneration,
            Blip2Processor, Blip2ForConditionalGeneration,
            AutoProcessor, AutoModelForCausalLM,
            VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
        )

        model_type = self.config.model_type

        if model_type == "blip":
            self.processor = BlipProcessor.from_pretrained(self.config.model_id)
            self.model = BlipForConditionalGeneration.from_pretrained(
                self.config.model_id,
                torch_dtype=torch.float32
            ).to(self.device)

        elif model_type == "blip2":
            self.processor = Blip2Processor.from_pretrained(self.config.model_id)
            self.model = Blip2ForConditionalGeneration.from_pretrained(
                self.config.model_id,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None
            )
            if self.device != "cuda":
                self.model = self.model.to(self.device)

        elif model_type == "git":
            self.processor = AutoProcessor.from_pretrained(self.config.model_id)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_id,
                torch_dtype=torch.float32
            ).to(self.device)

        elif model_type == "vit-gpt2":
            self.model = VisionEncoderDecoderModel.from_pretrained(
                self.config.model_id
            ).to(self.device)
            self.processor = ViTImageProcessor.from_pretrained(self.config.model_id)
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_id)

        self.model.eval()

    def _load_custom_model(self, model_path: Path):
        """Load a fine-tuned custom model."""
        from transformers import AutoProcessor, AutoModelForCausalLM
        from peft import PeftModel, PeftConfig

        # Check if it's a PEFT model
        peft_config_path = model_path / "adapter_config.json"
        if peft_config_path.exists():
            print("[Captioner] Detected PEFT/LoRA model")
            peft_config = PeftConfig.from_pretrained(str(model_path))
            base_model_id = peft_config.base_model_name_or_path

            self.processor = AutoProcessor.from_pretrained(base_model_id)
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_id,
                torch_dtype=torch.float32
            )
            self.model = PeftModel.from_pretrained(base_model, str(model_path))
        else:
            # Full model checkpoint
            self.processor = AutoProcessor.from_pretrained(str(model_path))
            self.model = AutoModelForCausalLM.from_pretrained(
                str(model_path),
                torch_dtype=torch.float32
            )

        self.model = self.model.to(self.device)
        self.model.eval()
        self.config = type('Config', (), {
            'model_type': 'custom',
            'max_length': 50,
            'num_beams': 4
        })()

    def caption_image(
        self,
        image: Image.Image,
        prompt: Optional[str] = None,
        max_length: Optional[int] = None,
        num_beams: Optional[int] = None
    ) -> str:
        """Generate caption for a single image."""
        max_length = max_length or self.config.max_length
        num_beams = num_beams or self.config.num_beams

        with torch.no_grad():
            if hasattr(self.config, 'model_type') and self.config.model_type == "vit-gpt2":
                pixel_values = self.processor(image, return_tensors="pt").pixel_values.to(self.device)
                output_ids = self.model.generate(
                    pixel_values,
                    max_length=max_length,
                    num_beams=num_beams
                )
                caption = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
            else:
                if prompt:
                    inputs = self.processor(image, prompt, return_tensors="pt").to(self.device)
                else:
                    inputs = self.processor(image, return_tensors="pt").to(self.device)

                output_ids = self.model.generate(
                    **inputs,
                    max_length=max_length,
                    num_beams=num_beams
                )
                caption = self.processor.decode(output_ids[0], skip_special_tokens=True)

        return caption.strip()

    def caption_directory(
        self,
        input_dir: Path,
        output_format: str = "txt",
        prompt: Optional[str] = None,
        recursive: bool = False
    ) -> Generator[dict, None, None]:
        """Caption all images in a directory."""
        image_extensions = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp"}

        if recursive:
            image_files = [
                f for f in input_dir.rglob("*")
                if f.suffix.lower() in image_extensions
            ]
        else:
            image_files = [
                f for f in input_dir.iterdir()
                if f.suffix.lower() in image_extensions
            ]

        print(f"\n[Captioner] Found {len(image_files)} images in {input_dir}")

        for image_path in tqdm(image_files, desc="Captioning"):
            try:
                image = Image.open(image_path).convert("RGB")
                caption = self.caption_image(image, prompt=prompt)

                result = {
                    "file": str(image_path),
                    "filename": image_path.name,
                    "caption": caption,
                    "model": getattr(self.config, 'name', 'custom'),
                    "timestamp": datetime.now().isoformat()
                }

                # Write caption file based on format
                if output_format == "txt":
                    caption_file = image_path.with_suffix(".txt")
                    caption_file.write_text(caption)
                elif output_format == "json":
                    caption_file = image_path.with_suffix(".caption.json")
                    caption_file.write_text(json.dumps(result, indent=2))

                yield result

            except Exception as e:
                print(f"\n[Captioner] Error processing {image_path}: {e}")
                yield {
                    "file": str(image_path),
                    "error": str(e)
                }


def main():
    parser = argparse.ArgumentParser(
        description="Generate captions for images"
    )

    parser.add_argument(
        "--input-dir", "-i",
        type=Path,
        required=True,
        help="Directory containing images to caption"
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="blip-base",
        help="Model name or path to custom model (default: blip-base)"
    )
    parser.add_argument(
        "--output-format", "-f",
        choices=["txt", "json", "jsonl"],
        default="txt",
        help="Output format: txt (sidecar), json (sidecar), jsonl (single file)"
    )
    parser.add_argument(
        "--output-file", "-o",
        type=Path,
        help="Output file for jsonl format"
    )
    parser.add_argument(
        "--prompt", "-p",
        type=str,
        help="Optional prompt for conditional captioning"
    )
    parser.add_argument(
        "--recursive", "-r",
        action="store_true",
        help="Process subdirectories recursively"
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
    parser.add_argument(
        "--collection-id",
        type=str,
        help="Optional parent collection ID (falls back to collection_manifest.json in input-dir)"
    )

    args = parser.parse_args()

    if args.list_models:
        list_models()
        return

    if not args.input_dir.exists():
        print(f"Error: Input directory does not exist: {args.input_dir}")
        sys.exit(1)

    # Check if model is a path or a model name
    custom_path = None
    model_name = args.model
    if "/" in args.model or Path(args.model).exists():
        custom_path = Path(args.model)
        if not custom_path.exists():
            custom_path = None

    captioner = ImageCaptioner(
        model_name=model_name,
        device=args.device,
        custom_model_path=custom_path
    )

    results = []
    for result in captioner.caption_directory(
        args.input_dir,
        output_format=args.output_format if args.output_format != "jsonl" else "txt",
        prompt=args.prompt,
        recursive=args.recursive
    ):
        results.append(result)

    # Write JSONL output if requested
    if args.output_format == "jsonl":
        output_file = args.output_file or args.input_dir / "captions.jsonl"
        with open(output_file, "w") as f:
            for result in results:
                f.write(json.dumps(result) + "\n")
        print(f"\n[Captioner] Saved captions to: {output_file}")

    # Summary
    successful = sum(1 for r in results if "error" not in r)
    print(f"\n{'='*60}")
    print(f"Captioning Complete!")
    print(f"{'='*60}")
    print(f"Processed: {len(results)} images")
    print(f"Successful: {successful}")
    print(f"Failed: {len(results) - successful}")

    # Write annotation manifest
    parent_id = args.collection_id or find_parent_collection_id(args.input_dir)
    record_annotation_manifest(
        name=f"caption-{model_name}",
        input_dir=args.input_dir,
        output_dir=args.input_dir,
        params={
            "model": model_name,
            "prompt": args.prompt,
            "format": args.output_format,
        },
        counts={
            "processed": len(results),
            "successful": successful,
            "failed": len(results) - successful,
        },
        parent_collection_id=parent_id,
        cfg=None,
    )


if __name__ == "__main__":
    main()
