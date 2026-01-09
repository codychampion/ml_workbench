#!/usr/bin/env python3
"""
Image Captioning Script for LoRA Training
==========================================
Generate captions/tags for images in a directory using BLIP or CLIP.

Usage:
    python -m pipelines.infer.caption_images --input ./data/collected/reddit/borderlands
"""

import argparse
import sys
from pathlib import Path
from typing import List
import torch
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def caption_with_blip(image_paths: List[Path], device: str = "cuda") -> List[str]:
    """Caption images using BLIP."""
    from transformers import BlipProcessor, BlipForConditionalGeneration

    print("Loading BLIP model...")
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    model = model.to(device)
    model.eval()

    captions = []
    print(f"\nCaptioning {len(image_paths)} images with BLIP...")

    for img_path in tqdm(image_paths):
        try:
            image = Image.open(img_path).convert('RGB')
            inputs = processor(image, return_tensors="pt").to(device)

            with torch.no_grad():
                out = model.generate(**inputs, max_length=75)

            caption = processor.decode(out[0], skip_special_tokens=True)
            captions.append(caption)

        except Exception as e:
            print(f"\nError captioning {img_path}: {e}")
            captions.append("")

    return captions


def caption_with_clip_interrogator(image_paths: List[Path], device: str = "cuda") -> List[str]:
    """Caption images using CLIP Interrogator (SD-style prompts)."""
    try:
        import clip
    except ImportError:
        raise ImportError("CLIP not installed. This should not happen in the infer container.")

    print("Loading CLIP model...")
    model, preprocess = clip.load("ViT-L/14", device=device)
    model.eval()

    # Simple CLIP-based captioning (tags most likely concepts)
    concepts = [
        "photo", "artwork", "digital art", "painting", "drawing", "screenshot",
        "anime", "cartoon", "realistic", "stylized", "3d render",
        "character", "landscape", "portrait", "scene", "object",
        "colorful", "dark", "bright", "moody", "vibrant"
    ]

    captions = []
    print(f"\nCaptioning {len(image_paths)} images with CLIP...")

    for img_path in tqdm(image_paths):
        try:
            image = Image.open(img_path).convert('RGB')
            image_input = preprocess(image).unsqueeze(0).to(device)
            text_inputs = clip.tokenize(concepts).to(device)

            with torch.no_grad():
                image_features = model.encode_image(image_input)
                text_features = model.encode_text(text_inputs)

                # Normalize
                image_features /= image_features.norm(dim=-1, keepdim=True)
                text_features /= text_features.norm(dim=-1, keepdim=True)

                # Calculate similarity
                similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)

                # Get top 3 concepts
                values, indices = similarity[0].topk(3)
                top_concepts = [concepts[idx] for idx in indices]
                caption = ", ".join(top_concepts)
                captions.append(caption)

        except Exception as e:
            print(f"\nError captioning {img_path}: {e}")
            captions.append("")

    return captions


def save_captions(image_paths: List[Path], captions: List[str], format: str = "txt"):
    """Save captions next to images."""
    print(f"\nSaving captions in .{format} format...")

    for img_path, caption in zip(image_paths, captions):
        if caption:
            caption_path = img_path.with_suffix(f".{format}")
            with open(caption_path, 'w', encoding='utf-8') as f:
                f.write(caption)

    print(f"✓ Saved {len(captions)} caption files")


def main():
    parser = argparse.ArgumentParser(description="Caption images for LoRA training")
    parser.add_argument("--input", type=Path, required=True, help="Input directory with images")
    parser.add_argument("--model", choices=["blip", "clip"], default="blip",
                       help="Captioning model (default: blip)")
    parser.add_argument("--format", choices=["txt", "caption"], default="txt",
                       help="Caption file format (default: txt)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                       help="Device to use (default: cuda if available)")
    parser.add_argument("--overwrite", action="store_true",
                       help="Overwrite existing caption files")

    args = parser.parse_args()

    # Find images
    image_extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    image_paths = [
        p for p in args.input.rglob("*")
        if p.suffix.lower() in image_extensions
    ]

    if not image_paths:
        print(f"Error: No images found in {args.input}")
        return

    print(f"\n{'='*60}")
    print(f"Image Captioning for LoRA Training")
    print(f"{'='*60}")
    print(f"Input: {args.input}")
    print(f"Images: {len(image_paths)}")
    print(f"Model: {args.model}")
    print(f"Device: {args.device}")

    # Filter out images that already have captions (unless overwrite)
    if not args.overwrite:
        uncaptioned = []
        for img_path in image_paths:
            caption_path = img_path.with_suffix(f".{args.format}")
            if not caption_path.exists():
                uncaptioned.append(img_path)

        if len(uncaptioned) < len(image_paths):
            print(f"Skipping {len(image_paths) - len(uncaptioned)} already captioned images")
            image_paths = uncaptioned

        if not image_paths:
            print("All images already have captions. Use --overwrite to regenerate.")
            return

    # Caption images
    if args.model == "blip":
        captions = caption_with_blip(image_paths, args.device)
    else:
        captions = caption_with_clip_interrogator(image_paths, args.device)

    # Save captions
    save_captions(image_paths, captions, args.format)

    print(f"\n{'='*60}")
    print("Captioning Complete!")
    print(f"{'='*60}")
    print(f"Caption files saved next to images in {args.input}")
    print("\nReady for LoRA training in ComfyUI!")


if __name__ == "__main__":
    main()
