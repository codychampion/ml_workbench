#!/usr/bin/env python3
"""
HunyuanVideo LoRA Training with Musubi Tuner
=============================================
Trains ComfyUI-compatible LoRA weights using Kohya-ss Musubi Tuner.

This wrapper handles the full workflow:
1. Cache latents (VAE encoding)
2. Cache text embeddings
3. Train LoRA adapter
4. Output safetensors file for ComfyUI
"""

import argparse
import subprocess
import json
from pathlib import Path
import shutil
import sys

def prepare_musubi_dataset(dataset_dir: Path, output_dir: Path, concept: str):
    """
    Prepare dataset in Musubi Tuner format.

    Musubi expects:
    - dataset/
      - img/
        - 001.jpg, 002.jpg, ...
      - meta_lat.json (or will be generated during caching)
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    img_dir = output_dir / "img"
    img_dir.mkdir(exist_ok=True)

    # Copy/link images with numbered filenames
    image_extensions = {".jpg", ".jpeg", ".png", ".webp"}
    images = []

    for img_path in sorted(dataset_dir.rglob("*")):
        if img_path.suffix.lower() in image_extensions:
            images.append(img_path)

    # Create numbered symlinks
    for idx, img_path in enumerate(images, start=1):
        dest = img_dir / f"{idx:03d}{img_path.suffix}"
        if dest.exists():
            dest.unlink()
        dest.symlink_to(img_path.absolute())

    print(f"[Dataset] Prepared {len(images)} images for Musubi Tuner")

    # Create meta_lat.json for unconditional training
    # Each image gets an empty caption
    meta_lat = {
        f"{idx:03d}{img_path.suffix}": {
            "caption": ""
        }
        for idx, img_path in enumerate(images, start=1)
    }

    meta_file = output_dir / "meta_lat.json"
    meta_file.write_text(json.dumps(meta_lat, indent=2))

    return output_dir, len(images)

def run_latent_caching(musubi_dir: Path, dataset_dir: Path, model_name: str):
    """
    Run latent caching (VAE encoding) for the dataset.
    """
    print("\n[Musubi] Step 1/3: Caching latents (VAE encoding)...")

    # Find the latent caching script
    cache_script = musubi_dir / "finetune" / "cache_latents.py"
    if not cache_script.exists():
        # Try alternative location
        cache_script = musubi_dir / "cache_latents.py"

    if not cache_script.exists():
        print(f"[ERROR] Could not find cache_latents.py in {musubi_dir}")
        return False

    cmd = [
        "accelerate", "launch",
        "--num_processes=1",
        "--num_machines=1",
        "--mixed_precision=bf16",
        str(cache_script),
        f"--pretrained_model_name_or_path={model_name}",
        f"--in_json={dataset_dir / 'meta_lat.json'}",
        "--full_bf16"
    ]

    try:
        result = subprocess.run(cmd, check=True, cwd=str(musubi_dir))
        print("[Musubi] ✅ Latent caching complete")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Latent caching failed: {e}")
        return False

def run_text_encoder_caching(musubi_dir: Path, dataset_dir: Path, model_name: str):
    """
    Run text encoder caching for the dataset.
    """
    print("\n[Musubi] Step 2/3: Caching text encoder outputs...")

    cache_script = musubi_dir / "finetune" / "cache_text_encoder_outputs.py"
    if not cache_script.exists():
        cache_script = musubi_dir / "cache_text_encoder_outputs.py"

    if not cache_script.exists():
        print(f"[ERROR] Could not find cache_text_encoder_outputs.py in {musubi_dir}")
        return False

    cmd = [
        "accelerate", "launch",
        "--num_processes=1",
        "--num_machines=1",
        str(cache_script),
        f"--pretrained_model_name_or_path={model_name}",
        f"--in_json={dataset_dir / 'meta_lat.json'}",
        "--full_bf16"
    ]

    try:
        result = subprocess.run(cmd, check=True, cwd=str(musubi_dir))
        print("[Musubi] ✅ Text encoder caching complete")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Text encoder caching failed: {e}")
        return False

def run_lora_training(
    musubi_dir: Path,
    dataset_dir: Path,
    output_dir: Path,
    model_name: str,
    num_images: int,
    epochs: int,
    lora_rank: int,
    lora_alpha: int,
    learning_rate: float,
    batch_size: int
):
    """
    Run LoRA training with Musubi Tuner.
    """
    print("\n[Musubi] Step 3/3: Training LoRA adapter...")

    train_script = musubi_dir / "finetune" / "train_network.py"
    if not train_script.exists():
        train_script = musubi_dir / "train_network.py"

    if not train_script.exists():
        print(f"[ERROR] Could not find train_network.py in {musubi_dir}")
        return False

    # Calculate training steps
    steps_per_epoch = num_images // batch_size
    max_train_steps = steps_per_epoch * epochs

    cmd = [
        "accelerate", "launch",
        "--num_processes=1",
        "--num_machines=1",
        "--mixed_precision=bf16",
        str(train_script),
        f"--pretrained_model_name_or_path={model_name}",
        f"--in_json={dataset_dir / 'meta_lat.json'}",
        f"--output_dir={output_dir}",
        f"--output_name=lora",
        "--network_module=networks.lora",
        f"--network_dim={lora_rank}",
        f"--network_alpha={lora_alpha}",
        f"--learning_rate={learning_rate}",
        f"--max_train_steps={max_train_steps}",
        f"--train_batch_size={batch_size}",
        "--save_model_as=safetensors",
        "--mixed_precision=bf16",
        "--save_precision=bf16",
        "--cache_latents",
        "--cache_text_encoder_outputs",
        "--gradient_checkpointing",
        f"--save_every_n_epochs={max(1, epochs // 4)}",
        "--logging_dir=logs",
        "--log_with=tensorboard",
        "--seed=42",
        "--full_bf16"
    ]

    try:
        result = subprocess.run(cmd, check=True, cwd=str(musubi_dir))
        print(f"[Musubi] ✅ LoRA training complete!")
        print(f"[Musubi] Output: {output_dir / 'lora.safetensors'}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] LoRA training failed: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="HunyuanVideo LoRA Training with Musubi Tuner")

    parser.add_argument("--dataset", "-d", type=Path, required=True, help="Dataset directory")
    parser.add_argument("--concept", "-c", type=str, required=True, help="Concept name")
    parser.add_argument("--model", type=str, default="tencent/HunyuanVideo",
                       help="Model name or path (default: tencent/HunyuanVideo)")
    parser.add_argument("--output", "-o", type=Path, default=None, help="Output directory")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=1, help="Training batch size")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--lora-rank", type=int, default=8, help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=16, help="LoRA alpha")
    parser.add_argument("--skip-cache", action="store_true",
                       help="Skip latent/text caching (if already cached)")

    args = parser.parse_args()

    if args.output is None:
        args.output = Path(f"./outputs/lora/{args.concept}")

    args.output.mkdir(parents=True, exist_ok=True)

    print(f"{'='*60}")
    print(f"HunyuanVideo LoRA Training with Musubi Tuner")
    print(f"{'='*60}")
    print(f"Concept: {args.concept}")
    print(f"Dataset: {args.dataset}")
    print(f"Model: {args.model}")
    print(f"Output: {args.output}")
    print(f"Epochs: {args.epochs}")
    print(f"LoRA rank: {args.lora_rank}")
    print(f"{'='*60}\n")

    # Check if Musubi Tuner is installed
    musubi_dir = Path("/opt/musubi-tuner")
    if not musubi_dir.exists():
        print(f"[ERROR] Musubi Tuner not found at {musubi_dir}")
        print(f"[ERROR] Please rebuild Docker image: docker compose build train")
        sys.exit(1)

    # Prepare dataset
    print("[Dataset] Preparing dataset for Musubi Tuner...")
    prepared_dataset = args.output / "musubi_dataset"
    dataset_dir, num_images = prepare_musubi_dataset(args.dataset, prepared_dataset, args.concept)

    # Run caching steps (unless skipped)
    if not args.skip_cache:
        if not run_latent_caching(musubi_dir, dataset_dir, args.model):
            sys.exit(1)

        if not run_text_encoder_caching(musubi_dir, dataset_dir, args.model):
            sys.exit(1)
    else:
        print("[Info] Skipping caching (--skip-cache enabled)")

    # Run training
    if not run_lora_training(
        musubi_dir,
        dataset_dir,
        args.output,
        args.model,
        num_images,
        args.epochs,
        args.lora_rank,
        args.lora_alpha,
        args.learning_rate,
        args.batch_size
    ):
        sys.exit(1)

    # Verify output exists
    lora_file = args.output / "lora.safetensors"
    if not lora_file.exists():
        print(f"[ERROR] Expected output file not found: {lora_file}")
        sys.exit(1)

    # Create a more descriptive filename
    final_name = f"{args.concept}_epoch{args.epochs}.safetensors"
    final_path = args.output / final_name
    shutil.copy(lora_file, final_path)

    print(f"\n{'='*60}")
    print(f"Training Complete!")
    print(f"{'='*60}")
    print(f"LoRA file: {final_path}")
    print(f"\nTo use in ComfyUI:")
    print(f"1. Copy to ComfyUI loras directory:")
    print(f"   cp {final_path} /path/to/ComfyUI/models/loras/")
    print(f"2. Use 'LoraLoaderModelOnly' node in your workflow")
    print(f"3. Load: {final_name}")
    print(f"4. Recommended strength: 0.8")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()
