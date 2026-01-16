#!/usr/bin/env python3
"""
HunyuanVideo Official LoRA Training Wrapper
============================================
Uses Tencent's official HunyuanVideo-1.5 training code for LoRA training.

This wrapper adapts the official training script to work with our dataset format.
"""

import argparse
import sys
import subprocess
from pathlib import Path
import json

def prepare_dataset_for_official_training(dataset_dir: Path, output_dir: Path):
    """
    Convert our dataset format to HunyuanVideo's expected format.

    HunyuanVideo expects:
    - data/
      - videos/ or images/
      - prompts.json (optional, for text-conditioned training)
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir = output_dir / "images"
    images_dir.mkdir(exist_ok=True)

    # Copy/link images
    image_extensions = {".jpg", ".jpeg", ".png", ".webp"}
    images = []

    for img_path in dataset_dir.rglob("*"):
        if img_path.suffix.lower() in image_extensions:
            # Create symlink or copy
            dest = images_dir / img_path.name
            if not dest.exists():
                dest.symlink_to(img_path.absolute())
            images.append(img_path.name)

    print(f"[Dataset] Prepared {len(images)} images for training")

    # Create prompts.json (empty for unconditional training)
    prompts = {img: "" for img in images}
    prompts_file = output_dir / "prompts.json"
    prompts_file.write_text(json.dumps(prompts, indent=2))

    return output_dir

def main():
    parser = argparse.ArgumentParser(description="HunyuanVideo Official LoRA Training")

    parser.add_argument("--dataset", "-d", type=Path, required=True, help="Dataset directory")
    parser.add_argument("--concept", "-c", type=str, required=True, help="Concept name")
    parser.add_argument("--model", type=str, required=True, help="Model path or HuggingFace repo")
    parser.add_argument("--output", "-o", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--resolution", type=int, default=512, help="Training resolution")

    args = parser.parse_args()

    if args.output is None:
        args.output = Path(f"./outputs/lora/{args.concept}")

    args.output.mkdir(parents=True, exist_ok=True)

    print(f"{'='*60}")
    print(f"HunyuanVideo Official LoRA Training")
    print(f"{'='*60}")
    print(f"Concept: {args.concept}")
    print(f"Dataset: {args.dataset}")
    print(f"Model: {args.model}")
    print(f"Output: {args.output}")
    print(f"Epochs: {args.epochs}")
    print(f"LoRA rank: {args.lora_rank}")
    print(f"{'='*60}\n")

    # Check if official repo exists (cloned to /opt to avoid volume mount override)
    official_repo = Path("/opt/HunyuanVideo-1.5")
    if not official_repo.exists():
        print(f"[ERROR] Official HunyuanVideo-1.5 repo not found at {official_repo}")
        print(f"[ERROR] Please rebuild Docker image: docker compose build train")
        sys.exit(1)

    # Check if they have training scripts
    train_script = official_repo / "train_lora.py"
    if not train_script.exists():
        # Try to find the actual training script
        possible_scripts = list(official_repo.rglob("*train*.py"))
        if possible_scripts:
            print(f"[Info] Found training scripts:")
            for script in possible_scripts[:5]:
                print(f"  - {script.relative_to(official_repo)}")
            train_script = possible_scripts[0]
        else:
            print(f"[ERROR] No training script found in {official_repo}")
            print(f"[Info] Listing repo contents:")
            subprocess.run(["ls", "-la", str(official_repo)])
            sys.exit(1)

    # Prepare dataset
    print(f"[Dataset] Preparing dataset for official training format...")
    prepared_dataset = args.output / "prepared_dataset"
    prepare_dataset_for_official_training(args.dataset, prepared_dataset)

    # Build command for official training script
    # Map our arguments to HunyuanVideo-1.5's expected format
    # Calculate max_steps from epochs and dataset size
    steps_per_epoch = len(list(prepared_dataset.glob("images/*"))) // args.batch_size
    max_steps = steps_per_epoch * args.epochs

    cmd = [
        "python", str(train_script),
        "--pretrained_model_root", args.model,
        "--pretrained_transformer_version", "720p_t2v",
        "--output_dir", str(args.output),
        "--use_lora",
        "--lora_r", str(args.lora_rank),
        "--lora_alpha", str(args.lora_alpha),
        "--learning_rate", str(args.learning_rate),
        "--max_steps", str(max_steps),
        "--batch_size", str(args.batch_size),
        "--dtype", "bf16",
        "--save_interval", "500",
        "--log_interval", "10",
        "--sp_size", "1",  # Single GPU (no sequence parallelism)
        "--enable_gradient_checkpointing",  # Memory efficiency
    ]

    print(f"[Training] Estimated steps: {max_steps} ({steps_per_epoch} steps/epoch × {args.epochs} epochs)")

    print(f"\n[Training] Running official training script...")
    print(f"[Training] Command: {' '.join(cmd)}\n")

    # Run training
    try:
        result = subprocess.run(cmd, check=False, cwd=str(official_repo))
        if result.returncode != 0:
            print(f"\n[ERROR] Training script failed with exit code {result.returncode}")
            print(f"\n[Info] The official script may have different arguments.")
            print(f"[Info] Check {official_repo}/README.md for correct usage.")
            print(f"\n[Info] You can also run training manually:")
            print(f"  docker compose --profile pipeline run --rm train bash")
            print(f"  cd /opt/HunyuanVideo-1.5")
            print(f"  python train_lora.py --help")
            sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] Failed to run training: {e}")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"Training Complete!")
    print(f"{'='*60}")
    print(f"Output: {args.output}")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()
