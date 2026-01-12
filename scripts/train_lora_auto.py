#!/usr/bin/env python3
"""
Automated LoRA Training Pipeline
Organizes scraped data and trains LoRA without manual intervention
"""
import os
import shutil
import argparse
import json
from pathlib import Path
from datetime import datetime


def organize_training_data(source_dir: Path, concept_name: str, repeats: int = 10) -> Path:
    """
    Automatically organize scraped images into Kohya format

    Args:
        source_dir: Directory with scraped images
        concept_name: Name for this concept (e.g., 'fallout', 'newvegas')
        repeats: Number of times to repeat each image per epoch

    Returns:
        Path to organized training directory
    """
    # Create training directory with Kohya naming convention
    training_root = Path("./data/training") / datetime.now().strftime("%Y%m%d_%H%M%S")
    kohya_folder = training_root / f"{repeats}_{concept_name}"
    kohya_folder.mkdir(parents=True, exist_ok=True)

    # Copy all images to the Kohya folder
    image_extensions = {'.jpg', '.jpeg', '.png', '.webp'}
    copied = 0

    source_path = Path(source_dir)
    if not source_path.exists():
        raise FileNotFoundError(f"Source directory does not exist: {source_dir}")

    for img_file in source_path.rglob('*'):
        if img_file.suffix.lower() in image_extensions:
            dest = kohya_folder / img_file.name
            shutil.copy2(img_file, dest)
            copied += 1

    if copied == 0:
        raise ValueError(f"No images found in {source_dir}. Supported formats: {image_extensions}")

    print(f"✓ Organized {copied} images into {kohya_folder}")
    return training_root


def generate_training_config(
    training_dir: Path,
    model_path: str,
    output_name: str,
    epochs: int = 10,
    learning_rate: float = 1e-4,
    batch_size: int = 1,
    max_train_steps: int = None
) -> Path:
    """
    Generate TOML config file for Kohya training

    Args:
        training_dir: Directory with {repeats}_{concept} folders
        model_path: Path to base model
        output_name: Name for output LoRA
        epochs: Number of training epochs
        learning_rate: Learning rate
        batch_size: Batch size
        max_train_steps: Maximum training steps (overrides epochs if set)

    Returns:
        Path to generated config file
    """
    output_dir = Path("./outputs/lora") / output_name
    output_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "model_arguments": {
            "pretrained_model_name_or_path": model_path,
        },
        "dataset_arguments": {
            "train_data_dir": str(training_dir.absolute()),
            "resolution": "512,512",
            "batch_size": batch_size,
            "enable_bucket": False,
        },
        "training_arguments": {
            "output_dir": str(output_dir.absolute()),
            "output_name": output_name,
            "save_precision": "fp16",
            "mixed_precision": "fp16",
            "learning_rate": learning_rate,
            "max_train_epochs": epochs,
            "save_every_n_epochs": 1,
            "optimizer_type": "AdamW8bit",
            "lr_scheduler": "cosine",
            "lr_warmup_steps": 0,
            "network_module": "networks.lora",
            "network_dim": 8,
            "network_alpha": 4,
        },
        "logging_arguments": {
            "log_with": "tensorboard",
            "logging_dir": "./logs",
        }
    }

    if max_train_steps:
        config["training_arguments"]["max_train_steps"] = max_train_steps
        del config["training_arguments"]["max_train_epochs"]

    config_path = output_dir / "config.toml"

    # Write TOML manually (simple key-value format)
    with open(config_path, 'w') as f:
        for section, values in config.items():
            f.write(f"[{section}]\n")
            for key, value in values.items():
                if isinstance(value, str):
                    f.write(f'{key} = "{value}"\n')
                elif isinstance(value, bool):
                    f.write(f'{key} = {str(value).lower()}\n')
                else:
                    f.write(f'{key} = {value}\n')
            f.write('\n')

    print(f"✓ Generated config at {config_path}")
    return config_path


def run_training(config_path: Path):
    """
    Execute Kohya training via CLI

    Args:
        config_path: Path to TOML config file
    """
    # Convert to relative path if it's absolute, otherwise use as-is
    try:
        rel_path = config_path.resolve().relative_to(Path.cwd().resolve())
    except ValueError:
        # If relative_to fails, assume it's already relative
        rel_path = config_path

    # Convert Windows backslashes to forward slashes for container path
    container_path = str(rel_path).replace('\\', '/')

    cmd = f"""
docker compose --profile kohya exec kohya \\
    /venv/bin/accelerate launch \\
    --dynamo_backend no \\
    --mixed_precision fp16 \\
    --num_processes 1 \\
    /app/sd-scripts/train_network.py \\
    --config_file /app/{container_path}
"""

    print(f"\n🚀 Starting training with command:\n{cmd}\n")
    os.system(cmd)


def main():
    parser = argparse.ArgumentParser(description="Automated LoRA training from scraped data")
    parser.add_argument("--source", required=True, help="Directory with scraped images")
    parser.add_argument("--concept", required=True, help="Concept name (e.g., 'fallout')")
    parser.add_argument("--model", default="/app/models/unet/wan2.2_t2v_high_noise_14B_fp8_scaled.safetensors",
                        help="Base model path")
    parser.add_argument("--output", help="Output LoRA name (defaults to concept name)")
    parser.add_argument("--repeats", type=int, default=10, help="Image repeats per epoch")
    parser.add_argument("--epochs", type=int, default=10, help="Training epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size")
    parser.add_argument("--max-steps", type=int, help="Max training steps (overrides epochs)")

    args = parser.parse_args()

    output_name = args.output or args.concept

    print("=" * 60)
    print("🤖 AUTOMATED LORA TRAINING PIPELINE")
    print("=" * 60)
    print(f"Source: {args.source}")
    print(f"Concept: {args.concept}")
    print(f"Output: {output_name}")
    print("=" * 60)

    # Step 1: Organize data
    print("\n[1/3] Organizing training data...")
    training_dir = organize_training_data(Path(args.source), args.concept, args.repeats)

    # Step 2: Generate config
    print("\n[2/3] Generating training config...")
    config_path = generate_training_config(
        training_dir,
        args.model,
        output_name,
        args.epochs,
        args.lr,
        args.batch_size,
        args.max_steps
    )

    # Step 3: Run training
    print("\n[3/3] Starting training...")
    run_training(config_path)

    print("\n✅ Training pipeline initiated!")
    print(f"Output will be saved to: ./outputs/lora/{output_name}/")


if __name__ == "__main__":
    main()
