#!/usr/bin/env python3
"""
HunyuanVideo 1.5 LoRA Training with Musubi Tuner
=================================================
Trains ComfyUI-compatible LoRA weights using Kohya-ss Musubi Tuner.

Based on official Musubi Tuner documentation:
https://github.com/kohya-ss/musubi-tuner/blob/main/docs/hunyuan_video_1_5.md
"""

import argparse
import subprocess
import os
from pathlib import Path
import shutil
import sys

def download_model_files(model_dir: Path):
    """
    Download required model files for HunyuanVideo 1.5 training.

    Required files:
    - DiT: transformer/720p_t2v/diffusion_pytorch_model.safetensors
    - VAE: vae/diffusion_pytorch_model.safetensors
    - Text Encoder: text_encoders/qwen_2.5_vl_7b.safetensors
    - BYT5: text_encoders/byt5_small_glyphxl_fp16.safetensors
    """
    model_dir.mkdir(parents=True, exist_ok=True)

    # Use HuggingFace CLI to download ComfyUI-repackaged model
    repo_id = "Comfy-Org/HunyuanVideo_repackaged"

    print(f"[Model] Downloading HunyuanVideo 1.5 models from {repo_id}...")
    print(f"[Model] This will download ~33GB, please be patient...")

    cmd = [
        "huggingface-cli", "download",
        repo_id,
        "--local-dir", str(model_dir),
        "--local-dir-use-symlinks", "False"
    ]

    try:
        subprocess.run(cmd, check=True)
        print(f"[Model] ✅ Model files downloaded to {model_dir}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Failed to download models: {e}")
        return False

def create_dataset_config(dataset_dir: Path, output_dir: Path, resolution: list):
    """
    Create TOML dataset configuration for Musubi Tuner.

    Format:
    [general]
    resolution = [width, height]
    caption_extension = ".txt"
    batch_size = 1
    enable_bucket = true

    [[datasets]]
    image_directory = "/path/to/images"
    cache_directory = "/path/to/cache"
    """
    config_file = output_dir / "dataset_config.toml"
    cache_dir = output_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # For unconditional training, create empty .txt files
    image_extensions = {".jpg", ".jpeg", ".png", ".webp"}
    num_images = 0

    for img_path in dataset_dir.rglob("*"):
        if img_path.suffix.lower() in image_extensions:
            txt_file = img_path.with_suffix(".txt")
            if not txt_file.exists():
                txt_file.write_text("")  # Empty caption for unconditional
            num_images += 1

    config_content = f"""[general]
resolution = {resolution}
caption_extension = ".txt"
batch_size = 1
enable_bucket = true
bucket_no_upscale = false

[[datasets]]
image_directory = "{dataset_dir.absolute()}"
cache_directory = "{cache_dir.absolute()}"
num_repeats = 1
"""

    config_file.write_text(config_content)
    print(f"[Dataset] Created config: {config_file}")
    print(f"[Dataset] Found {num_images} images")

    return config_file, num_images

def run_latent_caching(musubi_dir: Path, config_file: Path, model_dir: Path):
    """Cache latents using HunyuanVideo 1.5 VAE."""
    print("\n[Musubi] Step 1/3: Caching latents (VAE encoding)...")

    vae_path = model_dir / "vae" / "diffusion_pytorch_model.safetensors"
    if not vae_path.exists():
        vae_path = model_dir / "vae"  # Try directory path

    script = musubi_dir / "src" / "musubi_tuner" / "hv_1_5_cache_latents.py"

    cmd = [
        "python", str(script),
        "--dataset_config", str(config_file),
        "--vae", str(vae_path),
        "--vae_sample_size", "128",
        "--vae_tiling"
    ]

    try:
        subprocess.run(cmd, check=True, cwd=str(musubi_dir))
        print("[Musubi] ✅ Latent caching complete")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Latent caching failed: {e}")
        return False

def run_text_encoder_caching(musubi_dir: Path, config_file: Path, model_dir: Path):
    """Cache text encoder outputs."""
    print("\n[Musubi] Step 2/3: Caching text encoder outputs...")

    text_encoder_path = model_dir / "split_files" / "text_encoders" / "qwen_2.5_vl_7b.safetensors"
    byt5_path = model_dir / "split_files" / "text_encoders" / "byt5_small_glyphxl_fp16.safetensors"

    script = musubi_dir / "src" / "musubi_tuner" / "hv_1_5_cache_text_encoder_outputs.py"

    cmd = [
        "python", str(script),
        "--dataset_config", str(config_file),
        "--text_encoder", str(text_encoder_path),
        "--byt5", str(byt5_path),
        "--batch_size", "16",
        "--fp8_vl"
    ]

    try:
        subprocess.run(cmd, check=True, cwd=str(musubi_dir))
        print("[Musubi] ✅ Text encoder caching complete")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Text encoder caching failed: {e}")
        return False

def run_lora_training(
    musubi_dir: Path,
    config_file: Path,
    model_dir: Path,
    output_dir: Path,
    num_images: int,
    epochs: int,
    lora_rank: int,
    lora_alpha: int,
    learning_rate: float,
    batch_size: int
):
    """Run LoRA training with HunyuanVideo 1.5."""
    print("\n[Musubi] Step 3/3: Training LoRA adapter...")

    dit_path = model_dir / "transformer" / "720p_t2v" / "diffusion_pytorch_model.safetensors"
    vae_path = model_dir / "vae" / "diffusion_pytorch_model.safetensors"
    text_encoder_path = model_dir / "split_files" / "text_encoders" / "qwen_2.5_vl_7b.safetensors"
    byt5_path = model_dir / "split_files" / "text_encoders" / "byt5_small_glyphxl_fp16.safetensors"

    script = musubi_dir / "src" / "musubi_tuner" / "hv_1_5_train_network.py"

    steps_per_epoch = num_images // batch_size
    save_every_n_epochs = max(1, epochs // 4)

    cmd = [
        "accelerate", "launch",
        "--num_cpu_threads_per_process", "1",
        "--mixed_precision", "bf16",
        str(script),
        "--dit", str(dit_path),
        "--vae", str(vae_path),
        "--text_encoder", str(text_encoder_path),
        "--byt5", str(byt5_path),
        "--dataset_config", str(config_file),
        "--task", "t2v",
        "--sdpa",
        "--mixed_precision", "bf16",
        "--timestep_sampling", "shift",
        "--weighting_scheme", "none",
        "--discrete_flow_shift", "2.0",
        "--optimizer_type", "adamw8bit",
        f"--learning_rate={learning_rate}",
        "--gradient_checkpointing",
        "--network_module", "networks.lora_hv_1_5",
        f"--network_dim={lora_rank}",
        f"--network_alpha={lora_alpha}",
        f"--max_train_epochs={epochs}",
        f"--save_every_n_epochs={save_every_n_epochs}",
        "--output_dir", str(output_dir),
        "--output_name", "lora"
    ]

    try:
        subprocess.run(cmd, check=True, cwd=str(musubi_dir))
        print(f"[Musubi] ✅ LoRA training complete!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] LoRA training failed: {e}")
        return False

def convert_to_comfyui(musubi_dir: Path, lora_path: Path, output_path: Path):
    """Convert Musubi LoRA to ComfyUI format."""
    print("\n[Converting] Converting LoRA to ComfyUI format...")

    convert_script = musubi_dir / "src" / "musubi_tuner" / "networks" / "convert_hunyuan_video_1_5_lora_to_comfy.py"

    cmd = [
        "python", str(convert_script),
        str(lora_path),
        str(output_path)
    ]

    try:
        subprocess.run(cmd, check=True, cwd=str(musubi_dir))
        print(f"[Converting] ✅ ComfyUI LoRA saved to {output_path}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Conversion failed: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="HunyuanVideo 1.5 LoRA Training with Musubi Tuner")

    parser.add_argument("--dataset", "-d", type=Path, required=True, help="Dataset directory")
    parser.add_argument("--concept", "-c", type=str, required=True, help="Concept name")
    parser.add_argument("--output", "-o", type=Path, default=None, help="Output directory")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=1, help="Training batch size")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--lora-rank", type=int, default=32, help="LoRA rank (network_dim)")
    parser.add_argument("--lora-alpha", type=int, default=32, help="LoRA alpha (network_alpha)")
    parser.add_argument("--resolution", type=int, nargs=2, default=[960, 544], help="Resolution [width height]")
    parser.add_argument("--skip-cache", action="store_true", help="Skip caching if already done")
    parser.add_argument("--skip-download", action="store_true", help="Skip model download if already done")

    args = parser.parse_args()

    if args.output is None:
        args.output = Path(f"./outputs/lora/{args.concept}")

    args.output.mkdir(parents=True, exist_ok=True)

    print(f"{'='*60}")
    print(f"HunyuanVideo 1.5 LoRA Training with Musubi Tuner")
    print(f"{'='*60}")
    print(f"Concept: {args.concept}")
    print(f"Dataset: {args.dataset}")
    print(f"Output: {args.output}")
    print(f"Epochs: {args.epochs}")
    print(f"LoRA rank: {args.lora_rank}")
    print(f"Resolution: {args.resolution}")
    print(f"{'='*60}\n")

    # Check Musubi Tuner installation
    musubi_dir = Path("/opt/musubi-tuner")
    if not musubi_dir.exists():
        print(f"[ERROR] Musubi Tuner not found at {musubi_dir}")
        print(f"[ERROR] Please rebuild Docker image: docker compose build train")
        sys.exit(1)

    # Download models if needed
    model_dir = Path("/workspace/models/hunyuanvideo_1_5")
    if not args.skip_download:
        if not download_model_files(model_dir):
            sys.exit(1)
    else:
        print(f"[Info] Skipping model download (--skip-download enabled)")

    # Create dataset config
    print("[Dataset] Creating TOML configuration...")
    config_file, num_images = create_dataset_config(args.dataset, args.output, args.resolution)

    # Run caching steps
    if not args.skip_cache:
        if not run_latent_caching(musubi_dir, config_file, model_dir):
            sys.exit(1)

        if not run_text_encoder_caching(musubi_dir, config_file, model_dir):
            sys.exit(1)
    else:
        print("[Info] Skipping caching (--skip-cache enabled)")

    # Run training
    if not run_lora_training(
        musubi_dir,
        config_file,
        model_dir,
        args.output,
        num_images,
        args.epochs,
        args.lora_rank,
        args.lora_alpha,
        args.learning_rate,
        args.batch_size
    ):
        sys.exit(1)

    # Find the last epoch LoRA file
    lora_pattern = f"lora-{args.epochs:06d}.safetensors"
    lora_path = args.output / lora_pattern

    if not lora_path.exists():
        # Try without epoch number
        lora_path = args.output / "lora.safetensors"

    if not lora_path.exists():
        print(f"[ERROR] Could not find trained LoRA file in {args.output}")
        sys.exit(1)

    # Convert to ComfyUI format
    comfy_lora_path = args.output / f"{args.concept}_epoch{args.epochs}.safetensors"
    if not convert_to_comfyui(musubi_dir, lora_path, comfy_lora_path):
        print("[Warning] Conversion failed, but training completed")
        print(f"[Warning] You may need to convert manually")
        comfy_lora_path = lora_path

    print(f"\n{'='*60}")
    print(f"Training Complete!")
    print(f"{'='*60}")
    print(f"LoRA file: {comfy_lora_path}")
    print(f"\nTo use in ComfyUI:")
    print(f"1. Copy to ComfyUI loras directory:")
    print(f"   cp {comfy_lora_path} /path/to/ComfyUI/models/loras/")
    print(f"2. Use 'LoraLoaderModelOnly' node in your workflow")
    print(f"3. Load: {comfy_lora_path.name}")
    print(f"4. Recommended strength: 0.8")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()
