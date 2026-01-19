#!/usr/bin/env python3
"""
Simplified HunyuanVideo LoRA Training
=====================================
One-command LoRA training that handles everything automatically.

Usage:
    python train_lora_simple.py --dataset data/scraped/fallout_nv_20260116_113625

First run downloads models (~33GB, one-time, cached).
Subsequent runs skip directly to training.
"""

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Optional

# Paths
WORKSPACE = Path("/workspace")
MODELS_DIR = WORKSPACE / "models" / "hunyuanvideo_1_5"
MUSUBI_DIR = Path("/opt/musubi-tuner")


def print_section(title: str):
    """Print a section header."""
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}\n")


def check_models_downloaded() -> bool:
    """Check if required model files exist."""
    required_files = [
        MODELS_DIR / "vae" / "diffusion_pytorch_model.safetensors",
        MODELS_DIR / "transformer" / "720p_t2v" / "diffusion_pytorch_model.safetensors",
        MODELS_DIR / "text_encoders" / "qwen_2.5_vl_7b.safetensors",
        MODELS_DIR / "text_encoders" / "byt5_small_glyphxl_fp16.safetensors",
    ]

    return all(f.exists() for f in required_files)


def download_models():
    """Download HunyuanVideo models (one-time, ~33GB)."""
    print_section("FIRST-TIME SETUP: Downloading Models")

    print("This downloads ~33GB of models (one-time only):")
    print("  • DiT Transformer (720p T2V)")
    print("  • VAE (Variational Autoencoder)")
    print("  • Qwen 2.5 VL Text Encoder")
    print("  • BYT5 Tokenizer")
    print()
    print("These files will be cached. Future training runs skip this step.")
    print()

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Download main model (DiT + VAE) from Tencent
    print("Step 1/2: Downloading DiT & VAE from tencent/HunyuanVideo-1.5...")
    print("(Downloading full repo - this ensures all files are retrieved correctly)")
    cmd = [
        "huggingface-cli", "download",
        "tencent/HunyuanVideo-1.5",
        "--local-dir", str(MODELS_DIR),
        "--local-dir-use-symlinks", "False"
    ]

    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("✅ DiT & VAE downloaded")
    except subprocess.CalledProcessError as e:
        print(f"❌ Download failed: {e}")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
        print("\nTroubleshooting:")
        print("1. Check internet connection")
        print("2. Verify HuggingFace access (may need token)")
        print("3. Ensure sufficient disk space (~50GB)")
        print(f"\nManual download: https://huggingface.co/tencent/HunyuanVideo-1.5")
        return False

    # Download text encoders from ComfyUI repackaged (they're in FP16 but compatible)
    print("\nStep 2/2: Downloading text encoders from Comfy-Org...")
    text_encoder_dir = MODELS_DIR / "text_encoders"
    text_encoder_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "huggingface-cli", "download",
        "Comfy-Org/HunyuanVideo_1.5_repackaged",
        "split_files/text_encoders/qwen_2.5_vl_7b.safetensors",
        "split_files/text_encoders/byt5_small_glyphxl_fp16.safetensors",
        "--local-dir", str(MODELS_DIR / "temp_download"),
        "--local-dir-use-symlinks", "False"
    ]

    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        # Move files to correct location
        temp_dir = MODELS_DIR / "temp_download" / "split_files" / "text_encoders"
        import shutil
        if (temp_dir / "qwen_2.5_vl_7b.safetensors").exists():
            shutil.move(
                str(temp_dir / "qwen_2.5_vl_7b.safetensors"),
                str(text_encoder_dir / "qwen_2.5_vl_7b.safetensors")
            )
        if (temp_dir / "byt5_small_glyphxl_fp16.safetensors").exists():
            shutil.move(
                str(temp_dir / "byt5_small_glyphxl_fp16.safetensors"),
                str(text_encoder_dir / "byt5_small_glyphxl_fp16.safetensors")
            )
        # Clean up temp directory
        shutil.rmtree(str(MODELS_DIR / "temp_download"), ignore_errors=True)
        print("✅ Text encoders downloaded")
    except subprocess.CalledProcessError as e:
        print(f"❌ Text encoder download failed: {e}")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
        return False

    # Verify all files exist
    print("\n=== Verifying Downloaded Files ===")
    required_files = [
        MODELS_DIR / "vae" / "diffusion_pytorch_model.safetensors",
        MODELS_DIR / "transformer" / "720p_t2v" / "diffusion_pytorch_model.safetensors",
        MODELS_DIR / "text_encoders" / "qwen_2.5_vl_7b.safetensors",
        MODELS_DIR / "text_encoders" / "byt5_small_glyphxl_fp16.safetensors",
    ]

    all_exist = True
    for f in required_files:
        exists = f.exists()
        status = "✅" if exists else "❌"
        size = f"{f.stat().st_size / (1024**3):.2f} GB" if exists else "MISSING"
        print(f"{status} {f.name}: {size}")
        if not exists:
            all_exist = False

    if not all_exist:
        print("\n❌ Some files are missing after download!")
        print("Checking what was actually downloaded...")
        print(f"\nContents of {MODELS_DIR}:")
        for item in MODELS_DIR.rglob("*"):
            if item.is_file():
                rel_path = item.relative_to(MODELS_DIR)
                size_mb = item.stat().st_size / (1024**2)
                print(f"  {rel_path} ({size_mb:.1f} MB)")
        return False

    print("\n✅ All models downloaded and verified successfully!")
    return True


def prepare_dataset(dataset_dir: Path, output_dir: Path, resolution: list) -> tuple[Path, int]:
    """Create dataset config and caption files."""
    print_section("Preparing Dataset")

    cache_dir = output_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Create empty captions for unconditional training
    image_extensions = {".jpg", ".jpeg", ".png", ".webp"}
    num_images = 0

    for img_path in dataset_dir.rglob("*"):
        if img_path.suffix.lower() in image_extensions:
            txt_file = img_path.with_suffix(".txt")
            if not txt_file.exists():
                txt_file.write_text("")  # Empty = unconditional
            num_images += 1

    # Create TOML config
    config_file = output_dir / "dataset_config.toml"
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

    print(f"✅ Dataset prepared")
    print(f"   Images found: {num_images}")
    print(f"   Config: {config_file.name}")

    return config_file, num_images


def cache_latents(config_file: Path, skip_if_exists: bool = True) -> bool:
    """Cache VAE latents (Phase 1/3)."""
    print_section("Phase 1/3: Caching Latents (VAE Encoding)")

    cache_dir = config_file.parent / "cache"
    if skip_if_exists and cache_dir.exists() and any(cache_dir.glob("*.npz")):
        print("✅ Latents already cached (skipping)")
        return True

    print("Encoding images with VAE...")
    print("This takes 5-10 minutes (one-time per dataset)")

    vae_file = MODELS_DIR / "vae" / "diffusion_pytorch_model.safetensors"
    script = MUSUBI_DIR / "src" / "musubi_tuner" / "hv_1_5_cache_latents.py"

    cmd = [
        "python", str(script),
        "--dataset_config", str(config_file.absolute()),
        "--vae", str(vae_file.absolute()),
        "--vae_sample_size", "128"
    ]

    try:
        subprocess.run(cmd, check=True)
        print("✅ Latents cached")
        return True
    except subprocess.CalledProcessError:
        print("❌ Latent caching failed")
        return False


def cache_text_encoders(config_file: Path, skip_if_exists: bool = True) -> bool:
    """Cache text encoder outputs (Phase 2/3)."""
    print_section("Phase 2/3: Caching Text Encoders")

    cache_dir = config_file.parent / "cache"
    if skip_if_exists and cache_dir.exists() and any(cache_dir.glob("*.npz")):
        # Check if text encoder cache exists (different pattern than latents)
        if any(f.name.startswith("text") for f in cache_dir.glob("*.npz")):
            print("✅ Text encoders already cached (skipping)")
            return True

    print("Processing text prompts...")
    print("This takes 2-5 minutes (one-time per dataset)")

    text_encoder = MODELS_DIR / "text_encoders" / "qwen_2.5_vl_7b.safetensors"
    byt5 = MODELS_DIR / "text_encoders" / "byt5_small_glyphxl_fp16.safetensors"
    script = MUSUBI_DIR / "src" / "musubi_tuner" / "hv_1_5_cache_text_encoder_outputs.py"

    cmd = [
        "python", str(script),
        "--dataset_config", str(config_file.absolute()),
        "--text_encoder", str(text_encoder.absolute()),
        "--byt5", str(byt5.absolute()),
        "--batch_size", "16",
        "--fp8_vl"
    ]

    try:
        subprocess.run(cmd, check=True)
        print("✅ Text encoders cached")
        return True
    except subprocess.CalledProcessError:
        print("❌ Text encoder caching failed")
        return False


def train_lora(
    config_file: Path,
    output_dir: Path,
    concept: str,
    num_images: int,
    epochs: int,
    lora_rank: int,
    lora_alpha: int,
    learning_rate: float
) -> Optional[Path]:
    """Train LoRA adapter (Phase 3/3)."""
    print_section("Phase 3/3: Training LoRA")

    dit = MODELS_DIR / "transformer" / "720p_t2v" / "diffusion_pytorch_model.safetensors"
    vae = MODELS_DIR / "vae" / "diffusion_pytorch_model.safetensors"
    text_encoder = MODELS_DIR / "text_encoders" / "qwen_2.5_vl_7b.safetensors"
    byt5 = MODELS_DIR / "text_encoders" / "byt5_small_glyphxl_fp16.safetensors"
    script = MUSUBI_DIR / "src" / "musubi_tuner" / "hv_1_5_train_network.py"

    total_steps = num_images * epochs
    save_every = max(1, epochs // 4)

    print(f"Training configuration:")
    print(f"  • Images: {num_images}")
    print(f"  • Epochs: {epochs}")
    print(f"  • Total steps: {total_steps}")
    print(f"  • LoRA rank: {lora_rank}")
    print(f"  • Learning rate: {learning_rate}")
    print(f"  • Checkpoints every: {save_every} epochs")
    print()
    print("This will take 5-8 hours on RTX 5090...")
    print("You can safely close this terminal - training continues in container.")
    print()

    cmd = [
        "accelerate", "launch",
        "--num_cpu_threads_per_process", "1",
        "--mixed_precision", "bf16",
        str(script),
        "--dit", str(dit.absolute()),
        "--vae", str(vae.absolute()),
        "--text_encoder", str(text_encoder.absolute()),
        "--byt5", str(byt5.absolute()),
        "--dataset_config", str(config_file.absolute()),
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
        f"--save_every_n_epochs={save_every}",
        "--output_dir", str(output_dir.absolute()),
        "--output_name", "lora"
    ]

    try:
        subprocess.run(cmd, check=True)
        print("\n✅ Training complete!")

        # Find output file
        lora_file = output_dir / f"lora-{epochs:06d}.safetensors"
        if not lora_file.exists():
            lora_file = output_dir / "lora.safetensors"

        return lora_file if lora_file.exists() else None

    except subprocess.CalledProcessError:
        print("\n❌ Training failed")
        return None


def convert_to_comfyui(lora_file: Path, output_file: Path) -> bool:
    """Convert to ComfyUI format."""
    print_section("Converting to ComfyUI Format")

    script = MUSUBI_DIR / "src" / "musubi_tuner" / "networks" / "convert_hunyuan_video_1_5_lora_to_comfy.py"

    cmd = [
        "python", str(script),
        str(lora_file.absolute()),
        str(output_file.absolute())
    ]

    try:
        subprocess.run(cmd, check=True)
        print(f"✅ ComfyUI LoRA: {output_file.name}")
        return True
    except subprocess.CalledProcessError:
        print("⚠️  Conversion failed (training file may already be compatible)")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Simplified HunyuanVideo LoRA Training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic training (20 epochs, rank 32)
  python train_lora_simple.py --dataset data/scraped/fallout_nv_20260116_113625

  # More epochs for better results
  python train_lora_simple.py --dataset data/scraped/fallout_nv_20260116_113625 --epochs 30

  # Smaller LoRA (faster training, less VRAM)
  python train_lora_simple.py --dataset data/scraped/fallout_nv_20260116_113625 --lora-rank 16
        """
    )

    parser.add_argument("--dataset", "-d", type=Path, required=True,
                       help="Path to your image dataset")
    parser.add_argument("--concept", "-c", type=str, default=None,
                       help="Concept name (default: dataset folder name)")
    parser.add_argument("--output", "-o", type=Path, default=None,
                       help="Output directory (default: outputs/lora/<concept>)")
    parser.add_argument("--epochs", type=int, default=20,
                       help="Training epochs (default: 20)")
    parser.add_argument("--lora-rank", type=int, default=32,
                       help="LoRA rank (default: 32)")
    parser.add_argument("--lora-alpha", type=int, default=32,
                       help="LoRA alpha (default: 32)")
    parser.add_argument("--learning-rate", type=float, default=1e-4,
                       help="Learning rate (default: 1e-4)")
    parser.add_argument("--resolution", type=int, nargs=2, default=[960, 544],
                       help="Resolution [width height] (default: 960 544)")
    parser.add_argument("--skip-cache", action="store_true",
                       help="Skip caching if already done")

    args = parser.parse_args()

    # Set defaults
    if args.concept is None:
        args.concept = args.dataset.name

    if args.output is None:
        args.output = WORKSPACE / "outputs" / "lora" / args.concept

    args.output.mkdir(parents=True, exist_ok=True)

    # Print header
    print("\n" + "="*70)
    print("  HunyuanVideo LoRA Training - Simplified")
    print("="*70)
    print(f"\nConcept: {args.concept}")
    print(f"Dataset: {args.dataset}")
    print(f"Epochs: {args.epochs}")
    print(f"LoRA Rank: {args.lora_rank}")
    print(f"Output: {args.output}")

    # Check Musubi Tuner
    if not MUSUBI_DIR.exists():
        print("\n❌ Musubi Tuner not found!")
        print("   Run: docker compose build train")
        sys.exit(1)

    # Check/download models
    if not check_models_downloaded():
        print("\n⚠️  Models not found - first-time setup required")
        if not download_models():
            sys.exit(1)
    else:
        print("\n✅ Models already downloaded (cached)")

    # Prepare dataset
    config_file, num_images = prepare_dataset(args.dataset, args.output, args.resolution)

    if num_images == 0:
        print("\n❌ No images found in dataset!")
        sys.exit(1)

    # Cache latents and text encoders
    if not cache_latents(config_file, skip_if_exists=args.skip_cache):
        sys.exit(1)

    if not cache_text_encoders(config_file, skip_if_exists=args.skip_cache):
        sys.exit(1)

    # Train
    lora_file = train_lora(
        config_file,
        args.output,
        args.concept,
        num_images,
        args.epochs,
        args.lora_rank,
        args.lora_alpha,
        args.learning_rate
    )

    if not lora_file:
        print("\n❌ Training failed")
        sys.exit(1)

    # Convert to ComfyUI
    comfy_file = args.output / f"{args.concept}_epoch{args.epochs}.safetensors"
    if not convert_to_comfyui(lora_file, comfy_file):
        comfy_file = lora_file

    # Success message
    print_section("SUCCESS! LoRA Training Complete")
    print(f"✅ LoRA file: {comfy_file}")
    print(f"\nTo use in ComfyUI:")
    print(f"1. Copy to ComfyUI:")
    print(f"   cp {comfy_file} /path/to/ComfyUI/models/loras/")
    print(f"\n2. Load in workflow:")
    print(f"   Node: LoraLoaderModelOnly")
    print(f"   File: {comfy_file.name}")
    print(f"   Strength: 0.8 (recommended)")
    print()


if __name__ == "__main__":
    main()
