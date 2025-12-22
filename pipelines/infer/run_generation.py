#!/usr/bin/env python3
"""
Flux ComfyUI Image Generation Pipeline
======================================
Phase 1: CPU-only implementation with symbolic model for testing
         AIM logging and mocked B2 storage integration

This script demonstrates the complete MLOps pipeline structure:
1. Initialize experiment tracking (AIM)
2. Load/prepare data from storage (mocked B2)
3. Run inference with a minimal model
4. Log results and artifacts to AIM
5. Upload outputs to storage

PHASE 2/3 TODO:
- Replace symbolic model with real Flux/ComfyUI pipeline
- Enable GPU acceleration via SkyPilot
- Integrate Prefect for workflow orchestration
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any

import numpy as np
from PIL import Image
import torch

# No-op decorators (Prefect removed)
def flow(*args, **kwargs):
    def decorator(fn):
        return fn
    return decorator if not args or callable(args[0]) else decorator

def task(*args, **kwargs):
    def decorator(fn):
        return fn
    return decorator if not args or callable(args[0]) else decorator

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config import get_config
from utils.storage import get_s3_client

# AIM for experiment tracking
try:
    from aim import Run, Image as AimImage
    AIM_AVAILABLE = True
except ImportError:
    AIM_AVAILABLE = False
    print("[Warning] AIM not installed, metrics logging disabled")


class SymbolicImageGenerator:
    """
    Minimal symbolic image generator for Phase 1 testing.

    This generates simple procedural images to validate the pipeline
    without requiring actual diffusion model weights.

    PHASE 2/3 TODO: Replace with actual Flux/ComfyUI integration:
    - from diffusers import FluxPipeline
    - pipeline = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev")
    """

    def __init__(self, device: str = "cpu"):
        self.device = device
        print(f"[SymbolicGenerator] Initialized on device: {device}")

        # PHASE 2/3 TODO: Load real model
        # self.model = FluxPipeline.from_pretrained(
        #     "black-forest-labs/FLUX.1-dev",
        #     torch_dtype=torch.float16 if device == "cuda" else torch.float32
        # ).to(device)

    def generate(
        self,
        prompt: str,
        width: int = 256,
        height: int = 256,
        seed: Optional[int] = None,
        num_inference_steps: int = 4  # Symbolic, ignored in Phase 1
    ) -> np.ndarray:
        """
        Generate a symbolic image based on the prompt.

        Phase 1: Creates procedural patterns based on prompt hash
        Phase 2/3: Will use actual diffusion model inference
        """
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)

        # Create symbolic image based on prompt characteristics
        prompt_hash = hash(prompt) % 1000

        # Generate a gradient + noise pattern
        x = np.linspace(0, 1, width)
        y = np.linspace(0, 1, height)
        xx, yy = np.meshgrid(x, y)

        # Create RGB channels with different patterns
        r = np.sin(xx * 10 + prompt_hash * 0.01) * 0.5 + 0.5
        g = np.cos(yy * 10 + prompt_hash * 0.02) * 0.5 + 0.5
        b = np.sin((xx + yy) * 5 + prompt_hash * 0.03) * 0.5 + 0.5

        # Add some noise
        noise = np.random.randn(height, width, 3) * 0.1
        image = np.stack([r, g, b], axis=-1) + noise
        image = np.clip(image, 0, 1)

        # Convert to uint8
        image = (image * 255).astype(np.uint8)

        print(f"[SymbolicGenerator] Generated {width}x{height} image for prompt: '{prompt[:50]}...'")
        return image


@task(name="init-aim-tracking")
def init_aim(config: Dict[str, Any], run_name: Optional[str] = None) -> Optional[Run]:
    """
    Initialize AIM for experiment tracking.
    """
    if not AIM_AVAILABLE:
        return None

    app_config = get_config()

    run = Run(
        repo=str(app_config.aim.repo),
        experiment=app_config.aim.experiment,
    )

    # Set run name and metadata
    run_name = run_name or f"flux-gen-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    run.name = run_name

    # Log hyperparameters
    run["hparams"] = config

    # Add tags
    run.add_tag("flux-comfyui")
    run.add_tag("phase1")
    run.add_tag("cpu")

    print(f"[AIM] Initialized run: {run.name}")
    print(f"[AIM] Repo: {app_config.aim.repo}")

    return run


@flow(name="flux-generation-pipeline", log_prints=True)
def run_generation_pipeline(
    prompts: list[str],
    output_dir: Path,
    width: int = 256,
    height: int = 256,
    seed: Optional[int] = None
) -> list[Path]:
    """
    Run the complete image generation pipeline.

    Steps:
    1. Initialize AIM tracking
    2. Load any required data from B2 (mocked)
    3. Generate images
    4. Log to AIM and save outputs
    5. Upload results to B2 (mocked)
    """
    config = get_config()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Pipeline configuration
    pipeline_config = {
        "model": "symbolic-v1",  # PHASE 2/3: "flux-1-dev"
        "width": width,
        "height": height,
        "seed": seed,
        "device": config.compute.device,
        "num_prompts": len(prompts),
        "phase": "1-local-core"
    }

    # Step 1: Initialize AIM tracking
    run = init_aim(pipeline_config)

    # Step 2: Initialize S3-compatible client and check for any input data
    s3_client = get_s3_client(bucket="mlops-data")
    available_files = s3_client.list_objects(prefix="datasets/") if s3_client else []
    print(f"[Pipeline] Found {len(available_files)} files in storage")

    # Log data inventory
    if run:
        run.track(len(available_files), name="available_files", context={"subset": "data"})

    # Step 3: Initialize generator
    generator = SymbolicImageGenerator(device=config.compute.device)

    # Step 4: Generate images
    generated_paths = []

    for i, prompt in enumerate(prompts):
        current_seed = seed + i if seed else None

        print(f"\n[Pipeline] Generating image {i+1}/{len(prompts)}")

        # Generate image
        image_array = generator.generate(
            prompt=prompt,
            width=width,
            height=height,
            seed=current_seed
        )

        # Save locally
        image_path = output_dir / f"generated_{i:04d}.png"
        pil_image = Image.fromarray(image_array)
        pil_image.save(image_path)
        generated_paths.append(image_path)

        # Log to AIM
        if run:
            aim_image = AimImage(pil_image, caption=prompt[:100])
            run.track(aim_image, name="generated_images", step=i, context={"prompt": prompt[:50]})
            run.track(i, name="generation_step", step=i)
            if current_seed:
                run.track(current_seed, name="seed", step=i)

        print(f"[Pipeline] Saved: {image_path}")

    # Step 5: Upload results to S3 (if configured)
    if s3_client:
        print("\n[Pipeline] Uploading results to storage...")
        for path in generated_paths:
            dest_key = f"outputs/flux-comfyui/{path.name}"
            s3_client.upload_file(path, dest_key)
            print(f"[Storage] Uploaded {dest_key}")

    # Log final summary to AIM
    if run:
        run["summary"] = {
            "total_images": len(generated_paths),
            "output_dir": str(output_dir),
            "prompts": prompts,
        }
        run.close()

    print(f"\n[Pipeline] Complete! Generated {len(generated_paths)} images")
    print(f"[Pipeline] AIM run saved to: {config.aim.repo}")

    return generated_paths


def main():
    """Main entry point for the generation script."""
    parser = argparse.ArgumentParser(
        description="Flux ComfyUI Image Generation Pipeline (Phase 1: Symbolic)"
    )
    parser.add_argument(
        "--prompts",
        nargs="+",
        default=[
            "A serene mountain landscape at sunset",
            "A futuristic cityscape with flying cars",
            "An abstract pattern of geometric shapes"
        ],
        help="List of prompts to generate images for"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./outputs/flux-generations"),
        help="Directory to save generated images"
    )
    parser.add_argument(
        "--width",
        type=int,
        default=256,
        help="Image width (default: 256 for Phase 1)"
    )
    parser.add_argument(
        "--height",
        type=int,
        default=256,
        help="Image height (default: 256 for Phase 1)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("Flux ComfyUI Generation Pipeline")
    print("Phase 1: Local Core (CPU-only, Symbolic Model)")
    print("=" * 60)

    # Run the pipeline
    generated_paths = run_generation_pipeline(
        prompts=args.prompts,
        output_dir=args.output_dir,
        width=args.width,
        height=args.height,
        seed=args.seed
    )

    print("\n" + "=" * 60)
    print("Pipeline Execution Summary")
    print("=" * 60)
    print(f"Generated images: {len(generated_paths)}")
    print(f"Output directory: {args.output_dir}")
    print(f"Tracking: AIM (http://localhost:43800)")
    print(f"Storage: MinIO S3 (http://localhost:9001)")
    print("=" * 60)


if __name__ == "__main__":
    main()


# PHASE 2/3 TODO: Prefect workflow integration
# from prefect import flow, task
#
# @task(retries=3, retry_delay_seconds=60)
# def load_model_task(model_name: str, device: str):
#     """Load the Flux model with retry logic."""
#     pass
#
# @task
# def generate_batch_task(prompts: list, model, config: dict):
#     """Generate a batch of images."""
#     pass
#
# @flow(name="flux-generation-pipeline")
# def flux_generation_flow(prompts: list, config: dict):
#     """Prefect flow for orchestrated image generation."""
#     model = load_model_task(config["model"], config["device"])
#     results = generate_batch_task(prompts, model, config)
#     return results


# PHASE 2/3 TODO: SkyPilot integration for GPU provisioning
# import sky
#
# def launch_gpu_generation(prompts: list, gpu_type: str = "A100"):
#     """Launch generation on cloud GPU via SkyPilot."""
#     task = sky.Task(
#         run="python -m pipelines.infer.run_generation",
#         setup="pip install -r requirements.txt"
#     )
#     task.set_resources(sky.Resources(
#         accelerators={gpu_type: 1},
#         cloud=sky.AWS()  # or GCP, Azure
#     ))
#     sky.launch(task, cluster_name="flux-gen-cluster")
