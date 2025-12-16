#!/usr/bin/env python3
"""
Adversarial Patch Generation Pipeline
======================================
Phase 1: CPU-only implementation with symbolic attacks for testing
         W&B offline logging and FiftyOne visualization integration

This script demonstrates adversarial attack visualization:
1. Load sample dataset via FiftyOne
2. Generate symbolic adversarial patches
3. Apply patches to images
4. Visualize results in FiftyOne
5. Log attack metrics to W&B

Security Note: This is for authorized research and educational purposes only.
Adversarial ML research helps improve model robustness and security.

PHASE 2/3 TODO:
- Implement real adversarial patch algorithms (PGD, FGSM variants)
- Add model robustness evaluation metrics
- Enable GPU acceleration for faster attack generation
- Integrate Prefect for batch processing workflows
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, Tuple

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import wandb
import fiftyone as fo
import fiftyone.zoo as foz

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config import get_config
from data_transfer import B2Client


class SymbolicPatchGenerator:
    """
    Symbolic adversarial patch generator for Phase 1 testing.

    Generates visually distinctive patches without actual adversarial optimization.
    Used to validate the visualization and logging pipeline.

    PHASE 2/3 TODO: Replace with real adversarial algorithms:
    - Projected Gradient Descent (PGD)
    - Fast Gradient Sign Method (FGSM)
    - Carlini & Wagner (C&W)
    - DeepFool
    """

    def __init__(self, device: str = "cpu"):
        self.device = device
        print(f"[SymbolicPatchGen] Initialized on device: {device}")

        # PHASE 2/3 TODO: Load target model for actual adversarial optimization
        # self.target_model = torchvision.models.resnet50(pretrained=True).to(device)
        # self.target_model.eval()

    def generate_patch(
        self,
        patch_size: Tuple[int, int] = (64, 64),
        target_class: Optional[int] = None,
        seed: Optional[int] = None,
        pattern_type: str = "noise"
    ) -> np.ndarray:
        """
        Generate a symbolic adversarial patch.

        Phase 1: Creates distinctive visual patterns
        Phase 2/3: Will use actual adversarial optimization

        Args:
            patch_size: Size of the patch (height, width)
            target_class: Target class for targeted attacks (unused in Phase 1)
            seed: Random seed for reproducibility
            pattern_type: Type of pattern ("noise", "checkerboard", "gradient")
        """
        if seed is not None:
            np.random.seed(seed)

        h, w = patch_size

        if pattern_type == "noise":
            # High-frequency noise pattern
            patch = np.random.rand(h, w, 3)
            patch = patch * 0.5 + 0.25  # Normalize to [0.25, 0.75]

        elif pattern_type == "checkerboard":
            # Checkerboard pattern (often used in adversarial research)
            patch = np.zeros((h, w, 3))
            block_size = max(h // 8, 4)
            for i in range(0, h, block_size):
                for j in range(0, w, block_size):
                    if (i // block_size + j // block_size) % 2 == 0:
                        patch[i:i+block_size, j:j+block_size] = [1.0, 0.5, 0.0]  # Orange
                    else:
                        patch[i:i+block_size, j:j+block_size] = [0.0, 0.5, 1.0]  # Blue

        elif pattern_type == "gradient":
            # Gradient pattern with random colors
            base_color = np.random.rand(3)
            x = np.linspace(0, 1, w)
            y = np.linspace(0, 1, h)
            xx, yy = np.meshgrid(x, y)
            patch = np.zeros((h, w, 3))
            for c in range(3):
                patch[:, :, c] = base_color[c] * (0.5 + 0.5 * np.sin(xx * 10 + yy * 10 + c))

        else:
            # Default: uniform random
            patch = np.random.rand(h, w, 3)

        # Add border to make patch visible
        border_width = 2
        patch[:border_width, :] = [1, 0, 0]  # Red top
        patch[-border_width:, :] = [1, 0, 0]  # Red bottom
        patch[:, :border_width] = [1, 0, 0]  # Red left
        patch[:, -border_width:] = [1, 0, 0]  # Red right

        patch = (np.clip(patch, 0, 1) * 255).astype(np.uint8)

        print(f"[SymbolicPatchGen] Generated {pattern_type} patch: {patch_size}")
        return patch

    def apply_patch(
        self,
        image: np.ndarray,
        patch: np.ndarray,
        position: Tuple[int, int] = (50, 50),
        alpha: float = 1.0
    ) -> np.ndarray:
        """
        Apply adversarial patch to an image.

        Args:
            image: Original image (H, W, 3)
            patch: Adversarial patch
            position: Top-left position for patch placement
            alpha: Blending factor (1.0 = full replacement)
        """
        result = image.copy()
        ph, pw = patch.shape[:2]
        y, x = position

        # Ensure patch fits within image bounds
        y = min(y, image.shape[0] - ph)
        x = min(x, image.shape[1] - pw)

        # Apply patch with alpha blending
        region = result[y:y+ph, x:x+pw]
        result[y:y+ph, x:x+pw] = (alpha * patch + (1 - alpha) * region).astype(np.uint8)

        return result


def init_wandb(config: Dict[str, Any], run_name: Optional[str] = None) -> wandb.sdk.wandb_run.Run:
    """Initialize W&B for experiment tracking (offline mode in Phase 1)."""
    app_config = get_config()

    run = wandb.init(
        project=app_config.wandb.project,
        entity=app_config.wandb.entity,
        mode=app_config.wandb.mode,
        dir=str(app_config.wandb.dir),
        name=run_name or f"adv-patch-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
        config=config,
        tags=["adversarial", "fiftyone", "phase1", "cpu"],
        notes="Phase 1: Symbolic adversarial patches for pipeline validation"
    )

    print(f"[W&B] Initialized run: {run.name}")
    print(f"[W&B] Mode: {app_config.wandb.mode}")

    return run


def load_fiftyone_dataset(
    dataset_name: str = "quickstart",
    max_samples: int = 10
) -> fo.Dataset:
    """
    Load a FiftyOne dataset for visualization.

    Phase 1: Uses FiftyOne's quickstart dataset (small, downloadable)
    Phase 2/3: Can use larger datasets from B2 storage
    """
    print(f"[FiftyOne] Loading dataset: {dataset_name}")

    try:
        # Try to load existing dataset
        dataset = fo.load_dataset(dataset_name)
        print(f"[FiftyOne] Loaded existing dataset with {len(dataset)} samples")
    except ValueError:
        # Download quickstart dataset
        print(f"[FiftyOne] Downloading {dataset_name} dataset...")
        dataset = foz.load_zoo_dataset(
            dataset_name,
            max_samples=max_samples,
            shuffle=True,
            seed=42
        )
        print(f"[FiftyOne] Downloaded {len(dataset)} samples")

    return dataset


def run_adversarial_pipeline(
    output_dir: Path,
    patch_size: Tuple[int, int] = (64, 64),
    num_samples: int = 5,
    pattern_types: list = None,
    seed: int = 42,
    launch_app: bool = False
) -> fo.Dataset:
    """
    Run the adversarial patch visualization pipeline.

    Steps:
    1. Initialize W&B tracking
    2. Load FiftyOne dataset
    3. Generate adversarial patches
    4. Apply patches to images
    5. Create FiftyOne visualization dataset
    6. Log results to W&B
    """
    config = get_config()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if pattern_types is None:
        pattern_types = ["noise", "checkerboard", "gradient"]

    # Pipeline configuration
    pipeline_config = {
        "patch_size": patch_size,
        "num_samples": num_samples,
        "pattern_types": pattern_types,
        "seed": seed,
        "device": config.compute.device,
        "phase": "1-local-core"
    }

    # Step 1: Initialize W&B
    run = init_wandb(pipeline_config)

    # Step 2: Load FiftyOne dataset
    print("\n[Pipeline] Loading FiftyOne dataset...")
    source_dataset = load_fiftyone_dataset(max_samples=num_samples)

    # Step 3: Initialize patch generator
    generator = SymbolicPatchGenerator(device=config.compute.device)

    # Step 4: Initialize B2 client for data I/O (mocked)
    b2_client = B2Client()

    # Step 5: Create visualization dataset
    viz_dataset_name = f"adversarial_patches_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Delete if exists
    if viz_dataset_name in fo.list_datasets():
        fo.delete_dataset(viz_dataset_name)

    viz_dataset = fo.Dataset(viz_dataset_name)
    viz_dataset.persistent = True

    print(f"\n[Pipeline] Processing {num_samples} samples with {len(pattern_types)} pattern types...")

    # Tables for W&B logging
    attack_table = wandb.Table(columns=[
        "sample_id", "pattern_type", "original", "patched", "patch"
    ])

    sample_idx = 0
    for sample in source_dataset.take(num_samples):
        # Load original image
        original_path = sample.filepath
        original_image = np.array(Image.open(original_path).convert("RGB"))

        for pattern_type in pattern_types:
            print(f"\n[Pipeline] Sample {sample_idx + 1}: {pattern_type} pattern")

            # Generate patch
            patch = generator.generate_patch(
                patch_size=patch_size,
                pattern_type=pattern_type,
                seed=seed + sample_idx
            )

            # Determine patch position (center of image)
            h, w = original_image.shape[:2]
            position = (h // 2 - patch_size[0] // 2, w // 2 - patch_size[1] // 2)

            # Apply patch
            patched_image = generator.apply_patch(
                original_image,
                patch,
                position=position
            )

            # Save patched image
            patched_path = output_dir / f"patched_{sample_idx:04d}_{pattern_type}.png"
            Image.fromarray(patched_image).save(patched_path)

            # Save patch separately
            patch_path = output_dir / f"patch_{sample_idx:04d}_{pattern_type}.png"
            Image.fromarray(patch).save(patch_path)

            # Add to FiftyOne dataset
            fo_sample = fo.Sample(filepath=str(patched_path))
            fo_sample["original_path"] = original_path
            fo_sample["pattern_type"] = pattern_type
            fo_sample["patch_position"] = list(position)
            fo_sample["patch_size"] = list(patch_size)
            fo_sample["seed"] = seed + sample_idx

            # Add ground truth if available
            if sample.ground_truth is not None:
                fo_sample["ground_truth"] = sample.ground_truth

            viz_dataset.add_sample(fo_sample)

            # Log to W&B
            attack_table.add_data(
                f"sample_{sample_idx}",
                pattern_type,
                wandb.Image(original_image),
                wandb.Image(patched_image),
                wandb.Image(patch)
            )

            # Upload to B2 (mocked)
            b2_client.upload_file(
                source=patched_path,
                destination_name=f"outputs/adversarial/{patched_path.name}"
            )

            sample_idx += 1

    # Log summary to W&B
    wandb.log({
        "attack/summary_table": attack_table,
        "attack/total_samples": sample_idx,
        "attack/pattern_types": len(pattern_types)
    })

    # Save dataset
    viz_dataset.save()

    print(f"\n[Pipeline] Created FiftyOne dataset: {viz_dataset_name}")
    print(f"[Pipeline] Total samples: {len(viz_dataset)}")

    # Launch FiftyOne App if requested
    if launch_app:
        print("\n[FiftyOne] Launching visualization app...")
        print("[FiftyOne] Open http://localhost:5151 in your browser")
        session = fo.launch_app(viz_dataset, port=5151, address="0.0.0.0")

        # Keep the app running
        print("[FiftyOne] Press Ctrl+C to stop the app")
        try:
            session.wait()
        except KeyboardInterrupt:
            print("\n[FiftyOne] Shutting down...")

    # Finish W&B run
    wandb.finish()

    print(f"\n[Pipeline] Complete!")
    print(f"[Pipeline] Output directory: {output_dir}")
    print(f"[Pipeline] FiftyOne dataset: {viz_dataset_name}")

    return viz_dataset


def main():
    """Main entry point for the adversarial patch script."""
    parser = argparse.ArgumentParser(
        description="Adversarial Patch Generation Pipeline (Phase 1: Symbolic)"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./outputs/adversarial-patches"),
        help="Directory to save outputs"
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        nargs=2,
        default=[64, 64],
        help="Patch size (height width)"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=5,
        help="Number of samples to process"
    )
    parser.add_argument(
        "--patterns",
        nargs="+",
        default=["noise", "checkerboard", "gradient"],
        choices=["noise", "checkerboard", "gradient"],
        help="Pattern types to generate"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--launch-app",
        action="store_true",
        help="Launch FiftyOne visualization app after processing"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("Adversarial Patch Generation Pipeline")
    print("Phase 1: Local Core (CPU-only, Symbolic Attacks)")
    print("=" * 60)
    print("\nSecurity Notice: This tool is for authorized research and")
    print("educational purposes in adversarial machine learning.")
    print("=" * 60)

    # Run the pipeline
    dataset = run_adversarial_pipeline(
        output_dir=args.output_dir,
        patch_size=tuple(args.patch_size),
        num_samples=args.num_samples,
        pattern_types=args.patterns,
        seed=args.seed,
        launch_app=args.launch_app
    )

    print("\n" + "=" * 60)
    print("Pipeline Execution Summary")
    print("=" * 60)
    print(f"Processed samples: {len(dataset)}")
    print(f"Output directory: {args.output_dir}")
    print(f"FiftyOne dataset: {dataset.name}")
    print(f"W&B mode: offline (Phase 1)")
    print("\nTo view results in FiftyOne, run:")
    print(f"  python -c \"import fiftyone as fo; fo.load_dataset('{dataset.name}').launch()\"")
    print("\nTo sync W&B results, run:")
    print("  wandb sync ./outputs/wandb/offline-*")
    print("=" * 60)


if __name__ == "__main__":
    main()


# PHASE 2/3 TODO: Prefect workflow integration
# from prefect import flow, task
#
# @task(retries=2)
# def load_dataset_task(dataset_name: str, num_samples: int):
#     """Load FiftyOne dataset with retry logic."""
#     return load_fiftyone_dataset(dataset_name, num_samples)
#
# @task
# def generate_patches_task(samples, patch_config: dict):
#     """Generate adversarial patches for all samples."""
#     pass
#
# @task
# def evaluate_robustness_task(model, patched_images):
#     """Evaluate model robustness against patches."""
#     pass
#
# @flow(name="adversarial-evaluation-pipeline")
# def adversarial_flow(config: dict):
#     """Prefect flow for adversarial evaluation."""
#     dataset = load_dataset_task(config["dataset"], config["num_samples"])
#     patches = generate_patches_task(dataset, config["patch_config"])
#     results = evaluate_robustness_task(config["model"], patches)
#     return results


# PHASE 2/3 TODO: SkyPilot integration for GPU-accelerated attacks
# import sky
#
# def launch_gpu_attack_eval(config: dict, gpu_type: str = "V100"):
#     """Launch attack evaluation on cloud GPU via SkyPilot."""
#     task = sky.Task(
#         run="python projects/adversarial-patches/generate_patch.py --gpu",
#         setup="pip install -r requirements.txt"
#     )
#     task.set_resources(sky.Resources(
#         accelerators={gpu_type: 1},
#         use_spot=True,  # Cost optimization
#         cloud=sky.GCP()
#     ))
#     sky.launch(task, cluster_name="adv-patch-cluster")
