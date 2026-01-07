#!/usr/bin/env python3
"""
Adversarial Patch Generator with Style Constraints
===================================================
Generate adversarial patches that fool ML models while maintaining
aesthetic constraints (e.g., punk band patch style).

Usage:
    python -m pipelines.adversarial.generate_adv_patch \
        --target-model yolov8 \
        --attack evasion \
        --style "punk band patch, DIY aesthetic, black and white" \
        --test-images ./data/test/coco_persons \
        --output ./outputs/adv_patches/punk_evasion_v1.png

    python -m pipelines.adversarial.generate_adv_patch --hydra
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Any, Dict
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from utils.decorators import flow, task

# Check dependencies
try:
    import torch
    import torchvision
    from torchvision import transforms
    from torchvision.io import read_image
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import clip
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False

try:
    import hydra
    from omegaconf import DictConfig
except ImportError:
    hydra = None
    DictConfig = Any

# Import our modules
if TORCH_AVAILABLE:
    from pipelines.adversarial.target_models import get_target_model
    from pipelines.adversarial.style_constraints import StyleConstraintCombined
    from pipelines.adversarial.patch_optimizer import AdversarialPatchOptimizer


def load_test_images(
    image_dir: Path,
    num_images: int = 10,
    image_size: tuple = (640, 640)
) -> torch.Tensor:
    """
    Load test images for patch optimization.

    Args:
        image_dir: Directory containing test images
        num_images: Number of images to load
        image_size: Target image size

    Returns:
        Batch of images [B, 3, H, W]
    """
    image_dir = Path(image_dir)
    if not image_dir.exists():
        raise ValueError(f"Image directory not found: {image_dir}")

    # Find image files
    image_extensions = {".jpg", ".jpeg", ".png", ".webp"}
    image_files = [
        f for f in image_dir.iterdir()
        if f.suffix.lower() in image_extensions
    ][:num_images]

    if len(image_files) == 0:
        raise ValueError(f"No images found in {image_dir}")

    print(f"Loading {len(image_files)} test images from {image_dir}")

    # Load and preprocess images
    images = []
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ConvertImageDtype(torch.float),
    ])

    for img_path in image_files:
        img = read_image(str(img_path))
        img = transform(img)
        if img.shape[0] == 1:  # Grayscale
            img = img.repeat(3, 1, 1)
        images.append(img)

    # Stack into batch
    images_batch = torch.stack(images)
    return images_batch


@task(name="generate-adversarial-patch")
def generate_adversarial_patch(
    target_model: str,
    attack_objective: str,
    style_prompt: str,
    test_image_dir: Path,
    patch_size: tuple = (300, 300),
    iterations: int = 500,
    lambda_adv: float = 0.5,
    lambda_style: float = 0.4,
    use_eot: bool = True,
    eot_samples: int = 10,
    adversarial_prominence: float = 2.0,
    learning_rate: float = 0.01,
    num_test_images: int = 10,
    output_path: Path = Path("./outputs/adv_patches/patch.png"),
) -> Dict[str, Any]:
    """
    Generate adversarial patch with style constraints.

    Args:
        target_model: Model to attack (yolov8, clip-classifier)
        attack_objective: Attack type (evasion, misclassification)
        style_prompt: CLIP prompt for style constraint
        test_image_dir: Directory with test images
        patch_size: Patch dimensions (height, width)
        iterations: Optimization iterations
        lambda_adv: Weight for adversarial loss
        lambda_style: Weight for style loss
        use_eot: Whether to use Expectation over Transformation
        eot_samples: Number of EoT samples
        adversarial_prominence: Visibility of adversarial noise (0.5=subtle, 1.0=balanced, 2.0=prominent)
        learning_rate: Learning rate for optimization
        num_test_images: Number of test images to use
        output_path: Output path for patch

    Returns:
        Dict with patch path, metrics, and evaluation results
    """
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch not installed. Install with: pip install torch torchvision")

    if not CLIP_AVAILABLE:
        raise RuntimeError(
            "CLIP not installed. Install with: pip install git+https://github.com/openai/CLIP.git"
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"\n{'='*60}")
    print(f"Adversarial Patch Generation")
    print(f"{'='*60}")
    print(f"Target Model: {target_model}")
    print(f"Attack: {attack_objective}")
    print(f"Style: {style_prompt}")
    print(f"Patch Size: {patch_size}")
    print(f"Iterations: {iterations}")
    print(f"Adversarial Prominence: {adversarial_prominence}x (2.0=prominent, 1.0=balanced)")
    print(f"Device: {device}")
    print(f"EoT: {'Enabled' if use_eot else 'Disabled'}")
    if use_eot:
        print(f"EoT Samples: {eot_samples}")

    # Load test images
    print(f"\nLoading test images...")
    test_images = load_test_images(test_image_dir, num_test_images)
    test_images = test_images.to(device)
    print(f"Loaded {test_images.shape[0]} images of size {test_images.shape[2:]}")

    # Initialize target model
    print(f"\nInitializing target model: {target_model}")
    target_model_obj = get_target_model(target_model, device=device)

    # Initialize style constraints
    print(f"\nInitializing style constraints...")
    print(f"Style prompt: '{style_prompt}'")
    style_constraint = StyleConstraintCombined(
        style_prompts=style_prompt,
        clip_weight=1.0,
        tv_weight=0.1,
        print_weight=0.05,
        device=device
    )

    # Initialize optimizer
    print(f"\nInitializing patch optimizer...")
    optimizer = AdversarialPatchOptimizer(
        target_model=target_model_obj,
        style_constraint=style_constraint,
        patch_size=patch_size,
        lambda_adv=lambda_adv,
        lambda_style=lambda_style,
        use_eot=use_eot,
        eot_samples=eot_samples,
        adversarial_prominence=adversarial_prominence,
        device=device
    )

    # Optimize patch
    print(f"\n{'='*60}")
    print(f"Starting optimization...")
    print(f"{'='*60}\n")

    result = optimizer.optimize(
        test_images=test_images,
        iterations=iterations,
        learning_rate=learning_rate,
        attack_objective=attack_objective,
        save_path=output_path,
        log_interval=50
    )

    # Save metadata
    metadata = {
        "target_model": target_model,
        "attack_objective": attack_objective,
        "style_prompt": style_prompt,
        "patch_size": list(patch_size),
        "iterations": iterations,
        "lambda_adv": lambda_adv,
        "lambda_style": lambda_style,
        "use_eot": use_eot,
        "eot_samples": eot_samples if use_eot else 0,
        "learning_rate": learning_rate,
        "final_losses": result["final_losses"],
        "style_score": result["style_score"],
        "generated_at": datetime.now().isoformat(),
        "device": device,
    }

    metadata_path = output_path.with_suffix(".json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Generation Complete!")
    print(f"{'='*60}")
    print(f"Patch saved to: {output_path}")
    print(f"Metadata saved to: {metadata_path}")
    print(f"\nFinal Metrics:")
    print(f"  Adversarial Loss: {result['final_losses']['adversarial']:.4f}")
    print(f"  Style Loss: {result['final_losses']['style_total']:.4f}")
    print(f"  Style Similarity: {result['style_score']:.4f}")

    return {
        "patch_path": str(output_path),
        "metadata_path": str(metadata_path),
        **result
    }


def main():
    """CLI interface for adversarial patch generation."""
    parser = argparse.ArgumentParser(
        description="Generate adversarial patches with style constraints"
    )
    parser.add_argument(
        "--target-model",
        required=True,
        choices=["yolov8", "clip-classifier"],
        help="Model to attack"
    )
    parser.add_argument(
        "--attack",
        default="evasion",
        choices=["evasion", "misclassification"],
        help="Attack objective"
    )
    parser.add_argument(
        "--style",
        default="punk band patch, DIY aesthetic, black and white",
        help="CLIP prompt for style constraint"
    )
    parser.add_argument(
        "--test-images",
        type=Path,
        required=True,
        help="Directory containing test images"
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        nargs=2,
        default=[300, 300],
        help="Patch dimensions (height width)"
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=500,
        help="Optimization iterations"
    )
    parser.add_argument(
        "--lambda-adv",
        type=float,
        default=0.5,
        help="Weight for adversarial loss"
    )
    parser.add_argument(
        "--lambda-style",
        type=float,
        default=0.4,
        help="Weight for style loss"
    )
    parser.add_argument(
        "--no-eot",
        action="store_true",
        help="Disable Expectation over Transformation"
    )
    parser.add_argument(
        "--eot-samples",
        type=int,
        default=10,
        help="Number of EoT samples"
    )
    parser.add_argument(
        "--adversarial-prominence",
        type=float,
        default=2.0,
        help="Visibility of adversarial noise (0.5=subtle, 1.0=balanced, 2.0=prominent/glitchy)"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.01,
        help="Learning rate"
    )
    parser.add_argument(
        "--num-test-images",
        type=int,
        default=10,
        help="Number of test images to use"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("./outputs/adv_patches/patch.png"),
        help="Output path for patch"
    )
    parser.add_argument(
        "--hydra",
        action="store_true",
        help="Use Hydra config instead of CLI args"
    )

    args = parser.parse_args()

    result = generate_adversarial_patch(
        target_model=args.target_model,
        attack_objective=args.attack,
        style_prompt=args.style,
        test_image_dir=args.test_images,
        patch_size=tuple(args.patch_size),
        iterations=args.iterations,
        lambda_adv=args.lambda_adv,
        lambda_style=args.lambda_style,
        use_eot=not args.no_eot,
        eot_samples=args.eot_samples,
        adversarial_prominence=args.adversarial_prominence,
        learning_rate=args.learning_rate,
        num_test_images=args.num_test_images,
        output_path=args.output,
    )


if __name__ == "__main__":
    # Support Hydra-driven runs with --hydra flag
    if hydra is not None and "--hydra" in sys.argv:
        sys.argv = [arg for arg in sys.argv if arg != "--hydra"]

        @hydra.main(version_base="1.2", config_path="../../conf", config_name="config")
        def hydra_main(cfg: "DictConfig"):  # type: ignore[misc]
            adv_cfg = cfg.get("pipeline", {}).get("adversarial_patch", {})

            target_cfg = adv_cfg.get("target_model", {})
            style_cfg = adv_cfg.get("style", {})
            patch_cfg = adv_cfg.get("patch", {})
            opt_cfg = adv_cfg.get("optimization", {})
            output_cfg = adv_cfg.get("output", {})

            result = generate_adversarial_patch(
                target_model=target_cfg.get("type", "yolov8"),
                attack_objective=target_cfg.get("attack_objective", "evasion"),
                style_prompt=style_cfg.get("prompt", "punk band patch"),
                test_image_dir=Path(adv_cfg.get("test_images", "./data/test/coco_persons")),
                patch_size=tuple(patch_cfg.get("size", [300, 300])),
                iterations=opt_cfg.get("iterations", 500),
                lambda_adv=opt_cfg.get("lambda_adv", 0.5),
                lambda_style=opt_cfg.get("lambda_style", 0.4),
                use_eot=opt_cfg.get("use_eot", True),
                eot_samples=opt_cfg.get("eot_samples", 10),
                learning_rate=opt_cfg.get("learning_rate", 0.01),
                output_path=Path(output_cfg.get("path", "./outputs/adv_patches/patch.png")),
            )

        hydra_main()
    else:
        main()
