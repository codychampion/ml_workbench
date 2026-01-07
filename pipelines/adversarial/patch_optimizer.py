"""
Adversarial Patch Optimizer
============================
Optimize adversarial patches with style constraints and physical robustness.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Optional, Tuple
from pathlib import Path
from tqdm import tqdm

from pipelines.adversarial.target_models import TargetModel
from pipelines.adversarial.style_constraints import StyleConstraintCombined
from pipelines.adversarial.physical_transforms import PhysicalTransformPipeline, PatchApplicator, EOTWrapper


class AdversarialPatchOptimizer:
    """Optimize adversarial patches with combined objectives."""

    def __init__(
        self,
        target_model: TargetModel,
        style_constraint: StyleConstraintCombined,
        patch_size: Tuple[int, int] = (300, 300),
        lambda_adv: float = 0.5,
        lambda_style: float = 0.4,
        use_eot: bool = True,
        eot_samples: int = 10,
        adversarial_prominence: float = 1.0,
        device: str = "cuda"
    ):
        """
        Initialize adversarial patch optimizer.

        Args:
            target_model: Model to attack
            style_constraint: Style constraint module
            patch_size: Size of patch (H, W)
            lambda_adv: Weight for adversarial loss
            lambda_style: Weight for style loss
            use_eot: Whether to use Expectation over Transformation
            eot_samples: Number of EoT samples
            adversarial_prominence: How visible adversarial noise is (0.0=hidden, 2.0=very prominent)
            device: Device to run on
        """
        self.target_model = target_model
        self.style_constraint = style_constraint
        self.patch_size = patch_size
        self.lambda_adv = lambda_adv
        self.lambda_style = lambda_style
        self.adversarial_prominence = adversarial_prominence
        self.device = device

        # Adjust style constraint weights based on prominence
        # Higher prominence = less smoothing (TV loss), less printability constraint
        self.style_constraint.tv_weight *= (1.0 / max(adversarial_prominence, 0.1))
        self.style_constraint.print_weight *= (1.0 / max(adversarial_prominence, 0.1))

        # Initialize patch (random or from style)
        self.patch = self._initialize_patch()

        # Physical transformations
        self.transform_pipeline = PhysicalTransformPipeline(device=device)
        self.patch_applicator = PatchApplicator(patch_size=patch_size, device=device)

        # EoT wrapper
        self.use_eot = use_eot
        if use_eot:
            self.eot = EOTWrapper(self.transform_pipeline, num_samples=eot_samples)

    def _initialize_patch(self, method: str = "random") -> nn.Parameter:
        """
        Initialize the adversarial patch.

        Args:
            method: Initialization method ('random', 'gray', 'white')

        Returns:
            Patch as nn.Parameter
        """
        if method == "random":
            patch = torch.rand(3, *self.patch_size, device=self.device)
        elif method == "gray":
            patch = torch.full((3, *self.patch_size), 0.5, device=self.device)
        elif method == "white":
            patch = torch.ones(3, *self.patch_size, device=self.device)
        else:
            raise ValueError(f"Unknown init method: {method}")

        # Make it a learnable parameter
        patch = nn.Parameter(patch, requires_grad=True)
        return patch

    def compute_combined_loss(
        self,
        test_images: torch.Tensor,
        attack_objective: str = "evasion"
    ) -> Dict[str, torch.Tensor]:
        """
        Compute combined loss (adversarial + style).

        Args:
            test_images: Batch of test images to apply patch on
            attack_objective: Attack objective for adversarial loss

        Returns:
            Dictionary of losses
        """
        # Apply patch to test images
        patched_images = self.patch_applicator.apply_patch(
            test_images,
            self.patch
        )

        # Define loss computation function for EoT
        def compute_loss(images):
            # Adversarial loss
            adv_loss = self.target_model.compute_loss(images, attack_objective)

            # Style loss (only on patch, not full image)
            # Extract patch regions for style evaluation
            patch_only = self.patch.unsqueeze(0)  # [1, 3, H, W]
            style_losses = self.style_constraint.compute_loss(patch_only)

            return adv_loss, style_losses

        # Apply EoT if enabled
        if self.use_eot:
            total_adv_loss = 0.0
            total_style_losses = {k: 0.0 for k in ["style_total", "style_clip", "style_tv", "style_print"]}

            for _ in range(self.eot.num_samples):
                transformed = self.transform_pipeline(patched_images)
                adv_loss, style_losses = compute_loss(transformed)
                total_adv_loss += adv_loss
                for k, v in style_losses.items():
                    total_style_losses[k] += v

            # Average
            adv_loss = total_adv_loss / self.eot.num_samples
            style_losses = {k: v / self.eot.num_samples for k, v in total_style_losses.items()}
        else:
            adv_loss, style_losses = compute_loss(patched_images)

        # Combined loss
        total_loss = (
            self.lambda_adv * adv_loss +
            self.lambda_style * style_losses["style_total"]
        )

        return {
            "total": total_loss,
            "adversarial": adv_loss,
            **style_losses
        }

    def optimize(
        self,
        test_images: torch.Tensor,
        iterations: int = 500,
        learning_rate: float = 0.01,
        attack_objective: str = "evasion",
        save_path: Optional[Path] = None,
        log_interval: int = 50
    ) -> Dict[str, any]:
        """
        Optimize the adversarial patch.

        Args:
            test_images: Batch of images to test attack on
            iterations: Number of optimization iterations
            learning_rate: Learning rate for optimizer
            attack_objective: Attack objective
            save_path: Optional path to save patch
            log_interval: How often to log progress

        Returns:
            Dictionary with final patch and metrics
        """
        # Optimizer
        optimizer = optim.Adam([self.patch], lr=learning_rate)

        # Track metrics
        history = {
            "total_loss": [],
            "adv_loss": [],
            "style_loss": [],
        }

        # Optimization loop
        pbar = tqdm(range(iterations), desc="Optimizing patch")
        for iteration in pbar:
            optimizer.zero_grad()

            # Compute losses
            losses = self.compute_combined_loss(test_images, attack_objective)

            # Backward pass
            losses["total"].backward()

            # Gradient clipping (optional)
            torch.nn.utils.clip_grad_norm_([self.patch], max_norm=1.0)

            # Update patch
            optimizer.step()

            # Clamp patch to valid range [0, 1]
            with torch.no_grad():
                self.patch.data.clamp_(0, 1)

            # Log progress
            if iteration % log_interval == 0:
                history["total_loss"].append(losses["total"].item())
                history["adv_loss"].append(losses["adversarial"].item())
                history["style_loss"].append(losses["style_total"].item())

                pbar.set_postfix({
                    "total": f"{losses['total'].item():.4f}",
                    "adv": f"{losses['adversarial'].item():.4f}",
                    "style": f"{losses['style_total'].item():.4f}"
                })

        # Final evaluation
        with torch.no_grad():
            final_losses = self.compute_combined_loss(test_images, attack_objective)
            style_score = self.style_constraint.get_similarity_score(
                self.patch.unsqueeze(0)
            )

        # Save patch if requested
        if save_path:
            self.save_patch(save_path)

        return {
            "patch": self.patch.detach().cpu(),
            "final_losses": {k: v.item() if torch.is_tensor(v) else v
                            for k, v in final_losses.items()},
            "style_score": style_score,
            "history": history
        }

    def save_patch(self, path: Path):
        """Save patch as image file."""
        from torchvision.utils import save_image
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        save_image(self.patch.data, path)
        print(f"Patch saved to: {path}")

    def load_patch(self, path: Path):
        """Load patch from image file."""
        from torchvision.io import read_image
        patch = read_image(str(path)).float() / 255.0
        self.patch = nn.Parameter(patch.to(self.device), requires_grad=True)
        print(f"Patch loaded from: {path}")
