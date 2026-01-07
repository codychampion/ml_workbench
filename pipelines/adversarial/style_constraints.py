"""Style constraints for adversarial patches using CLIP."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import torch.nn.functional as F
from typing import List, Union

from utils.clip_utils import load_clip, encode_text_prompts, prepare_images_for_clip


class CLIPStyleConstraint:
    """CLIP-based style constraint for adversarial patches."""

    def __init__(
        self,
        style_prompts: Union[str, List[str]],
        model_name: str = "ViT-B/32",
        device: str = "cuda"
    ):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.clip_model, _, self.clip_module = load_clip(model_name, device)
        self.style_prompts = [style_prompts] if isinstance(style_prompts, str) else style_prompts
        self.text_features = encode_text_prompts(self.style_prompts, self.clip_model, self.clip_module, device)

    def compute_style_loss(self, images: torch.Tensor) -> torch.Tensor:
        """Compute style loss (lower = better match to style prompts)."""
        images_normalized = prepare_images_for_clip(images, device=self.device)

        # Encode image features
        image_features = self.clip_model.encode_image(images_normalized)
        image_features /= image_features.norm(dim=-1, keepdim=True)

        # Compute similarity (scaled by 100 as in CLIP paper)
        similarities = 100.0 * image_features @ self.text_features.T
        max_similarity = similarities.max(dim=-1)[0]  # Best match across prompts

        return -max_similarity.mean()  # Negative because we want to maximize

    def get_similarity_score(self, images: torch.Tensor) -> float:
        """Get CLIP similarity score in [0, 1] range."""
        with torch.no_grad():
            loss = self.compute_style_loss(images)
            similarity = -loss.item() / 100.0  # Convert back to similarity
            return max(0.0, min(1.0, (similarity + 1) / 2))  # Normalize to [0, 1]


class TotalVariationLoss:
    """Total Variation loss for smooth patches."""

    def __init__(self, weight: float = 1.0):
        self.weight = weight

    def __call__(self, images: torch.Tensor) -> torch.Tensor:
        """Compute TV loss (lower = smoother)."""
        diff_h = torch.abs(images[:, :, :-1, :] - images[:, :, 1:, :])
        diff_w = torch.abs(images[:, :, :, :-1] - images[:, :, :, 1:])
        return self.weight * (diff_h.mean() + diff_w.mean())


class PrintabilityLoss:
    """Encourage patches to use printable colors."""

    def __init__(self, weight: float = 1.0, num_colors: int = 8):
        self.weight = weight
        self.num_colors = num_colors
        colors = torch.linspace(0, 1, num_colors)
        self.palette = torch.stack(torch.meshgrid(colors, colors, colors, indexing='ij')).reshape(3, -1).T

    def __call__(self, images: torch.Tensor) -> torch.Tensor:
        """Compute printability loss (distance to palette colors)."""
        images_flat = images.permute(0, 2, 3, 1).reshape(-1, 3)  # [B*H*W, 3]
        palette = self.palette.to(images.device)
        distances = torch.cdist(images_flat, palette, p=2)
        return self.weight * distances.min(dim=1)[0].mean()


class StyleConstraintCombined:
    """Combined style constraints: CLIP + TV + Printability."""

    def __init__(
        self,
        style_prompts: Union[str, List[str]],
        clip_weight: float = 1.0,
        tv_weight: float = 0.1,
        print_weight: float = 0.05,
        device: str = "cuda"
    ):
        self.clip_constraint = CLIPStyleConstraint(style_prompts, device=device)
        self.tv_loss = TotalVariationLoss(weight=tv_weight)
        self.print_loss = PrintabilityLoss(weight=print_weight)
        self.clip_weight = clip_weight
        self.tv_weight = tv_weight  # Store for prominence adjustment
        self.print_weight = print_weight

    def compute_loss(self, images: torch.Tensor) -> dict:
        """Compute all style losses."""
        clip_loss = self.clip_constraint.compute_style_loss(images)
        tv_loss = self.tv_loss(images)
        print_loss = self.print_loss(images)

        return {
            "style_total": self.clip_weight * clip_loss + tv_loss + print_loss,
            "style_clip": clip_loss,
            "style_tv": tv_loss,
            "style_print": print_loss,
        }

    def get_similarity_score(self, images: torch.Tensor) -> float:
        """Get CLIP similarity score for evaluation."""
        return self.clip_constraint.get_similarity_score(images)
