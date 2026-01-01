"""Physical transformations for adversarial robustness (EoT training)."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import torch.nn.functional as F
from typing import Tuple, Optional

from utils.transforms import (
    apply_affine_transform,
    create_rotation_matrix,
    create_scale_matrix,
    apply_color_jitter
)


class PhysicalTransformPipeline:
    """Simulate physical world transformations for robust adversarial patches."""

    def __init__(
        self,
        brightness_range: Tuple[float, float] = (0.8, 1.2),
        contrast_range: Tuple[float, float] = (0.8, 1.2),
        rotation_range: Tuple[float, float] = (-15, 15),
        scale_range: Tuple[float, float] = (0.8, 1.2),
        noise_std: float = 0.01,
        device: str = "cuda"
    ):
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.rotation_range = rotation_range
        self.scale_range = scale_range
        self.noise_std = noise_std
        self.device = device

    def apply_rotation(self, images: torch.Tensor, angle: Optional[float] = None) -> torch.Tensor:
        """Apply random rotation."""
        B = images.size(0)
        if angle is None:
            angles = torch.empty(B, device=self.device).uniform_(*self.rotation_range)
        else:
            angles = torch.full((B,), angle, device=self.device)

        theta = create_rotation_matrix(B, angles, self.device)
        return apply_affine_transform(images, theta)

    def apply_scale(self, images: torch.Tensor, scale: Optional[float] = None) -> torch.Tensor:
        """Apply random scaling."""
        B = images.size(0)
        if scale is None:
            scales = torch.empty(B, device=self.device).uniform_(*self.scale_range)
        else:
            scales = torch.full((B,), scale, device=self.device)

        theta = create_scale_matrix(B, scales, self.device)
        return apply_affine_transform(images, theta)

    def apply_noise(self, images: torch.Tensor, std: Optional[float] = None) -> torch.Tensor:
        """Apply Gaussian noise."""
        if std is None:
            std = self.noise_std

        noise = torch.randn_like(images) * std
        return torch.clamp(images + noise, 0, 1)

    def apply_perspective(self, images: torch.Tensor, distortion: float = 0.1) -> torch.Tensor:
        """Apply random perspective transformation."""
        B, C, H, W = images.shape

        # Random perspective distortion
        src_points = torch.tensor([
            [0, 0], [W-1, 0], [W-1, H-1], [0, H-1]
        ], dtype=torch.float32, device=self.device).unsqueeze(0).repeat(B, 1, 1)

        # Add random offsets
        offsets = torch.randn(B, 4, 2, device=self.device) * distortion * min(H, W)
        dst_points = src_points + offsets

        # Compute perspective transformation (simplified - would need full implementation)
        # For now, use affine as approximation
        # Full perspective would require cv2 or kornia
        return images  # Placeholder

    def apply_random_transforms(self, images: torch.Tensor) -> torch.Tensor:
        """Apply all random transformations."""
        images = apply_color_jitter(images, self.brightness_range, self.contrast_range, self.device)
        images = self.apply_rotation(images)
        images = self.apply_scale(images)
        images = self.apply_noise(images)
        return images

    def __call__(self, images: torch.Tensor) -> torch.Tensor:
        """Apply random transformations (alias for apply_random_transforms)."""
        return self.apply_random_transforms(images)


class PatchApplicator:
    """Apply adversarial patch to images at random locations and scales."""

    def __init__(
        self,
        patch_size: Tuple[int, int] = (300, 300),
        apply_location: str = "random",  # 'random', 'center', 'chest'
        device: str = "cuda"
    ):
        """
        Initialize patch applicator.

        Args:
            patch_size: Size of the patch (H, W)
            apply_location: Where to place patch on images
            device: Device to run on
        """
        self.patch_size = patch_size
        self.apply_location = apply_location
        self.device = device

    def apply_patch(
        self,
        images: torch.Tensor,
        patch: torch.Tensor,
        locations: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Apply patch to images.

        Args:
            images: Batch of images [B, 3, H, W]
            patch: Adversarial patch [3, patch_h, patch_w]
            locations: Optional specific locations [B, 2] (y, x coordinates)

        Returns:
            Images with patch applied
        """
        B, C, H, W = images.shape
        patch_h, patch_w = self.patch_size

        # Ensure patch is correct size
        if patch.shape[-2:] != (patch_h, patch_w):
            patch = F.interpolate(
                patch.unsqueeze(0),
                size=(patch_h, patch_w),
                mode='bilinear'
            ).squeeze(0)

        # Determine patch locations
        if locations is None:
            if self.apply_location == "random":
                # Random locations
                y = torch.randint(0, H - patch_h, (B,), device=self.device)
                x = torch.randint(0, W - patch_w, (B,), device=self.device)
            elif self.apply_location == "center":
                # Center of image
                y = torch.full((B,), (H - patch_h) // 2, device=self.device)
                x = torch.full((B,), (W - patch_w) // 2, device=self.device)
            elif self.apply_location == "chest":
                # Upper-center (approximating chest area for person)
                y = torch.full((B,), H // 3, device=self.device)
                x = torch.full((B,), (W - patch_w) // 2, device=self.device)
            else:
                raise ValueError(f"Unknown location: {self.apply_location}")
        else:
            y, x = locations[:, 0], locations[:, 1]

        # Apply patch to each image
        patched_images = images.clone()
        for i in range(B):
            patched_images[i, :, y[i]:y[i]+patch_h, x[i]:x[i]+patch_w] = patch

        return patched_images


class EOTWrapper:
    """Expectation over Transformation wrapper for robust adversarial training."""

    def __init__(
        self,
        transform_pipeline: PhysicalTransformPipeline,
        num_samples: int = 20
    ):
        """
        Initialize EoT wrapper.

        Args:
            transform_pipeline: Physical transformation pipeline
            num_samples: Number of transformed samples per image
        """
        self.transform_pipeline = transform_pipeline
        self.num_samples = num_samples

    def compute_eot_loss(
        self,
        images: torch.Tensor,
        loss_fn: callable
    ) -> torch.Tensor:
        """
        Compute loss averaged over multiple random transformations.

        Args:
            images: Batch of images
            loss_fn: Function that takes images and returns loss

        Returns:
            Average loss over transformations
        """
        total_loss = 0.0

        for _ in range(self.num_samples):
            # Apply random transformations
            transformed = self.transform_pipeline(images)

            # Compute loss on transformed images
            loss = loss_fn(transformed)
            total_loss += loss

        # Average over samples
        avg_loss = total_loss / self.num_samples
        return avg_loss
