"""Image transformation utilities for adversarial robustness."""

import torch
import torch.nn.functional as F
from typing import Tuple


def apply_affine_transform(
    images: torch.Tensor,
    theta: torch.Tensor,
    padding_mode: str = 'border'
) -> torch.Tensor:
    """
    Apply affine transformation to images.

    Args:
        images: Images [B, 3, H, W]
        theta: Affine matrix [B, 2, 3]
        padding_mode: border, zeros, or reflection

    Returns:
        Transformed images
    """
    grid = F.affine_grid(theta, images.size(), align_corners=False)
    return F.grid_sample(images, grid, align_corners=False, padding_mode=padding_mode)


def create_rotation_matrix(
    batch_size: int,
    angles_deg: torch.Tensor,
    device: str = "cuda"
) -> torch.Tensor:
    """
    Create rotation affine matrices.

    Args:
        batch_size: Batch size
        angles_deg: Rotation angles in degrees [B]
        device: Device

    Returns:
        Affine matrices [B, 2, 3]
    """
    angles_rad = angles_deg * (3.14159265359 / 180.0)
    cos = torch.cos(angles_rad)
    sin = torch.sin(angles_rad)

    theta = torch.zeros(batch_size, 2, 3, device=device)
    theta[:, 0, 0] = cos
    theta[:, 0, 1] = -sin
    theta[:, 1, 0] = sin
    theta[:, 1, 1] = cos

    return theta


def create_scale_matrix(
    batch_size: int,
    scales: torch.Tensor,
    device: str = "cuda"
) -> torch.Tensor:
    """
    Create scaling affine matrices.

    Args:
        batch_size: Batch size
        scales: Scale factors [B]
        device: Device

    Returns:
        Affine matrices [B, 2, 3]
    """
    theta = torch.zeros(batch_size, 2, 3, device=device)
    theta[:, 0, 0] = scales
    theta[:, 1, 1] = scales

    return theta


def apply_color_jitter(
    images: torch.Tensor,
    brightness: Tuple[float, float] = (0.8, 1.2),
    contrast: Tuple[float, float] = (0.8, 1.2),
    device: str = "cuda"
) -> torch.Tensor:
    """
    Apply random brightness and contrast.

    Args:
        images: Images [B, 3, H, W] in [0, 1]
        brightness: (min, max) brightness multiplier
        contrast: (min, max) contrast multiplier
        device: Device

    Returns:
        Jittered images
    """
    B = images.size(0)

    # Brightness
    brightness_factor = torch.empty(B, 1, 1, 1, device=device).uniform_(*brightness)
    images = torch.clamp(images * brightness_factor, 0, 1)

    # Contrast
    contrast_factor = torch.empty(B, 1, 1, 1, device=device).uniform_(*contrast)
    mean = images.mean(dim=[2, 3], keepdim=True)
    images = torch.clamp((images - mean) * contrast_factor + mean, 0, 1)

    return images
