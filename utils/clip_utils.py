"""CLIP model utilities (loading, normalization, encoding)."""

import torch
import torch.nn.functional as F
from typing import Tuple, List, Union, Optional

from utils.constants import CLIP_MEAN, CLIP_STD, CLIP_INPUT_SIZE, DEFAULT_CLIP_MODEL


def load_clip(model_name: str = DEFAULT_CLIP_MODEL, device: str = "cuda"):
    """
    Load CLIP model with error handling.

    Args:
        model_name: CLIP model variant (e.g., "ViT-B/32")
        device: Device to load model on

    Returns:
        (model, preprocess, clip_module)

    Raises:
        ImportError: If CLIP not installed
    """
    try:
        import clip
        model, preprocess = clip.load(model_name, device=device)
        model.eval()
        return model, preprocess, clip
    except ImportError:
        raise ImportError(
            "CLIP not installed. Install with:\n"
            "pip install git+https://github.com/openai/CLIP.git"
        )


def normalize_for_clip(images: torch.Tensor, device: str = "cuda") -> torch.Tensor:
    """
    Normalize images for CLIP input.

    Args:
        images: Tensor [B, 3, H, W] in [0, 1] range
        device: Device for mean/std tensors

    Returns:
        Normalized images
    """
    mean = torch.tensor(CLIP_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(CLIP_STD, device=device).view(1, 3, 1, 1)
    return (images - mean) / std


def prepare_images_for_clip(
    images: torch.Tensor,
    size: Tuple[int, int] = CLIP_INPUT_SIZE,
    device: str = "cuda"
) -> torch.Tensor:
    """
    Resize and normalize images for CLIP.

    Args:
        images: Images [B, 3, H, W] in [0, 1]
        size: Target size (default 224x224)
        device: Device

    Returns:
        CLIP-ready images
    """
    # Resize to CLIP input size
    if images.shape[-2:] != size:
        images = F.interpolate(images, size=size, mode='bilinear', align_corners=False)

    # Normalize
    return normalize_for_clip(images, device)


def encode_text_prompts(
    prompts: Union[str, List[str]],
    clip_model,
    clip_module,
    device: str = "cuda"
) -> torch.Tensor:
    """
    Encode text prompts to CLIP embeddings.

    Args:
        prompts: Single prompt or list of prompts
        clip_model: CLIP model instance
        clip_module: CLIP module (for tokenize function)
        device: Device

    Returns:
        Normalized text features [N, dim]
    """
    if isinstance(prompts, str):
        prompts = [prompts]

    with torch.no_grad():
        tokens = clip_module.tokenize(prompts).to(device)
        text_features = clip_model.encode_text(tokens)
        text_features /= text_features.norm(dim=-1, keepdim=True)

    return text_features


def compute_clip_similarity(
    images: torch.Tensor,
    text_features: torch.Tensor,
    clip_model,
    device: str = "cuda",
    normalize: bool = True
) -> torch.Tensor:
    """
    Compute CLIP similarity between images and text.

    Args:
        images: Images [B, 3, H, W] in [0, 1] or already normalized
        text_features: Precomputed text features [N, dim]
        clip_model: CLIP model
        device: Device
        normalize: Whether to normalize images (set False if already normalized)

    Returns:
        Similarity scores [B, N]
    """
    if normalize:
        images = prepare_images_for_clip(images, device=device)

    # Encode images
    image_features = clip_model.encode_image(images)
    image_features /= image_features.norm(dim=-1, keepdim=True)

    # Compute similarity (scaled by 100 as in CLIP paper)
    similarity = 100.0 * image_features @ text_features.T

    return similarity
