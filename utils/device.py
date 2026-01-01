"""Device detection and management utilities."""

import torch


def get_device(device: str = "auto") -> str:
    """
    Resolve device string to actual device.

    Args:
        device: "auto", "cuda", "mps", or "cpu"

    Returns:
        Actual device string (cuda/mps/cpu)
    """
    if device == "auto":
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    return device


def print_device_info(device: str, prefix: str = ""):
    """Print device information with optional prefix."""
    prefix_str = f"[{prefix}] " if prefix else ""
    print(f"{prefix_str}Using device: {device}")

    if device == "cuda":
        print(f"{prefix_str}GPU: {torch.cuda.get_device_name(0)}")
        print(f"{prefix_str}CUDA version: {torch.version.cuda}")
