"""
Target Model Wrappers for Adversarial Attacks
==============================================
Unified interface for different victim models (YOLO, Faster R-CNN, CLIP).
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path


class TargetModel(ABC):
    """Abstract base class for target models."""

    def __init__(self, device: str = "cuda"):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.model = None

    @abstractmethod
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Forward pass through model."""
        pass

    @abstractmethod
    def compute_loss(self, images: torch.Tensor, attack_objective: str) -> torch.Tensor:
        """Compute adversarial loss for given objective."""
        pass

    def preprocess(self, images: torch.Tensor) -> torch.Tensor:
        """Preprocess images for model input."""
        return images

    def to(self, device: str):
        """Move model to device."""
        self.device = device
        if self.model is not None:
            self.model = self.model.to(device)
        return self


class YOLOv8Wrapper(TargetModel):
    """Wrapper for YOLOv8 person detector."""

    def __init__(self, model_path: str = "yolov8n.pt", device: str = "cuda"):
        super().__init__(device)
        try:
            from ultralytics import YOLO
            self.model = YOLO(model_path)
            self.model.to(self.device)
        except ImportError:
            raise ImportError(
                "ultralytics not installed. Install with: pip install ultralytics"
            )

        self.person_class = 0  # COCO person class ID
        self.conf_threshold = 0.25

    def forward(self, images: torch.Tensor) -> List[Dict]:
        """Run YOLO detection."""
        # YOLOv8 expects images in [0, 255] range
        images_uint8 = (images * 255).byte()
        results = self.model(images_uint8, verbose=False)
        return results

    def compute_loss(self, images: torch.Tensor, attack_objective: str = "evasion") -> torch.Tensor:
        """
        Compute adversarial loss for YOLO.

        Args:
            images: Batch of images [B, 3, H, W] in [0, 1] range
            attack_objective: 'evasion' to suppress person detection

        Returns:
            Loss tensor (higher = more detections = worse for attack)
        """
        results = self.forward(images)

        if attack_objective == "evasion":
            # Goal: Minimize person detections
            # Loss = sum of person detection confidences
            total_conf = 0.0
            for result in results:
                boxes = result.boxes
                if boxes is not None and len(boxes) > 0:
                    # Filter for person class
                    person_mask = boxes.cls == self.person_class
                    if person_mask.any():
                        person_confs = boxes.conf[person_mask]
                        total_conf += person_confs.sum()

            # Return as tensor for backprop
            return torch.tensor(total_conf, device=self.device, requires_grad=True)

        else:
            raise ValueError(f"Unknown attack objective: {attack_objective}")


class CLIPClassifierWrapper(TargetModel):
    """Wrapper for CLIP-based image classifier."""

    def __init__(
        self,
        model_name: str = "ViT-B/32",
        target_classes: List[str] = None,
        device: str = "cuda"
    ):
        super().__init__(device)
        try:
            import clip
            self.model, self.preprocess_fn = clip.load(model_name, device=device)
            self.model.eval()
        except ImportError:
            raise ImportError(
                "CLIP not installed. Install with: pip install git+https://github.com/openai/CLIP.git"
            )

        # Default target classes for age classification
        self.target_classes = target_classes or [
            "a photo of a young person",
            "a photo of a middle-aged person",
            "a photo of an elderly person"
        ]

        # Encode text features once
        with torch.no_grad():
            text_tokens = clip.tokenize(self.target_classes).to(device)
            self.text_features = self.model.encode_text(text_tokens)
            self.text_features /= self.text_features.norm(dim=-1, keepdim=True)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Encode images with CLIP."""
        # CLIP expects images normalized with specific mean/std
        # Assume images are already preprocessed
        image_features = self.model.encode_image(images)
        image_features /= image_features.norm(dim=-1, keepdim=True)

        # Compute similarity to text classes
        logits = 100.0 * image_features @ self.text_features.T
        return logits

    def compute_loss(
        self,
        images: torch.Tensor,
        attack_objective: str = "misclassification",
        target_class: int = None
    ) -> torch.Tensor:
        """
        Compute adversarial loss for CLIP classifier.

        Args:
            images: Batch of images [B, 3, H, W]
            attack_objective: 'misclassification' or 'targeted'
            target_class: For targeted attacks, the desired class

        Returns:
            Loss tensor
        """
        logits = self.forward(images)
        probs = torch.softmax(logits, dim=-1)

        if attack_objective == "misclassification":
            # Maximize entropy (make predictions uncertain)
            log_probs = torch.log_softmax(logits, dim=-1)
            entropy = -(probs * log_probs).sum(dim=-1).mean()
            return -entropy  # Negative because we want to maximize

        elif attack_objective == "targeted" and target_class is not None:
            # Maximize probability of target class
            target_probs = probs[:, target_class]
            return -target_probs.mean()  # Negative to maximize

        else:
            raise ValueError(f"Unknown attack objective: {attack_objective}")


def get_target_model(model_type: str, **kwargs) -> TargetModel:
    """
    Factory function to create target model wrappers.

    Args:
        model_type: Type of model ('yolov8', 'clip-classifier')
        **kwargs: Model-specific arguments

    Returns:
        TargetModel instance
    """
    if model_type == "yolov8":
        return YOLOv8Wrapper(**kwargs)
    elif model_type == "clip-classifier":
        return CLIPClassifierWrapper(**kwargs)
    else:
        raise ValueError(
            f"Unknown model type: {model_type}. "
            f"Supported: ['yolov8', 'clip-classifier']"
        )
