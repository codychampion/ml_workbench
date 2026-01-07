"""Shared constants used across pipelines."""

# Image and video file extensions
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp"}
VIDEO_EXTENSIONS = {".mp4", ".webm", ".mov", ".avi", ".mkv"}
MEDIA_EXTENSIONS = IMAGE_EXTENSIONS | VIDEO_EXTENSIONS

# Common CLIP normalization constants
CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]
CLIP_INPUT_SIZE = (224, 224)

# Default model paths
DEFAULT_BLIP_MODEL = "Salesforce/blip-image-captioning-base"
DEFAULT_CLIP_MODEL = "ViT-B/32"
DEFAULT_YOLO_MODEL = "yolov8n.pt"
