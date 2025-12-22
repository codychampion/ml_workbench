"""
Storage Utilities - S3-Compatible Storage Helpers
==================================================
Simple helpers for S3 storage from env vars or Hydra config.

Usage:
    from utils.storage import get_s3_client
    s3 = get_s3_client()
    s3.upload_file(Path("model.pt"), "models/v1/model.pt")
"""

import os
from pathlib import Path
from typing import Optional
from omegaconf import DictConfig, OmegaConf

try:
    from data_transfer.s3_client import S3Client, S3Config
    S3_AVAILABLE = True
except ImportError:
    S3_AVAILABLE = False
    S3Client = S3Config = None


def get_s3_client(bucket: str = None) -> Optional["S3Client"]:
    """Get S3 client from environment variables."""
    if not S3_AVAILABLE:
        return None
    return S3Client(S3Config(
        endpoint=os.environ.get("S3_ENDPOINT", "http://minio:9000"),
        access_key=os.environ.get("S3_ACCESS_KEY", "mlops-admin"),
        secret_key=os.environ.get("S3_SECRET_KEY", "mlops-dev-password"),
        bucket=bucket or os.environ.get("S3_BUCKET", "mlops-data"),
        region=os.environ.get("S3_REGION", "us-east-1"),
    ))


def init_storage_from_hydra(cfg: DictConfig, bucket_type: str = "data") -> Optional["S3Client"]:
    """Initialize S3 client from Hydra config."""
    if not S3_AVAILABLE:
        return None
    storage_cfg = cfg.get("storage", cfg.get("infrastructure", {}).get("storage", {}))
    storage_dict = OmegaConf.to_container(storage_cfg, resolve=True) if storage_cfg else {}
    bucket = storage_dict.get("buckets", {}).get(bucket_type, f"mlops-{bucket_type}")
    return get_s3_client(bucket)


def upload_model(cfg: DictConfig, model_path: Path, model_name: str, version: str = "latest") -> bool:
    """Upload model to S3."""
    client = init_storage_from_hydra(cfg, bucket_type="models")
    return client.upload_file(model_path, f"models/{model_name}/{version}/{model_path.name}") if client else False


def download_model(cfg: DictConfig, model_name: str, version: str = "latest", dest: Path = None) -> Optional[Path]:
    """Download model from S3."""
    client = init_storage_from_hydra(cfg, bucket_type="models")
    if not client:
        return None
    objects = client.list_objects(f"models/{model_name}/{version}/")
    if not objects:
        return None
    dest = dest or Path(f"./models/{model_name}/{version}/{objects[0].name}")
    return dest if client.download_file(objects[0].key, dest) else None


def sync_dataset(cfg: DictConfig, local_dir: Path, remote_prefix: str, direction: str = "upload") -> int:
    """Sync dataset to/from S3."""
    client = init_storage_from_hydra(cfg, bucket_type="data")
    if not client:
        return 0
    return client.sync_directory(local_dir, remote_prefix) if direction == "upload" else client.download_directory(remote_prefix, local_dir)
