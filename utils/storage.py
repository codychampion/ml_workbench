"""
Storage Utilities
=================
Helper functions for initializing S3-compatible storage.

Credential Priority:
    1. HashiCorp Vault (if available)
    2. Hydra configuration
    3. Environment variables

Usage:
    from utils.storage import init_storage_from_hydra

    @hydra.main(config_path="../conf", config_name="config")
    def main(cfg: DictConfig):
        s3 = init_storage_from_hydra(cfg)
        s3.upload_file(Path("model.pt"), "models/latest/model.pt")
"""

import os
from pathlib import Path
from typing import Optional

from omegaconf import DictConfig, OmegaConf

# Import S3 client
try:
    from data_transfer.s3_client import S3Client, S3Config
    S3_AVAILABLE = True
except ImportError:
    S3_AVAILABLE = False
    S3Client = None
    S3Config = None

# Import Vault client
try:
    from .vault import get_s3_credentials, S3Credentials
    VAULT_AVAILABLE = True
except ImportError:
    VAULT_AVAILABLE = False
    get_s3_credentials = None
    S3Credentials = None


def _get_credentials_from_vault() -> Optional[dict]:
    """Try to get S3 credentials from Vault."""
    if not VAULT_AVAILABLE:
        return None

    try:
        creds = get_s3_credentials(fallback_to_env=False)
        return {
            "endpoint": creds.endpoint,
            "access_key": creds.access_key,
            "secret_key": creds.secret_key,
            "region": creds.region,
        }
    except (ValueError, Exception):
        return None


def init_storage_from_hydra(
    cfg: DictConfig,
    bucket_type: str = "data",
    use_vault: bool = True
) -> Optional["S3Client"]:
    """
    Initialize S3 client with credentials from Vault, Hydra, or environment.

    Credential priority:
        1. Vault (if use_vault=True and available)
        2. Hydra configuration
        3. Environment variables

    Args:
        cfg: Hydra DictConfig object
        bucket_type: Which bucket to use - "data", "models", or "outputs"
        use_vault: Whether to try Vault first for credentials

    Returns:
        S3Client instance or None if not available
    """
    if not S3_AVAILABLE:
        print("[Storage] Warning: boto3 not installed, S3 storage disabled")
        print("[Storage] Install with: pip install boto3")
        return None

    # Try Vault first
    vault_creds = None
    if use_vault:
        vault_creds = _get_credentials_from_vault()
        if vault_creds:
            print("[Storage] Using credentials from Vault")

    # Get storage config from Hydra
    storage_cfg = cfg.get("storage", cfg.get("infrastructure", {}).get("storage", {}))
    storage_dict = OmegaConf.to_container(storage_cfg, resolve=True) if storage_cfg else {}

    # Determine credentials (Vault > Hydra > Env)
    if vault_creds:
        endpoint = vault_creds["endpoint"]
        access_key = vault_creds["access_key"]
        secret_key = vault_creds["secret_key"]
        region = vault_creds["region"]
    else:
        endpoint = storage_dict.get("endpoint", os.environ.get("S3_ENDPOINT", "http://minio:9000"))
        access_key = storage_dict.get("access_key", os.environ.get("S3_ACCESS_KEY", "mlops-admin"))
        secret_key = storage_dict.get("secret_key", os.environ.get("S3_SECRET_KEY", "mlops-dev-password"))
        region = storage_dict.get("region", os.environ.get("S3_REGION", "us-east-1"))

    # Get bucket name based on type
    buckets = storage_dict.get("buckets", {})
    bucket = buckets.get(bucket_type, f"mlops-{bucket_type}")

    # Create config
    config = S3Config(
        endpoint=endpoint,
        access_key=access_key,
        secret_key=secret_key,
        bucket=bucket,
        region=region,
        secure=endpoint.startswith("https"),
        path_style=True,  # MinIO requires path-style
    )

    # Create client
    try:
        client = S3Client(config)
        print(f"[Storage] Initialized S3 client")
        print(f"[Storage] Endpoint: {endpoint}")
        print(f"[Storage] Bucket: {bucket}")
        return client
    except Exception as e:
        print(f"[Storage] Error initializing S3 client: {e}")
        return None


def upload_model(
    cfg: DictConfig,
    model_path: Path,
    model_name: str,
    version: str = "latest"
) -> bool:
    """
    Upload a model to S3 storage.

    Args:
        cfg: Hydra config
        model_path: Local path to model file
        model_name: Name for the model
        version: Version string

    Returns:
        True on success
    """
    client = init_storage_from_hydra(cfg, bucket_type="models")
    if not client:
        return False

    key = f"models/{model_name}/{version}/{model_path.name}"
    return client.upload_file(model_path, key)


def download_model(
    cfg: DictConfig,
    model_name: str,
    version: str = "latest",
    destination: Optional[Path] = None
) -> Optional[Path]:
    """
    Download a model from S3 storage.

    Args:
        cfg: Hydra config
        model_name: Name of the model
        version: Version string
        destination: Local destination path

    Returns:
        Path to downloaded model or None
    """
    client = init_storage_from_hydra(cfg, bucket_type="models")
    if not client:
        return None

    # List objects to find the model file
    prefix = f"models/{model_name}/{version}/"
    objects = client.list_objects(prefix)

    if not objects:
        print(f"[Storage] Model not found: {model_name}/{version}")
        return None

    # Download first matching object
    obj = objects[0]
    if destination is None:
        destination = Path(f"./models/{model_name}/{version}/{obj.name}")

    if client.download_file(obj.key, destination):
        return destination
    return None


def sync_dataset(
    cfg: DictConfig,
    local_dir: Path,
    remote_prefix: str,
    direction: str = "upload"
) -> int:
    """
    Sync dataset between local and S3.

    Args:
        cfg: Hydra config
        local_dir: Local directory
        remote_prefix: S3 prefix
        direction: "upload" or "download"

    Returns:
        Number of files synced
    """
    client = init_storage_from_hydra(cfg, bucket_type="data")
    if not client:
        return 0

    if direction == "upload":
        return client.sync_directory(local_dir, remote_prefix)
    else:
        return client.download_directory(remote_prefix, local_dir)
