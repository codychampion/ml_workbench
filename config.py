"""
MLOps Workspace Configuration
=============================
Central configuration module using environment variables with sensible defaults.
All sensitive credentials should be set via environment variables or retrieved from Vault.

NOTE: Prefer Hydra configuration (conf/) for new code. This module is for
legacy compatibility and simple scripts.

Infrastructure:
  - AIM: Experiment tracking (self-hosted, replaces W&B)
  - Vault: Secrets management
  - LiteLLM: LLM API gateway
  - MinIO/S3: Object storage (use S3Client from data_transfer/)
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class AIMConfig:
    """AIM experiment tracking configuration (replaces W&B)."""

    # Local repo path for experiments
    repo: Path = field(default_factory=lambda: Path(os.getenv("AIM_REPO", "./outputs/aim")))

    # Remote server (for distributed tracking)
    server: Optional[str] = field(default_factory=lambda: os.getenv("AIM_SERVER"))

    # Experiment metadata
    experiment: str = field(default_factory=lambda: os.getenv("AIM_EXPERIMENT", "default"))

    # UI settings
    ui_port: int = field(default_factory=lambda: int(os.getenv("AIM_UI_PORT", "43800")))
    tracking_port: int = field(default_factory=lambda: int(os.getenv("AIM_TRACKING_PORT", "53800")))

    @property
    def is_remote(self) -> bool:
        """Check if using remote AIM server."""
        return self.server is not None


@dataclass
class VaultConfig:
    """HashiCorp Vault secrets management configuration."""

    addr: str = field(default_factory=lambda: os.getenv("VAULT_ADDR", "http://vault:8200"))
    token: Optional[str] = field(default_factory=lambda: os.getenv("VAULT_TOKEN"))

    # Secret paths
    secrets_path: str = field(default_factory=lambda: os.getenv("VAULT_SECRETS_PATH", "secret/mlops"))

    @property
    def is_available(self) -> bool:
        """Check if Vault is configured."""
        return self.token is not None


@dataclass
class LiteLLMConfig:
    """LiteLLM API gateway configuration."""

    api_base: str = field(default_factory=lambda: os.getenv("LITELLM_API_BASE", "http://litellm:4000"))
    api_key: Optional[str] = field(default_factory=lambda: os.getenv("LITELLM_API_KEY", "sk-mlops-dev-key"))

    # Default model for local inference
    default_model: str = field(default_factory=lambda: os.getenv("LITELLM_DEFAULT_MODEL", "ollama/mistral"))


@dataclass
class S3Config:
    """S3-compatible storage configuration (MinIO, AWS S3, Backblaze B2)."""

    endpoint: str = field(
        default_factory=lambda: os.getenv("S3_ENDPOINT", "http://minio:9000")
    )
    access_key: Optional[str] = field(
        default_factory=lambda: os.getenv("S3_ACCESS_KEY", "mlops-admin")
    )
    secret_key: Optional[str] = field(
        default_factory=lambda: os.getenv("S3_SECRET_KEY", "mlops-dev-password")
    )
    default_bucket: str = field(
        default_factory=lambda: os.getenv("S3_DEFAULT_BUCKET", "mlops-data")
    )
    region: str = field(
        default_factory=lambda: os.getenv("S3_REGION", "us-east-1")
    )


@dataclass
class FiftyOneConfig:
    """FiftyOne data visualization configuration."""

    dataset_dir: Path = field(
        default_factory=lambda: Path(os.getenv("FIFTYONE_DATASET_DIR", "./data/fiftyone"))
    )
    port: int = field(default_factory=lambda: int(os.getenv("FIFTYONE_PORT", "5151")))
    address: str = field(default_factory=lambda: os.getenv("FIFTYONE_ADDRESS", "0.0.0.0"))
    database_uri: str = field(
        default_factory=lambda: os.getenv("FIFTYONE_DATABASE_URI", "mongodb://mongodb:27017/fiftyone")
    )


@dataclass
class ComputeConfig:
    """Compute resource configuration."""

    device: str = field(default_factory=lambda: os.getenv("COMPUTE_DEVICE", "cpu"))
    num_workers: int = field(
        default_factory=lambda: int(os.getenv("NUM_WORKERS", "2"))
    )


@dataclass
class ProjectPaths:
    """Standard project paths."""

    root: Path = field(default_factory=lambda: Path(os.getenv("PROJECT_ROOT", ".")))
    data_raw: Path = field(default_factory=lambda: Path("./data/raw"))
    data_collected: Path = field(default_factory=lambda: Path("./data/collected"))
    data_processed: Path = field(default_factory=lambda: Path("./data/processed"))
    outputs: Path = field(default_factory=lambda: Path("./outputs"))
    models: Path = field(default_factory=lambda: Path("./models"))
    logs: Path = field(default_factory=lambda: Path("./outputs/logs"))

    def ensure_dirs(self) -> None:
        """Create all project directories if they don't exist."""
        for path_attr in ["data_raw", "data_collected", "data_processed", "outputs", "models", "logs"]:
            path = getattr(self, path_attr)
            if path.exists() and not path.is_dir():
                # Rename the conflicting file so the directory can be created
                backup = path.with_suffix(path.suffix + ".bak")
                print(f"[config] Warning: {path} exists as a file. Moving to {backup} and creating directory.")
                path.rename(backup)
            path.mkdir(parents=True, exist_ok=True)


@dataclass
class Config:
    """Main configuration container."""

    # Core services
    aim: AIMConfig = field(default_factory=AIMConfig)
    vault: VaultConfig = field(default_factory=VaultConfig)
    litellm: LiteLLMConfig = field(default_factory=LiteLLMConfig)

    # Storage and data services
    s3: S3Config = field(default_factory=S3Config)
    fiftyone: FiftyOneConfig = field(default_factory=FiftyOneConfig)

    # Compute
    compute: ComputeConfig = field(default_factory=ComputeConfig)
    paths: ProjectPaths = field(default_factory=ProjectPaths)

    # Environment identification
    environment: str = field(
        default_factory=lambda: os.getenv("MLOPS_ENV", "development")
    )
    debug: bool = field(
        default_factory=lambda: os.getenv("DEBUG", "false").lower() == "true"
    )

    def __post_init__(self):
        """Ensure required directories exist on initialization."""
        self.paths.ensure_dirs()
        self.aim.repo.mkdir(parents=True, exist_ok=True)


# Global configuration instance
config = Config()


def get_config() -> Config:
    """Get the global configuration instance."""
    return config


def reload_config() -> Config:
    """Reload configuration from environment variables."""
    global config
    config = Config()
    return config


def get_secret(path: str, key: str) -> Optional[str]:
    """
    Get a secret from Vault.

    Args:
        path: Secret path in Vault (e.g., "mlops/api-keys")
        key: Key within the secret

    Returns:
        Secret value or None if not found
    """
    cfg = get_config()
    if not cfg.vault.is_available:
        return None

    try:
        import hvac
        client = hvac.Client(url=cfg.vault.addr, token=cfg.vault.token)
        secret = client.secrets.kv.v2.read_secret_version(path=path)
        return secret["data"]["data"].get(key)
    except Exception:
        return None
