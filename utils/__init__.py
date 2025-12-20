"""MLOps Workbench Utilities."""

from .hydra_aim import init_aim_from_hydra, log_hydra_config, AimCallback
from .storage import (
    init_storage_from_hydra,
    upload_model,
    download_model,
    sync_dataset,
)
from .vault import (
    VaultClient,
    get_secret,
    get_secret_value,
    get_s3_credentials,
    get_api_key,
    get_database_url,
    init_vault_secrets,
)
from .model_registry import (
    ModelRegistry,
    ModelVersion,
    ModelInfo,
    get_registry,
    register_model,
    load_model,
    list_models,
)

__all__ = [
    # AIM integration
    "init_aim_from_hydra",
    "log_hydra_config",
    "AimCallback",
    # Storage utilities
    "init_storage_from_hydra",
    "upload_model",
    "download_model",
    "sync_dataset",
    # Vault secrets
    "VaultClient",
    "get_secret",
    "get_secret_value",
    "get_s3_credentials",
    "get_api_key",
    "get_database_url",
    "init_vault_secrets",
    # Model registry
    "ModelRegistry",
    "ModelVersion",
    "ModelInfo",
    "get_registry",
    "register_model",
    "load_model",
    "list_models",
]
