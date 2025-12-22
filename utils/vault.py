"""
Secrets Management - Environment-First with Optional Vault
===========================================================
Default: Use environment variables directly.
Optional: If VAULT_ADDR is set, try Vault first with env fallback.

Usage:
    from utils.vault import get_s3_credentials, get_api_key

    creds = get_s3_credentials()  # Uses S3_* env vars
    key = get_api_key("openai")   # Uses OPENAI_API_KEY env var
"""

import os
from dataclasses import dataclass
from typing import Optional, Any

# Optional Vault support
try:
    import hvac
    HVAC_AVAILABLE = True
except ImportError:
    HVAC_AVAILABLE = False
    hvac = None


@dataclass
class S3Credentials:
    """S3/MinIO credentials."""
    endpoint: str
    access_key: str
    secret_key: str
    region: str = "us-east-1"


def _vault_enabled() -> bool:
    """Check if Vault is configured and available."""
    return HVAC_AVAILABLE and bool(os.environ.get("VAULT_ADDR"))


def _get_vault_secret(path: str) -> Optional[dict]:
    """Get secret from Vault if enabled."""
    if not _vault_enabled():
        return None
    try:
        client = hvac.Client(
            url=os.environ.get("VAULT_ADDR"),
            token=os.environ.get("VAULT_TOKEN", ""),
        )
        if not client.is_authenticated():
            return None
        response = client.secrets.kv.v2.read_secret_version(
            path=f"mlops/{path}",
            mount_point="secret",
        )
        return response.get("data", {}).get("data", {})
    except Exception:
        return None


def get_s3_credentials() -> S3Credentials:
    """Get S3 credentials (env vars, with optional Vault fallback)."""
    vault_secret = _get_vault_secret("storage/s3") or {}
    return S3Credentials(
        endpoint=vault_secret.get("endpoint") or os.environ.get("S3_ENDPOINT", "http://minio:9000"),
        access_key=vault_secret.get("access_key") or os.environ.get("S3_ACCESS_KEY", "mlops-admin"),
        secret_key=vault_secret.get("secret_key") or os.environ.get("S3_SECRET_KEY", "mlops-dev-password"),
        region=vault_secret.get("region") or os.environ.get("S3_REGION", "us-east-1"),
    )


def get_api_key(service: str) -> Optional[str]:
    """Get API key from env var (with optional Vault fallback)."""
    vault_secret = _get_vault_secret(f"api_keys/{service}") or {}
    if vault_secret.get("api_key"):
        return vault_secret["api_key"]
    # Try common env var patterns
    for pattern in [f"{service.upper()}_API_KEY", f"{service.upper()}_KEY"]:
        if value := os.environ.get(pattern):
            return value
    return None


def get_secret(path: str, default: Any = None) -> Any:
    """Get secret from Vault (if enabled) or return default."""
    return _get_vault_secret(path) or default
