"""
HashiCorp Vault Integration
===========================
Centralized secrets management using HashiCorp Vault.

All secrets should be stored in Vault and retrieved at runtime.
Never hardcode credentials in code or config files.

Environment Variables:
    VAULT_ADDR: Vault server address (default: http://vault:8200)
    VAULT_TOKEN: Vault authentication token
    VAULT_SECRETS_PATH: Base path for secrets (default: secret/mlops)

Usage:
    from utils.vault import get_secret, get_s3_credentials, VaultClient

    # Get a single secret
    api_key = get_secret("openai/api_key")

    # Get S3 credentials (auto-loads from Vault)
    creds = get_s3_credentials()

    # Use client directly
    vault = VaultClient()
    vault.put_secret("myapp/config", {"key": "value"})
    config = vault.get_secret("myapp/config")
"""

import os
from dataclasses import dataclass
from typing import Dict, Optional, Any
from functools import lru_cache

# hvac is the official Vault Python client
try:
    import hvac
    HVAC_AVAILABLE = True
except ImportError:
    HVAC_AVAILABLE = False
    hvac = None


@dataclass
class VaultConfig:
    """Configuration for Vault connection."""
    addr: str
    token: str
    secrets_path: str = "secret/mlops"
    mount_point: str = "secret"  # KV v2 mount point

    @classmethod
    def from_env(cls) -> "VaultConfig":
        """Create config from environment variables."""
        return cls(
            addr=os.environ.get("VAULT_ADDR", "http://vault:8200"),
            token=os.environ.get("VAULT_TOKEN", "mlops-dev-token"),
            secrets_path=os.environ.get("VAULT_SECRETS_PATH", "secret/mlops"),
            mount_point=os.environ.get("VAULT_MOUNT_POINT", "secret"),
        )


class VaultClient:
    """
    HashiCorp Vault client for secrets management.

    Wraps hvac client with convenience methods for common operations.
    """

    def __init__(self, config: Optional[VaultConfig] = None):
        """
        Initialize Vault client.

        Args:
            config: Vault configuration. If None, loads from environment.
        """
        if not HVAC_AVAILABLE:
            raise ImportError(
                "hvac is required for Vault integration. "
                "Install with: pip install hvac"
            )

        self.config = config or VaultConfig.from_env()
        self._client = hvac.Client(
            url=self.config.addr,
            token=self.config.token,
        )

        # Verify connection
        if not self._client.is_authenticated():
            raise ConnectionError(
                f"Failed to authenticate with Vault at {self.config.addr}"
            )

    @property
    def is_connected(self) -> bool:
        """Check if connected and authenticated."""
        try:
            return self._client.is_authenticated()
        except Exception:
            return False

    def _full_path(self, path: str) -> str:
        """Get full secret path including base path."""
        base = self.config.secrets_path.rstrip("/")
        path = path.lstrip("/")
        # Remove mount point prefix if present
        if base.startswith(f"{self.config.mount_point}/"):
            base = base[len(self.config.mount_point) + 1:]
        return f"{base}/{path}" if base else path

    def get_secret(self, path: str) -> Optional[Dict[str, Any]]:
        """
        Get a secret from Vault.

        Args:
            path: Secret path relative to base secrets path.

        Returns:
            Secret data as dictionary or None if not found.
        """
        full_path = self._full_path(path)
        try:
            response = self._client.secrets.kv.v2.read_secret_version(
                path=full_path,
                mount_point=self.config.mount_point,
            )
            return response.get("data", {}).get("data", {})
        except hvac.exceptions.InvalidPath:
            print(f"[Vault] Secret not found: {full_path}")
            return None
        except Exception as e:
            print(f"[Vault] Error reading secret {full_path}: {e}")
            return None

    def put_secret(self, path: str, data: Dict[str, Any]) -> bool:
        """
        Store a secret in Vault.

        Args:
            path: Secret path relative to base secrets path.
            data: Secret data as dictionary.

        Returns:
            True on success.
        """
        full_path = self._full_path(path)
        try:
            self._client.secrets.kv.v2.create_or_update_secret(
                path=full_path,
                secret=data,
                mount_point=self.config.mount_point,
            )
            print(f"[Vault] Stored secret: {full_path}")
            return True
        except Exception as e:
            print(f"[Vault] Error storing secret {full_path}: {e}")
            return False

    def delete_secret(self, path: str) -> bool:
        """
        Delete a secret from Vault.

        Args:
            path: Secret path relative to base secrets path.

        Returns:
            True on success.
        """
        full_path = self._full_path(path)
        try:
            self._client.secrets.kv.v2.delete_metadata_and_all_versions(
                path=full_path,
                mount_point=self.config.mount_point,
            )
            print(f"[Vault] Deleted secret: {full_path}")
            return True
        except Exception as e:
            print(f"[Vault] Error deleting secret {full_path}: {e}")
            return False

    def list_secrets(self, path: str = "") -> list:
        """
        List secrets at a path.

        Args:
            path: Path to list (relative to base secrets path).

        Returns:
            List of secret names/paths.
        """
        full_path = self._full_path(path)
        try:
            response = self._client.secrets.kv.v2.list_secrets(
                path=full_path,
                mount_point=self.config.mount_point,
            )
            return response.get("data", {}).get("keys", [])
        except Exception as e:
            print(f"[Vault] Error listing secrets at {full_path}: {e}")
            return []


# Singleton instance for convenience functions
_vault_client: Optional[VaultClient] = None


def _get_client() -> Optional[VaultClient]:
    """Get or create singleton Vault client."""
    global _vault_client
    if _vault_client is None:
        try:
            _vault_client = VaultClient()
        except (ImportError, ConnectionError) as e:
            print(f"[Vault] Client unavailable: {e}")
            return None
    return _vault_client


def get_secret(path: str, default: Any = None) -> Any:
    """
    Get a secret value from Vault.

    Args:
        path: Secret path (e.g., "s3/credentials", "openai/api_key")
        default: Default value if secret not found

    Returns:
        Secret data dictionary or default value
    """
    client = _get_client()
    if client is None:
        return default

    secret = client.get_secret(path)
    return secret if secret is not None else default


def get_secret_value(path: str, key: str, default: Any = None) -> Any:
    """
    Get a specific key from a secret.

    Args:
        path: Secret path
        key: Key within the secret
        default: Default value if not found

    Returns:
        Secret value or default
    """
    secret = get_secret(path, {})
    return secret.get(key, default)


# =============================================================================
# Convenience functions for common secrets
# =============================================================================

@dataclass
class S3Credentials:
    """S3/MinIO credentials."""
    endpoint: str
    access_key: str
    secret_key: str
    region: str = "us-east-1"


def get_s3_credentials(
    path: str = "storage/s3",
    fallback_to_env: bool = True
) -> S3Credentials:
    """
    Get S3 credentials from Vault with env var fallback.

    Args:
        path: Vault path for S3 credentials
        fallback_to_env: If True, fall back to environment variables

    Returns:
        S3Credentials object
    """
    # Try Vault first
    secret = get_secret(path, {})

    if secret:
        return S3Credentials(
            endpoint=secret.get("endpoint", os.environ.get("S3_ENDPOINT", "http://minio:9000")),
            access_key=secret.get("access_key", os.environ.get("S3_ACCESS_KEY", "")),
            secret_key=secret.get("secret_key", os.environ.get("S3_SECRET_KEY", "")),
            region=secret.get("region", os.environ.get("S3_REGION", "us-east-1")),
        )

    # Fall back to environment variables
    if fallback_to_env:
        return S3Credentials(
            endpoint=os.environ.get("S3_ENDPOINT", "http://minio:9000"),
            access_key=os.environ.get("S3_ACCESS_KEY", "mlops-admin"),
            secret_key=os.environ.get("S3_SECRET_KEY", "mlops-dev-password"),
            region=os.environ.get("S3_REGION", "us-east-1"),
        )

    raise ValueError("S3 credentials not found in Vault and fallback disabled")


def get_api_key(service: str, fallback_env_var: Optional[str] = None) -> Optional[str]:
    """
    Get an API key from Vault.

    Args:
        service: Service name (e.g., "openai", "anthropic", "huggingface")
        fallback_env_var: Environment variable to check if Vault fails

    Returns:
        API key string or None
    """
    # Try Vault
    secret = get_secret(f"api_keys/{service}", {})
    if secret:
        return secret.get("api_key") or secret.get("key")

    # Fall back to env var
    if fallback_env_var:
        return os.environ.get(fallback_env_var)

    # Try common env var patterns
    env_patterns = [
        f"{service.upper()}_API_KEY",
        f"{service.upper()}_KEY",
    ]
    for pattern in env_patterns:
        value = os.environ.get(pattern)
        if value:
            return value

    return None


def get_database_url(
    name: str = "default",
    fallback_env_var: str = "DATABASE_URL"
) -> Optional[str]:
    """
    Get database connection URL from Vault.

    Args:
        name: Database name/identifier
        fallback_env_var: Environment variable fallback

    Returns:
        Database URL string or None
    """
    secret = get_secret(f"databases/{name}", {})
    if secret:
        return secret.get("url") or secret.get("connection_string")

    return os.environ.get(fallback_env_var)


# =============================================================================
# Vault initialization helper
# =============================================================================

def init_vault_secrets():
    """
    Initialize Vault with default secrets structure.

    Run this once to set up the secrets hierarchy.
    """
    client = _get_client()
    if client is None:
        print("[Vault] Cannot initialize - client unavailable")
        return False

    # Default secrets structure
    default_secrets = {
        "storage/s3": {
            "endpoint": "http://minio:9000",
            "access_key": "mlops-admin",
            "secret_key": "mlops-dev-password",
            "region": "us-east-1",
            "_comment": "Update these for production"
        },
        "storage/b2": {
            "endpoint": "https://s3.us-west-000.backblazeb2.com",
            "key_id": "",
            "app_key": "",
            "bucket": "",
            "_comment": "Backblaze B2 credentials"
        },
        "api_keys/openai": {
            "api_key": "",
            "_comment": "OpenAI API key"
        },
        "api_keys/anthropic": {
            "api_key": "",
            "_comment": "Anthropic API key"
        },
        "api_keys/huggingface": {
            "api_key": "",
            "_comment": "HuggingFace token"
        },
        "databases/mongodb": {
            "url": "mongodb://mongodb:27017/mlops",
            "_comment": "MongoDB connection URL"
        },
        "databases/redis": {
            "url": "redis://redis:6379/0",
            "_comment": "Redis connection URL"
        },
    }

    print("[Vault] Initializing default secrets structure...")
    for path, data in default_secrets.items():
        # Only create if doesn't exist
        existing = client.get_secret(path)
        if existing is None:
            client.put_secret(path, data)
            print(f"[Vault] Created: {path}")
        else:
            print(f"[Vault] Exists: {path}")

    print("[Vault] Initialization complete")
    return True
