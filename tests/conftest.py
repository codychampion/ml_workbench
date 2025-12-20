#!/usr/bin/env python3
"""
Pytest Configuration and Fixtures
==================================
Shared fixtures for MLOps Workbench integration tests.
"""

import os
import pytest
import requests
from typing import Optional
from dataclasses import dataclass

# Service configuration - can be overridden via environment variables
# In Docker, each service has its own hostname; locally, all use localhost
def get_host(service_env: str, default_host: str = "localhost") -> str:
    """Get host for a service, falling back to SERVICE_HOST or localhost."""
    return os.environ.get(service_env, os.environ.get("SERVICE_HOST", default_host))

# Service hosts (configurable per-service for Docker networking)
MINIO_HOST = get_host("MINIO_HOST")
MONGODB_HOST = get_host("MONGODB_HOST")
REDIS_HOST = get_host("REDIS_HOST")
POSTGRES_HOST = get_host("POSTGRES_HOST")
AIM_HOST = get_host("AIM_HOST")
PREFECT_HOST = get_host("PREFECT_HOST")
VAULT_HOST = get_host("VAULT_HOST")
LITELLM_HOST = get_host("LITELLM_HOST")
KHOJ_HOST = get_host("KHOJ_HOST")
COUCHDB_HOST = get_host("COUCHDB_HOST")
ZOTERO_HOST = get_host("ZOTERO_HOST")
GE_HOST = get_host("GE_HOST")
LABEL_STUDIO_HOST = get_host("LABEL_STUDIO_HOST")
CVAT_HOST = get_host("CVAT_HOST")
FIFTYONE_HOST = get_host("FIFTYONE_HOST")
SPOTLIGHT_HOST = get_host("SPOTLIGHT_HOST")
COMFYUI_HOST = get_host("COMFYUI_HOST")

# Legacy: single SERVICE_HOST for backward compatibility
SERVICE_HOST = os.environ.get("SERVICE_HOST", "localhost")


@dataclass
class ServiceConfig:
    """Configuration for a service endpoint."""
    name: str
    url: str
    health_endpoint: str = "/health"
    expected_status: int = 200
    timeout: int = 10
    auth: Optional[tuple] = None
    headers: Optional[dict] = None


# All services to test
SERVICES = {
    # Core Infrastructure
    "minio": ServiceConfig(
        name="MinIO",
        url=f"http://{MINIO_HOST}:9000",
        health_endpoint="/minio/health/live",
    ),
    "minio_console": ServiceConfig(
        name="MinIO Console",
        url=f"http://{MINIO_HOST}:9001",
        health_endpoint="/",
    ),
    "mongodb": ServiceConfig(
        name="MongoDB",
        url=f"mongodb://{MONGODB_HOST}:27017",
        health_endpoint=None,  # Uses pymongo
    ),
    "redis": ServiceConfig(
        name="Redis",
        url=f"redis://{REDIS_HOST}:6379",
        health_endpoint=None,  # Uses redis-py
    ),
    "postgres": ServiceConfig(
        name="PostgreSQL",
        url=f"postgresql://mlops:mlops_dev_password@{POSTGRES_HOST}:5432/mlops",
        health_endpoint=None,  # Uses psycopg2
    ),

    # Experiment Tracking & Orchestration
    "aim": ServiceConfig(
        name="AIM",
        url=f"http://{AIM_HOST}:43800",
        health_endpoint="/",
    ),
    "prefect": ServiceConfig(
        name="Prefect",
        url=f"http://{PREFECT_HOST}:4200",
        health_endpoint="/api/health",
    ),

    # Data Quality
    "great_expectations": ServiceConfig(
        name="Great Expectations",
        url=f"http://{GE_HOST}:8084",
        health_endpoint="/",
    ),

    # Annotation Services
    "label_studio": ServiceConfig(
        name="Label Studio",
        url=f"http://{LABEL_STUDIO_HOST}:8080",
        health_endpoint="/health",
    ),
    "cvat": ServiceConfig(
        name="CVAT",
        url=f"http://{CVAT_HOST}:8080",
        health_endpoint="/",
    ),
    "spotlight": ServiceConfig(
        name="Spotlight",
        url=f"http://{SPOTLIGHT_HOST}:8000",
        health_endpoint="/",
    ),
    "fiftyone": ServiceConfig(
        name="FiftyOne",
        url=f"http://{FIFTYONE_HOST}:5151",
        health_endpoint="/",
    ),

    # Knowledge Stack
    "khoj": ServiceConfig(
        name="Khoj",
        url=f"http://{KHOJ_HOST}:42110",
        health_endpoint="/api/health",
    ),
    "couchdb": ServiceConfig(
        name="CouchDB (Obsidian)",
        url=f"http://{COUCHDB_HOST}:5984",
        health_endpoint="/_up",
        auth=("obsidian", "mlops-dev-password"),
    ),
    "zotero": ServiceConfig(
        name="Zotero",
        url=f"http://{ZOTERO_HOST}:8085",
        health_endpoint="/health",
    ),

    # Other Services
    "vault": ServiceConfig(
        name="HashiCorp Vault",
        url=f"http://{VAULT_HOST}:8200",
        health_endpoint="/v1/sys/health",
    ),
    "litellm": ServiceConfig(
        name="LiteLLM",
        url=f"http://{LITELLM_HOST}:4000",
        health_endpoint="/health",
    ),
    "comfyui": ServiceConfig(
        name="ComfyUI",
        url=f"http://{COMFYUI_HOST}:8188",
        health_endpoint="/",
    ),
}


@pytest.fixture
def service_host():
    """Return the service host (legacy, use specific hosts instead)."""
    return SERVICE_HOST


@pytest.fixture
def minio_client():
    """Create MinIO/S3 client."""
    import boto3
    from botocore.config import Config

    client = boto3.client(
        "s3",
        endpoint_url=f"http://{MINIO_HOST}:9000",
        aws_access_key_id="mlops-admin",
        aws_secret_access_key="mlops-dev-password",
        config=Config(signature_version="s3v4"),
        region_name="us-east-1",
    )
    return client


@pytest.fixture
def mongodb_client():
    """Create MongoDB client."""
    from pymongo import MongoClient

    client = MongoClient(f"mongodb://{MONGODB_HOST}:27017")
    yield client
    client.close()


@pytest.fixture
def redis_client():
    """Create Redis client."""
    import redis

    client = redis.Redis(host=REDIS_HOST, port=6379, decode_responses=True)
    yield client
    client.close()


@pytest.fixture
def postgres_connection():
    """Create PostgreSQL connection."""
    import psycopg2

    conn = psycopg2.connect(
        host=POSTGRES_HOST,
        port=5432,
        user="mlops",
        password="mlops_dev_password",
        database="postgres",
    )
    yield conn
    conn.close()


@pytest.fixture
def vault_client():
    """Create Vault client."""
    import hvac

    client = hvac.Client(
        url=f"http://{VAULT_HOST}:8200",
        token="mlops-dev-token",
    )
    return client


@pytest.fixture
def http_session():
    """Create requests session with default timeout."""
    session = requests.Session()
    session.timeout = 10
    return session


def check_http_health(config: ServiceConfig) -> tuple[bool, str]:
    """Check HTTP service health."""
    try:
        url = f"{config.url}{config.health_endpoint}"
        kwargs = {"timeout": config.timeout}

        if config.auth:
            kwargs["auth"] = config.auth
        if config.headers:
            kwargs["headers"] = config.headers

        response = requests.get(url, **kwargs)
        if response.status_code == config.expected_status:
            return True, f"OK ({response.status_code})"
        else:
            return False, f"Unexpected status: {response.status_code}"
    except requests.exceptions.ConnectionError:
        return False, "Connection refused"
    except requests.exceptions.Timeout:
        return False, "Timeout"
    except Exception as e:
        return False, str(e)


# Pytest configuration
def pytest_configure(config):
    """Add custom markers."""
    config.addinivalue_line("markers", "slow: marks tests as slow")
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "requires_gpu: marks tests that require GPU")
