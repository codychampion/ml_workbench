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
        url=f"http://{SERVICE_HOST}:9000",
        health_endpoint="/minio/health/live",
    ),
    "minio_console": ServiceConfig(
        name="MinIO Console",
        url=f"http://{SERVICE_HOST}:9001",
        health_endpoint="/",
    ),
    "mongodb": ServiceConfig(
        name="MongoDB",
        url=f"mongodb://{SERVICE_HOST}:27017",
        health_endpoint=None,  # Uses pymongo
    ),
    "redis": ServiceConfig(
        name="Redis",
        url=f"redis://{SERVICE_HOST}:6379",
        health_endpoint=None,  # Uses redis-py
    ),
    "postgres": ServiceConfig(
        name="PostgreSQL",
        url=f"postgresql://mlops:mlops_dev_password@{SERVICE_HOST}:5432/mlops",
        health_endpoint=None,  # Uses psycopg2
    ),

    # Experiment Tracking & Orchestration
    "aim": ServiceConfig(
        name="AIM",
        url=f"http://{SERVICE_HOST}:43800",
        health_endpoint="/",
    ),
    "prefect": ServiceConfig(
        name="Prefect",
        url=f"http://{SERVICE_HOST}:4200",
        health_endpoint="/api/health",
    ),

    # Data Quality
    "great_expectations": ServiceConfig(
        name="Great Expectations",
        url=f"http://{SERVICE_HOST}:8084",
        health_endpoint="/",
    ),

    # Annotation Services
    "label_studio": ServiceConfig(
        name="Label Studio",
        url=f"http://{SERVICE_HOST}:8081",
        health_endpoint="/health",
    ),
    "cvat": ServiceConfig(
        name="CVAT",
        url=f"http://{SERVICE_HOST}:8082",
        health_endpoint="/",
    ),
    "spotlight": ServiceConfig(
        name="Spotlight",
        url=f"http://{SERVICE_HOST}:8083",
        health_endpoint="/",
    ),
    "fiftyone": ServiceConfig(
        name="FiftyOne",
        url=f"http://{SERVICE_HOST}:5151",
        health_endpoint="/",
    ),

    # Knowledge Stack
    "khoj": ServiceConfig(
        name="Khoj",
        url=f"http://{SERVICE_HOST}:42110",
        health_endpoint="/api/health",
    ),
    "couchdb": ServiceConfig(
        name="CouchDB (Obsidian)",
        url=f"http://{SERVICE_HOST}:5984",
        health_endpoint="/_up",
        auth=("obsidian", "mlops-dev-password"),
    ),
    "zotero": ServiceConfig(
        name="Zotero",
        url=f"http://{SERVICE_HOST}:8085",
        health_endpoint="/health",
    ),

    # Other Services
    "vault": ServiceConfig(
        name="HashiCorp Vault",
        url=f"http://{SERVICE_HOST}:8200",
        health_endpoint="/v1/sys/health",
    ),
    "litellm": ServiceConfig(
        name="LiteLLM",
        url=f"http://{SERVICE_HOST}:4000",
        health_endpoint="/health",
    ),
    "comfyui": ServiceConfig(
        name="ComfyUI",
        url=f"http://{SERVICE_HOST}:8188",
        health_endpoint="/",
    ),
}


@pytest.fixture
def service_host():
    """Return the service host."""
    return SERVICE_HOST


@pytest.fixture
def minio_client():
    """Create MinIO/S3 client."""
    import boto3
    from botocore.config import Config

    client = boto3.client(
        "s3",
        endpoint_url=f"http://{SERVICE_HOST}:9000",
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

    client = MongoClient(f"mongodb://{SERVICE_HOST}:27017")
    yield client
    client.close()


@pytest.fixture
def redis_client():
    """Create Redis client."""
    import redis

    client = redis.Redis(host=SERVICE_HOST, port=6379, decode_responses=True)
    yield client
    client.close()


@pytest.fixture
def postgres_connection():
    """Create PostgreSQL connection."""
    import psycopg2

    conn = psycopg2.connect(
        host=SERVICE_HOST,
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
        url=f"http://{SERVICE_HOST}:8200",
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
