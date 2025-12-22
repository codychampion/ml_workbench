"""Pytest Configuration - Lean Test Fixtures."""

import os
import pytest
import requests
from dataclasses import dataclass
from typing import Optional


def get_host(env_var: str, default: str = "localhost") -> str:
    return os.environ.get(env_var, os.environ.get("SERVICE_HOST", default))


MINIO_HOST = get_host("MINIO_HOST")
MONGODB_HOST = get_host("MONGODB_HOST")
POSTGRES_HOST = get_host("POSTGRES_HOST")
AIM_HOST = get_host("AIM_HOST")
KHOJ_HOST = get_host("KHOJ_HOST")
LABEL_STUDIO_HOST = get_host("LABEL_STUDIO_HOST")
FIFTYONE_HOST = get_host("FIFTYONE_HOST")
SERVICE_HOST = os.environ.get("SERVICE_HOST", "localhost")


@dataclass
class ServiceConfig:
    name: str
    url: str
    health_endpoint: str = "/health"
    expected_status: int = 200
    timeout: int = 10
    auth: Optional[tuple] = None


SERVICES = {
    "minio": ServiceConfig("MinIO", f"http://{MINIO_HOST}:9000", "/minio/health/live"),
    "minio_console": ServiceConfig("MinIO Console", f"http://{MINIO_HOST}:9001", "/"),
    "mongodb": ServiceConfig("MongoDB", f"mongodb://{MONGODB_HOST}:27017", None),
    "postgres": ServiceConfig("PostgreSQL", f"postgresql://mlops:mlops_dev_password@{POSTGRES_HOST}:5432/mlops", None),
    "aim": ServiceConfig("AIM", f"http://{AIM_HOST}:43800", "/"),
    "label_studio": ServiceConfig("Label Studio", f"http://{LABEL_STUDIO_HOST}:8080", "/health"),
    "khoj": ServiceConfig("Khoj", f"http://{KHOJ_HOST}:42110", "/api/health"),
    "fiftyone": ServiceConfig("FiftyOne", f"http://{FIFTYONE_HOST}:5151", "/"),
}


@pytest.fixture
def minio_client():
    import boto3
    from botocore.config import Config
    return boto3.client("s3", endpoint_url=f"http://{MINIO_HOST}:9000",
                       aws_access_key_id="mlops-admin", aws_secret_access_key="mlops-dev-password",
                       config=Config(signature_version="s3v4"), region_name="us-east-1")


@pytest.fixture
def mongodb_client():
    from pymongo import MongoClient
    client = MongoClient(f"mongodb://{MONGODB_HOST}:27017")
    yield client
    client.close()


@pytest.fixture
def postgres_connection():
    import psycopg2
    conn = psycopg2.connect(host=POSTGRES_HOST, port=5432, user="mlops", password="mlops_dev_password", database="postgres")
    yield conn
    conn.close()


def check_http_health(config: ServiceConfig) -> tuple:
    try:
        url = f"{config.url}{config.health_endpoint}"
        response = requests.get(url, timeout=config.timeout, auth=config.auth)
        return (response.status_code == config.expected_status, f"OK ({response.status_code})")
    except requests.exceptions.ConnectionError:
        return (False, "Connection refused")
    except Exception as e:
        return (False, str(e))


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: marks tests as slow")
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
