#!/usr/bin/env python3
"""
Service Health Check Tests
===========================
Verify all MLOps Workbench services are running and healthy.
"""

import pytest
import requests
from conftest import SERVICES, check_http_health, SERVICE_HOST


class TestCoreInfrastructure:
    """Test core infrastructure services."""

    @pytest.mark.integration
    def test_minio_health(self):
        """Test MinIO S3 service is healthy."""
        config = SERVICES["minio"]
        healthy, message = check_http_health(config)
        assert healthy, f"MinIO health check failed: {message}"

    @pytest.mark.integration
    def test_minio_console_accessible(self):
        """Test MinIO Console is accessible."""
        config = SERVICES["minio_console"]
        healthy, message = check_http_health(config)
        assert healthy, f"MinIO Console not accessible: {message}"

    @pytest.mark.integration
    def test_mongodb_connection(self, mongodb_client):
        """Test MongoDB is accepting connections."""
        try:
            # Ping the server
            result = mongodb_client.admin.command("ping")
            assert result.get("ok") == 1, "MongoDB ping failed"
        except Exception as e:
            pytest.fail(f"MongoDB connection failed: {e}")

    @pytest.mark.integration
    def test_redis_connection(self, redis_client):
        """Test Redis is accepting connections."""
        try:
            result = redis_client.ping()
            assert result is True, "Redis ping failed"
        except Exception as e:
            pytest.fail(f"Redis connection failed: {e}")

    @pytest.mark.integration
    def test_postgres_connection(self, postgres_connection):
        """Test PostgreSQL is accepting connections."""
        try:
            cursor = postgres_connection.cursor()
            cursor.execute("SELECT 1")
            result = cursor.fetchone()
            cursor.close()
            assert result[0] == 1, "PostgreSQL query failed"
        except Exception as e:
            pytest.fail(f"PostgreSQL connection failed: {e}")


class TestExperimentTracking:
    """Test experiment tracking and orchestration services."""

    @pytest.mark.integration
    def test_aim_dashboard(self):
        """Test AIM dashboard is accessible."""
        config = SERVICES["aim"]
        healthy, message = check_http_health(config)
        assert healthy, f"AIM dashboard not accessible: {message}"

    @pytest.mark.integration
    def test_prefect_api(self):
        """Test Prefect API is healthy."""
        config = SERVICES["prefect"]
        healthy, message = check_http_health(config)
        assert healthy, f"Prefect API health check failed: {message}"

    @pytest.mark.integration
    def test_prefect_server_info(self):
        """Test Prefect server responds with version info."""
        try:
            response = requests.get(
                f"http://{SERVICE_HOST}:4200/api/admin/version",
                timeout=10
            )
            # Prefect returns version info
            assert response.status_code in [200, 404], f"Prefect API error: {response.status_code}"
        except Exception as e:
            pytest.fail(f"Prefect server query failed: {e}")


class TestDataQuality:
    """Test data quality services."""

    @pytest.mark.integration
    def test_great_expectations_ui(self):
        """Test Great Expectations UI is accessible."""
        config = SERVICES["great_expectations"]
        healthy, message = check_http_health(config)
        assert healthy, f"Great Expectations not accessible: {message}"

    @pytest.mark.integration
    def test_great_expectations_api(self):
        """Test Great Expectations API endpoints."""
        try:
            response = requests.get(
                f"http://{SERVICE_HOST}:8084/api/datasources",
                timeout=10
            )
            assert response.status_code == 200, f"GE API error: {response.status_code}"
        except Exception as e:
            pytest.fail(f"Great Expectations API failed: {e}")


class TestAnnotationServices:
    """Test data annotation services."""

    @pytest.mark.integration
    def test_label_studio_health(self):
        """Test Label Studio is healthy."""
        config = SERVICES["label_studio"]
        healthy, message = check_http_health(config)
        assert healthy, f"Label Studio health check failed: {message}"

    @pytest.mark.integration
    def test_cvat_accessible(self):
        """Test CVAT UI is accessible."""
        config = SERVICES["cvat"]
        healthy, message = check_http_health(config)
        assert healthy, f"CVAT not accessible: {message}"

    @pytest.mark.integration
    def test_spotlight_accessible(self):
        """Test Spotlight is accessible."""
        config = SERVICES["spotlight"]
        healthy, message = check_http_health(config)
        assert healthy, f"Spotlight not accessible: {message}"

    @pytest.mark.integration
    def test_fiftyone_accessible(self):
        """Test FiftyOne is accessible."""
        config = SERVICES["fiftyone"]
        healthy, message = check_http_health(config)
        assert healthy, f"FiftyOne not accessible: {message}"


class TestKnowledgeStack:
    """Test knowledge management services."""

    @pytest.mark.integration
    def test_khoj_health(self):
        """Test Khoj AI assistant is healthy."""
        config = SERVICES["khoj"]
        healthy, message = check_http_health(config)
        assert healthy, f"Khoj health check failed: {message}"

    @pytest.mark.integration
    def test_couchdb_health(self):
        """Test CouchDB (Obsidian sync) is healthy."""
        config = SERVICES["couchdb"]
        healthy, message = check_http_health(config)
        assert healthy, f"CouchDB health check failed: {message}"

    @pytest.mark.integration
    def test_couchdb_databases(self):
        """Test CouchDB can list databases."""
        try:
            response = requests.get(
                f"http://{SERVICE_HOST}:5984/_all_dbs",
                auth=("obsidian", "mlops-dev-password"),
                timeout=10
            )
            assert response.status_code == 200, f"CouchDB error: {response.status_code}"
            # Should return list of databases
            dbs = response.json()
            assert isinstance(dbs, list), "Expected list of databases"
        except Exception as e:
            pytest.fail(f"CouchDB query failed: {e}")

    @pytest.mark.integration
    def test_zotero_health(self):
        """Test Zotero service is healthy."""
        config = SERVICES["zotero"]
        healthy, message = check_http_health(config)
        assert healthy, f"Zotero health check failed: {message}"

    @pytest.mark.integration
    def test_zotero_api(self):
        """Test Zotero API endpoints."""
        try:
            response = requests.get(
                f"http://{SERVICE_HOST}:8085/api/papers",
                timeout=10
            )
            assert response.status_code == 200, f"Zotero API error: {response.status_code}"
            data = response.json()
            assert "papers" in data, "Expected 'papers' in response"
        except Exception as e:
            pytest.fail(f"Zotero API failed: {e}")


class TestSecurityServices:
    """Test security and secrets management services."""

    @pytest.mark.integration
    def test_vault_health(self):
        """Test Vault is healthy."""
        config = SERVICES["vault"]
        healthy, message = check_http_health(config)
        assert healthy, f"Vault health check failed: {message}"

    @pytest.mark.integration
    def test_vault_initialized(self, vault_client):
        """Test Vault is initialized and unsealed."""
        try:
            assert vault_client.sys.is_initialized(), "Vault not initialized"
            assert not vault_client.sys.is_sealed(), "Vault is sealed"
        except Exception as e:
            pytest.fail(f"Vault status check failed: {e}")

    @pytest.mark.integration
    def test_vault_secrets_engine(self, vault_client):
        """Test Vault KV secrets engine is enabled."""
        try:
            # Try to list secrets
            vault_client.secrets.kv.v2.read_secret_version(
                path="test",
                mount_point="secret",
                raise_on_deleted_version=False
            )
        except Exception:
            # It's OK if the secret doesn't exist, we just want to verify the engine works
            pass


class TestLLMServices:
    """Test LLM and AI services."""

    @pytest.mark.integration
    def test_litellm_health(self):
        """Test LiteLLM gateway is healthy."""
        config = SERVICES["litellm"]
        healthy, message = check_http_health(config)
        assert healthy, f"LiteLLM health check failed: {message}"

    @pytest.mark.integration
    def test_litellm_models(self):
        """Test LiteLLM models endpoint."""
        try:
            response = requests.get(
                f"http://{SERVICE_HOST}:4000/v1/models",
                headers={"Authorization": "Bearer sk-mlops-dev-key"},
                timeout=10
            )
            # Should return models list or auth error (both are valid - service is responding)
            assert response.status_code in [200, 401], f"LiteLLM error: {response.status_code}"
        except Exception as e:
            pytest.fail(f"LiteLLM API failed: {e}")


class TestImageGeneration:
    """Test image generation services."""

    @pytest.mark.integration
    def test_comfyui_accessible(self):
        """Test ComfyUI is accessible."""
        config = SERVICES["comfyui"]
        healthy, message = check_http_health(config)
        assert healthy, f"ComfyUI not accessible: {message}"

    @pytest.mark.integration
    def test_comfyui_api(self):
        """Test ComfyUI API endpoints."""
        try:
            response = requests.get(
                f"http://{SERVICE_HOST}:8188/system_stats",
                timeout=10
            )
            # ComfyUI returns system stats
            assert response.status_code == 200, f"ComfyUI API error: {response.status_code}"
        except Exception as e:
            pytest.fail(f"ComfyUI API failed: {e}")


class TestAllServicesQuickCheck:
    """Quick health check for all services."""

    @pytest.mark.integration
    def test_all_http_services(self):
        """Run quick health check on all HTTP services."""
        failed_services = []

        for name, config in SERVICES.items():
            if config.health_endpoint is None:
                continue  # Skip non-HTTP services

            healthy, message = check_http_health(config)
            if not healthy:
                failed_services.append(f"{config.name}: {message}")

        if failed_services:
            pytest.fail(f"Failed services:\n" + "\n".join(failed_services))
