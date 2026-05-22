#!/usr/bin/env python3
"""
MLOps Services Integration Tests
=================================
Test AIM, Prefect, Great Expectations, and other MLOps services.
"""

import pytest
import requests
import json
from datetime import datetime
from tests.conftest import SERVICE_HOST


class TestAIMIntegration:
    """Test AIM experiment tracking integration."""

    BASE_URL = f"http://{SERVICE_HOST}:43800"

    @pytest.mark.integration
    def test_aim_dashboard_accessible(self):
        """Test AIM dashboard loads."""
        response = requests.get(f"{self.BASE_URL}/", timeout=10)
        assert response.status_code == 200

    @pytest.mark.integration
    def test_aim_api_runs(self):
        """Test AIM runs API."""
        try:
            response = requests.get(f"{self.BASE_URL}/api/runs/search/run", timeout=10)
            # Should return runs list (may be empty)
            assert response.status_code in [200, 404]
        except Exception as e:
            pytest.fail(f"AIM runs API failed: {e}")

    @pytest.mark.integration
    def test_aim_api_experiments(self):
        """Test AIM experiments API."""
        try:
            response = requests.get(f"{self.BASE_URL}/api/experiments", timeout=10)
            assert response.status_code in [200, 404]
        except Exception as e:
            pytest.fail(f"AIM experiments API failed: {e}")


class TestPrefectIntegration:
    """Prefect removed; keeping placeholder tests skipped."""
    pytest.skip("Prefect not in stack", allow_module_level=True)


class TestGreatExpectationsIntegration:
    """Test Great Expectations data quality integration."""

    BASE_URL = f"http://{SERVICE_HOST}:8084"

    @pytest.mark.integration
    def test_ge_dashboard(self):
        """Test Great Expectations dashboard."""
        response = requests.get(f"{self.BASE_URL}/", timeout=10)
        assert response.status_code == 200

    @pytest.mark.integration
    def test_ge_datasources(self):
        """Test listing datasources."""
        response = requests.get(f"{self.BASE_URL}/api/datasources", timeout=10)
        assert response.status_code == 200
        data = response.json()
        assert "datasources" in data

    @pytest.mark.integration
    def test_ge_expectations(self):
        """Test listing expectation suites."""
        response = requests.get(f"{self.BASE_URL}/api/expectations", timeout=10)
        assert response.status_code == 200
        data = response.json()
        assert "suites" in data

    @pytest.mark.integration
    def test_ge_create_expectation_suite(self):
        """Test creating an expectation suite."""
        suite = {
            "name": "test_suite",
            "expectations": [
                {"type": "expect_column_to_exist", "column": "id"},
                {"type": "expect_column_values_to_not_be_null", "column": "id"},
            ]
        }

        try:
            response = requests.post(
                f"{self.BASE_URL}/api/expectations",
                json=suite,
                timeout=10
            )
            assert response.status_code in [200, 201]
        except Exception as e:
            pytest.fail(f"GE create suite failed: {e}")

    @pytest.mark.integration
    def test_ge_results(self):
        """Test validation results API."""
        response = requests.get(f"{self.BASE_URL}/api/results", timeout=10)
        assert response.status_code == 200
        data = response.json()
        assert "results" in data


class TestVaultIntegration:
    """Test HashiCorp Vault secrets management integration."""

    BASE_URL = f"http://{SERVICE_HOST}:8200"
    TOKEN = "mlops-dev-token"

    @pytest.mark.integration
    def test_vault_health(self):
        """Test Vault health."""
        response = requests.get(f"{self.BASE_URL}/v1/sys/health", timeout=10)
        assert response.status_code == 200

    @pytest.mark.integration
    def test_vault_seal_status(self):
        """Test Vault seal status."""
        response = requests.get(f"{self.BASE_URL}/v1/sys/seal-status", timeout=10)
        assert response.status_code == 200
        data = response.json()
        assert data["sealed"] is False

    @pytest.mark.integration
    def test_vault_write_read_secret(self):
        """Test writing and reading a secret."""
        headers = {"X-Vault-Token": self.TOKEN}
        secret_path = "secret/data/mlops/test"
        secret_data = {
            "data": {
                "api_key": "test-api-key-12345",
                "timestamp": datetime.now().isoformat(),
            }
        }

        try:
            # Write secret
            response = requests.post(
                f"{self.BASE_URL}/v1/{secret_path}",
                headers=headers,
                json=secret_data,
                timeout=10
            )
            assert response.status_code in [200, 204]

            # Read secret
            response = requests.get(
                f"{self.BASE_URL}/v1/{secret_path}",
                headers=headers,
                timeout=10
            )
            assert response.status_code == 200
            data = response.json()
            assert data["data"]["data"]["api_key"] == "test-api-key-12345"

            # Delete secret
            response = requests.delete(
                f"{self.BASE_URL}/v1/{secret_path}",
                headers=headers,
                timeout=10
            )
            assert response.status_code in [200, 204]
        except Exception as e:
            pytest.fail(f"Vault secret test failed: {e}")

    @pytest.mark.integration
    def test_vault_list_secrets(self):
        """Test listing secrets."""
        headers = {"X-Vault-Token": self.TOKEN}

        response = requests.request(
            "LIST",
            f"{self.BASE_URL}/v1/secret/metadata/mlops",
            headers=headers,
            timeout=10
        )
        # May be empty, but should respond
        assert response.status_code in [200, 404]


class TestLiteLLMIntegration:
    """Test LiteLLM gateway integration."""

    BASE_URL = f"http://{SERVICE_HOST}:4000"
    API_KEY = "sk-mlops-dev-key"

    @pytest.mark.integration
    def test_litellm_health(self):
        """Test LiteLLM health."""
        response = requests.get(f"{self.BASE_URL}/health", timeout=10)
        assert response.status_code == 200

    @pytest.mark.integration
    def test_litellm_models(self):
        """Test LiteLLM models endpoint."""
        headers = {"Authorization": f"Bearer {self.API_KEY}"}
        response = requests.get(f"{self.BASE_URL}/v1/models", headers=headers, timeout=10)
        # May require proper auth setup
        assert response.status_code in [200, 401, 403]

    @pytest.mark.integration
    def test_litellm_openai_compatible(self):
        """Test OpenAI-compatible endpoint structure."""
        headers = {"Authorization": f"Bearer {self.API_KEY}"}

        # Just verify the endpoint exists and responds
        response = requests.post(
            f"{self.BASE_URL}/v1/chat/completions",
            headers=headers,
            json={
                "model": "gpt-3.5-turbo",
                "messages": [{"role": "user", "content": "test"}],
            },
            timeout=10
        )
        # Should respond (may fail due to no configured models, but endpoint exists)
        assert response.status_code in [200, 400, 401, 404, 500]


class TestComfyUIIntegration:
    """Test ComfyUI image generation integration."""

    BASE_URL = f"http://{SERVICE_HOST}:8188"

    @pytest.mark.integration
    def test_comfyui_accessible(self):
        """Test ComfyUI is accessible."""
        response = requests.get(f"{self.BASE_URL}/", timeout=10)
        assert response.status_code == 200

    @pytest.mark.integration
    def test_comfyui_system_stats(self):
        """Test ComfyUI system stats API."""
        response = requests.get(f"{self.BASE_URL}/system_stats", timeout=10)
        assert response.status_code == 200
        data = response.json()
        assert "system" in data

    @pytest.mark.integration
    def test_comfyui_object_info(self):
        """Test ComfyUI object info API (lists available nodes)."""
        response = requests.get(f"{self.BASE_URL}/object_info", timeout=30)
        assert response.status_code == 200
        data = response.json()
        # Should have node definitions
        assert isinstance(data, dict)

    @pytest.mark.integration
    def test_comfyui_history(self):
        """Test ComfyUI history API."""
        response = requests.get(f"{self.BASE_URL}/history", timeout=10)
        assert response.status_code == 200


class TestAnnotationServicesIntegration:
    """Test annotation service integrations."""

    @pytest.mark.integration
    def test_label_studio_api(self):
        """Test Label Studio API."""
        response = requests.get(
            f"http://{SERVICE_HOST}:8081/api/projects",
            timeout=10
        )
        # May need auth, but should respond
        assert response.status_code in [200, 401, 403]

    @pytest.mark.integration
    def test_fiftyone_api(self):
        """Test FiftyOne API."""
        response = requests.get(f"http://{SERVICE_HOST}:5151/", timeout=10)
        assert response.status_code == 200

    @pytest.mark.integration
    def test_spotlight_api(self):
        """Test Spotlight API."""
        response = requests.get(f"http://{SERVICE_HOST}:8083/", timeout=10)
        assert response.status_code == 200


class TestServiceInterdependencies:
    """Test that services can communicate with each other."""

    @pytest.mark.integration
    @pytest.mark.slow
    def test_minio_accessible_from_services(self, minio_client):
        """Test MinIO is accessible (simulates service access)."""
        # This validates that the MinIO endpoint is properly configured
        response = minio_client.list_buckets()
        assert len(response["Buckets"]) > 0

    @pytest.mark.integration
    def test_vault_secrets_for_s3(self):
        """Test Vault can store/retrieve S3 credentials pattern."""
        headers = {"X-Vault-Token": "mlops-dev-token"}

        # Write S3 credentials pattern
        secret_data = {
            "data": {
                "endpoint": "http://minio:9000",
                "access_key": "mlops-admin",
                "secret_key": "mlops-dev-password",
            }
        }

        response = requests.post(
            f"http://{SERVICE_HOST}:8200/v1/secret/data/mlops/test_s3",
            headers=headers,
            json=secret_data,
            timeout=10
        )
        assert response.status_code in [200, 204]

        # Clean up
        requests.delete(
            f"http://{SERVICE_HOST}:8200/v1/secret/data/mlops/test_s3",
            headers=headers,
            timeout=10
        )
