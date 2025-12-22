"""Service Health Check Tests - Lean Stack."""

import pytest
from conftest import SERVICES, check_http_health


class TestCoreInfrastructure:
    """Test core services (MinIO, Postgres, MongoDB)."""

    @pytest.mark.integration
    def test_minio_health(self):
        healthy, msg = check_http_health(SERVICES["minio"])
        assert healthy, f"MinIO failed: {msg}"

    @pytest.mark.integration
    def test_minio_console(self):
        healthy, msg = check_http_health(SERVICES["minio_console"])
        assert healthy, f"MinIO Console failed: {msg}"

    @pytest.mark.integration
    def test_mongodb_connection(self, mongodb_client):
        result = mongodb_client.admin.command("ping")
        assert result.get("ok") == 1

    @pytest.mark.integration
    def test_postgres_connection(self, postgres_connection):
        cursor = postgres_connection.cursor()
        cursor.execute("SELECT 1")
        assert cursor.fetchone()[0] == 1
        cursor.close()


class TestOptionalServices:
    """Test optional profile services."""

    @pytest.mark.integration
    def test_aim_dashboard(self):
        healthy, msg = check_http_health(SERVICES["aim"])
        assert healthy, f"AIM failed: {msg}"

    @pytest.mark.integration
    def test_label_studio(self):
        healthy, msg = check_http_health(SERVICES["label_studio"])
        assert healthy, f"Label Studio failed: {msg}"

    @pytest.mark.integration
    def test_khoj(self):
        healthy, msg = check_http_health(SERVICES["khoj"])
        assert healthy, f"Khoj failed: {msg}"

    @pytest.mark.integration
    def test_fiftyone(self):
        healthy, msg = check_http_health(SERVICES["fiftyone"])
        assert healthy, f"FiftyOne failed: {msg}"


class TestQuickHealthCheck:
    """Quick check of all HTTP services."""

    @pytest.mark.integration
    def test_all_http_services(self):
        failed = []
        for name, config in SERVICES.items():
            if config.health_endpoint:
                healthy, msg = check_http_health(config)
                if not healthy:
                    failed.append(f"{config.name}: {msg}")
        if failed:
            pytest.fail(f"Failed: {', '.join(failed)}")
