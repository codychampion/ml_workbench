#!/usr/bin/env python3
"""
MinIO/S3 Integration Tests
===========================
Test S3-compatible storage operations with MinIO.
"""

import io
import json
import pytest
from datetime import datetime
from tests.conftest import SERVICE_HOST


class TestMinIOBuckets:
    """Test MinIO bucket operations."""

    @pytest.mark.integration
    def test_list_buckets(self, minio_client):
        """Test listing buckets."""
        response = minio_client.list_buckets()
        buckets = [b["Name"] for b in response["Buckets"]]

        # Check expected buckets exist
        expected_buckets = ["mlops-data", "mlops-models", "mlops-outputs"]
        for bucket in expected_buckets:
            assert bucket in buckets, f"Expected bucket '{bucket}' not found"

    @pytest.mark.integration
    def test_bucket_exists(self, minio_client):
        """Test bucket existence check."""
        try:
            minio_client.head_bucket(Bucket="mlops-data")
        except Exception as e:
            pytest.fail(f"Bucket 'mlops-data' does not exist: {e}")

    @pytest.mark.integration
    def test_create_test_bucket(self, minio_client):
        """Test creating a new bucket."""
        test_bucket = "mlops-test-bucket"

        try:
            # Create bucket
            minio_client.create_bucket(Bucket=test_bucket)

            # Verify it exists
            minio_client.head_bucket(Bucket=test_bucket)

            # Clean up
            minio_client.delete_bucket(Bucket=test_bucket)
        except Exception as e:
            # Clean up on failure
            try:
                minio_client.delete_bucket(Bucket=test_bucket)
            except:
                pass
            pytest.fail(f"Bucket creation test failed: {e}")


class TestMinIOObjects:
    """Test MinIO object operations."""

    TEST_BUCKET = "mlops-data"
    TEST_PREFIX = "_test/"

    @pytest.mark.integration
    def test_upload_object(self, minio_client):
        """Test uploading an object."""
        key = f"{self.TEST_PREFIX}test_upload.txt"
        content = f"Test upload at {datetime.now().isoformat()}"

        try:
            minio_client.put_object(
                Bucket=self.TEST_BUCKET,
                Key=key,
                Body=content.encode(),
                ContentType="text/plain",
            )

            # Verify object exists
            minio_client.head_object(Bucket=self.TEST_BUCKET, Key=key)

            # Clean up
            minio_client.delete_object(Bucket=self.TEST_BUCKET, Key=key)
        except Exception as e:
            pytest.fail(f"Object upload failed: {e}")

    @pytest.mark.integration
    def test_download_object(self, minio_client):
        """Test downloading an object."""
        key = f"{self.TEST_PREFIX}test_download.txt"
        content = "Test download content"

        try:
            # Upload
            minio_client.put_object(
                Bucket=self.TEST_BUCKET,
                Key=key,
                Body=content.encode(),
            )

            # Download
            response = minio_client.get_object(Bucket=self.TEST_BUCKET, Key=key)
            downloaded = response["Body"].read().decode()

            assert downloaded == content, "Downloaded content doesn't match"

            # Clean up
            minio_client.delete_object(Bucket=self.TEST_BUCKET, Key=key)
        except Exception as e:
            pytest.fail(f"Object download failed: {e}")

    @pytest.mark.integration
    def test_list_objects(self, minio_client):
        """Test listing objects."""
        keys = [
            f"{self.TEST_PREFIX}list_test_1.txt",
            f"{self.TEST_PREFIX}list_test_2.txt",
            f"{self.TEST_PREFIX}list_test_3.txt",
        ]

        try:
            # Upload test objects
            for key in keys:
                minio_client.put_object(
                    Bucket=self.TEST_BUCKET,
                    Key=key,
                    Body=b"test",
                )

            # List objects with prefix
            response = minio_client.list_objects_v2(
                Bucket=self.TEST_BUCKET,
                Prefix=self.TEST_PREFIX,
            )

            listed_keys = [obj["Key"] for obj in response.get("Contents", [])]
            for key in keys:
                assert key in listed_keys, f"Expected key '{key}' not in listing"

            # Clean up
            for key in keys:
                minio_client.delete_object(Bucket=self.TEST_BUCKET, Key=key)
        except Exception as e:
            pytest.fail(f"Object listing failed: {e}")

    @pytest.mark.integration
    def test_delete_object(self, minio_client):
        """Test deleting an object."""
        key = f"{self.TEST_PREFIX}test_delete.txt"

        try:
            # Upload
            minio_client.put_object(
                Bucket=self.TEST_BUCKET,
                Key=key,
                Body=b"to delete",
            )

            # Verify exists
            minio_client.head_object(Bucket=self.TEST_BUCKET, Key=key)

            # Delete
            minio_client.delete_object(Bucket=self.TEST_BUCKET, Key=key)

            # Verify deleted
            try:
                minio_client.head_object(Bucket=self.TEST_BUCKET, Key=key)
                pytest.fail("Object should have been deleted")
            except minio_client.exceptions.ClientError as e:
                if e.response["Error"]["Code"] != "404":
                    raise
        except Exception as e:
            pytest.fail(f"Object deletion failed: {e}")

    @pytest.mark.integration
    def test_upload_json(self, minio_client):
        """Test uploading JSON data."""
        key = f"{self.TEST_PREFIX}test_data.json"
        data = {
            "experiment": "test",
            "metrics": {"loss": 0.5, "accuracy": 0.95},
            "timestamp": datetime.now().isoformat(),
        }

        try:
            minio_client.put_object(
                Bucket=self.TEST_BUCKET,
                Key=key,
                Body=json.dumps(data).encode(),
                ContentType="application/json",
            )

            # Download and verify
            response = minio_client.get_object(Bucket=self.TEST_BUCKET, Key=key)
            downloaded = json.loads(response["Body"].read().decode())

            assert downloaded["experiment"] == data["experiment"]
            assert downloaded["metrics"] == data["metrics"]

            # Clean up
            minio_client.delete_object(Bucket=self.TEST_BUCKET, Key=key)
        except Exception as e:
            pytest.fail(f"JSON upload failed: {e}")

    @pytest.mark.integration
    def test_multipart_upload(self, minio_client):
        """Test multipart upload for larger files."""
        key = f"{self.TEST_PREFIX}test_large.bin"
        # Create 6MB of data (minimum part size is 5MB)
        data = b"x" * (6 * 1024 * 1024)

        try:
            # Use upload_fileobj for multipart
            minio_client.upload_fileobj(
                io.BytesIO(data),
                self.TEST_BUCKET,
                key,
            )

            # Verify size
            response = minio_client.head_object(Bucket=self.TEST_BUCKET, Key=key)
            assert response["ContentLength"] == len(data)

            # Clean up
            minio_client.delete_object(Bucket=self.TEST_BUCKET, Key=key)
        except Exception as e:
            pytest.fail(f"Multipart upload failed: {e}")


class TestMinIOIntegrationWithServices:
    """Test MinIO integration with other services."""

    @pytest.mark.integration
    def test_presigned_url(self, minio_client):
        """Test generating presigned URLs."""
        key = "_test/presigned_test.txt"

        try:
            # Upload test file
            minio_client.put_object(
                Bucket="mlops-data",
                Key=key,
                Body=b"presigned content",
            )

            # Generate presigned URL
            url = minio_client.generate_presigned_url(
                "get_object",
                Params={"Bucket": "mlops-data", "Key": key},
                ExpiresIn=300,
            )

            assert url is not None
            assert "mlops-data" in url or key in url

            # Clean up
            minio_client.delete_object(Bucket="mlops-data", Key=key)
        except Exception as e:
            pytest.fail(f"Presigned URL generation failed: {e}")

    @pytest.mark.integration
    def test_bucket_policy_models(self, minio_client):
        """Test models bucket is accessible."""
        try:
            # List objects in models bucket
            response = minio_client.list_objects_v2(
                Bucket="mlops-models",
                MaxKeys=1,
            )
            # Should not throw an error
            assert "Contents" in response or response.get("KeyCount", 0) >= 0
        except Exception as e:
            pytest.fail(f"Models bucket access failed: {e}")

    @pytest.mark.integration
    def test_bucket_policy_outputs(self, minio_client):
        """Test outputs bucket is accessible (should be public download)."""
        import requests

        key = "_test/public_test.txt"

        try:
            # Upload test file
            minio_client.put_object(
                Bucket="mlops-outputs",
                Key=key,
                Body=b"public content",
            )

            # Try to download without auth (public download policy)
            url = f"http://{SERVICE_HOST}:9000/mlops-outputs/{key}"
            response = requests.get(url, timeout=10)

            # Should be accessible (public download policy was set in minio-init)
            assert response.status_code == 200, f"Public download failed: {response.status_code}"

            # Clean up
            minio_client.delete_object(Bucket="mlops-outputs", Key=key)
        except Exception as e:
            pytest.fail(f"Public bucket access test failed: {e}")
