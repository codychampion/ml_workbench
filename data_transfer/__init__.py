"""
Data Transfer Module - S3-Compatible Storage
=============================================
Works with MinIO locally and any S3-compatible storage in production.
"""

try:
    from .s3_client import S3Client, S3Config, S3Object, get_s3_client
    S3_AVAILABLE = True
except ImportError:
    S3Client = S3Config = S3Object = get_s3_client = None
    S3_AVAILABLE = False

class B2Client:
    """
    Backblaze B2-compatible client placeholder that reuses the S3Client.

    Pipelines currently import B2Client; this wrapper forwards all operations
    to the S3-compatible implementation so we have a single storage client.
    """

    def __init__(self, config: "S3Config" = None):
        if not S3_AVAILABLE:
            raise ImportError("boto3 is required for B2Client/S3Client")
        self._client = S3Client(config)

    def __getattr__(self, item):
        return getattr(self._client, item)


__all__ = ["S3Client", "S3Config", "S3Object", "get_s3_client", "S3_AVAILABLE", "B2Client"]
