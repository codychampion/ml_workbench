"""
Data Transfer Module
====================
Handles cloud storage operations with rate limiting and security features.

Recommended: Use S3Client for new code - works with MinIO locally and
any S3-compatible storage (Backblaze B2, AWS S3, etc.) in production.

Legacy: B2Client/MockedB2Client are kept for backwards compatibility.
"""

from .b2_client import B2Client, MockedB2Client

# S3-compatible client (recommended for new code)
try:
    from .s3_client import S3Client, S3Config, S3Object, get_s3_client
    S3_AVAILABLE = True
except ImportError:
    S3Client = None
    S3Config = None
    S3Object = None
    get_s3_client = None
    S3_AVAILABLE = False

__all__ = [
    # Legacy B2 clients
    "B2Client",
    "MockedB2Client",
    # S3-compatible client (recommended)
    "S3Client",
    "S3Config",
    "S3Object",
    "get_s3_client",
    "S3_AVAILABLE",
]
