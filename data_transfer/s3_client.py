"""
S3-Compatible Storage Client
============================
Unified client for S3-compatible object storage that works with:
- MinIO (local development)
- Backblaze B2 (production via S3-compatible API)
- AWS S3
- Any S3-compatible storage

This makes the codebase cloud-agnostic - switch providers by changing
environment variables, no code changes needed.

Environment Variables:
    S3_ENDPOINT: S3 API endpoint (e.g., http://minio:9000, https://s3.us-west-000.backblazeb2.com)
    S3_ACCESS_KEY: Access key ID
    S3_SECRET_KEY: Secret access key
    S3_BUCKET: Default bucket name
    S3_REGION: Region (default: us-east-1)

Usage:
    from data_transfer.s3_client import S3Client

    client = S3Client()
    client.upload_file(Path("model.pt"), "models/v1/model.pt")
    client.download_file("models/v1/model.pt", Path("./model.pt"))
"""

import os
import hashlib
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Iterator, BinaryIO
from urllib.parse import urlparse

# boto3 for S3-compatible API
try:
    import boto3
    from botocore.config import Config
    from botocore.exceptions import ClientError, NoCredentialsError
    BOTO3_AVAILABLE = True
except ImportError:
    BOTO3_AVAILABLE = False
    boto3 = None
    Config = None
    ClientError = Exception
    NoCredentialsError = Exception


@dataclass
class S3Object:
    """Represents an object in S3-compatible storage."""
    key: str
    size: int
    last_modified: datetime
    etag: str
    storage_class: str = "STANDARD"
    metadata: Dict[str, str] = field(default_factory=dict)

    @property
    def name(self) -> str:
        """Get the object name (last part of key)."""
        return Path(self.key).name


@dataclass
class S3Config:
    """Configuration for S3-compatible storage."""
    endpoint: str
    access_key: str
    secret_key: str
    bucket: str
    region: str = "us-east-1"
    secure: bool = True
    path_style: bool = True  # MinIO requires path-style addressing

    @classmethod
    def from_env(cls) -> "S3Config":
        """Create config from environment variables."""
        endpoint = os.environ.get("S3_ENDPOINT", "http://localhost:9000")
        return cls(
            endpoint=endpoint,
            access_key=os.environ.get("S3_ACCESS_KEY", "mlops-admin"),
            secret_key=os.environ.get("S3_SECRET_KEY", "mlops-dev-password"),
            bucket=os.environ.get("S3_BUCKET", "mlops-data"),
            region=os.environ.get("S3_REGION", "us-east-1"),
            secure=endpoint.startswith("https"),
            path_style=True,  # Always use path-style for compatibility
        )

    @classmethod
    def for_backblaze(
        cls,
        key_id: str,
        app_key: str,
        bucket: str,
        region: str = "us-west-000"
    ) -> "S3Config":
        """Create config for Backblaze B2 S3-compatible API."""
        return cls(
            endpoint=f"https://s3.{region}.backblazeb2.com",
            access_key=key_id,
            secret_key=app_key,
            bucket=bucket,
            region=region,
            secure=True,
            path_style=False,  # B2 uses virtual-hosted style
        )

    @classmethod
    def for_aws(
        cls,
        access_key: str,
        secret_key: str,
        bucket: str,
        region: str = "us-east-1"
    ) -> "S3Config":
        """Create config for AWS S3."""
        return cls(
            endpoint=f"https://s3.{region}.amazonaws.com",
            access_key=access_key,
            secret_key=secret_key,
            bucket=bucket,
            region=region,
            secure=True,
            path_style=False,
        )


class S3Client:
    """
    S3-compatible storage client.

    Works with MinIO, Backblaze B2, AWS S3, and any S3-compatible storage.
    """

    def __init__(self, config: Optional[S3Config] = None):
        """
        Initialize S3 client.

        Args:
            config: S3 configuration. If None, loads from environment.
        """
        if not BOTO3_AVAILABLE:
            raise ImportError(
                "boto3 is required for S3 storage. "
                "Install with: pip install boto3"
            )

        self.config = config or S3Config.from_env()
        self._client = self._create_client()
        self._resource = self._create_resource()

    def _create_client(self):
        """Create boto3 S3 client."""
        boto_config = Config(
            signature_version='s3v4',
            s3={'addressing_style': 'path' if self.config.path_style else 'virtual'}
        )

        return boto3.client(
            's3',
            endpoint_url=self.config.endpoint,
            aws_access_key_id=self.config.access_key,
            aws_secret_access_key=self.config.secret_key,
            region_name=self.config.region,
            config=boto_config,
        )

    def _create_resource(self):
        """Create boto3 S3 resource."""
        boto_config = Config(
            signature_version='s3v4',
            s3={'addressing_style': 'path' if self.config.path_style else 'virtual'}
        )

        return boto3.resource(
            's3',
            endpoint_url=self.config.endpoint,
            aws_access_key_id=self.config.access_key,
            aws_secret_access_key=self.config.secret_key,
            region_name=self.config.region,
            config=boto_config,
        )

    # =========================================================================
    # Bucket Operations
    # =========================================================================

    def create_bucket(self, bucket: Optional[str] = None) -> bool:
        """
        Create a bucket if it doesn't exist.

        Args:
            bucket: Bucket name. Uses default if not specified.

        Returns:
            True if created or already exists, False on error.
        """
        bucket = bucket or self.config.bucket
        try:
            self._client.head_bucket(Bucket=bucket)
            print(f"[S3] Bucket '{bucket}' already exists")
            return True
        except ClientError as e:
            error_code = e.response.get('Error', {}).get('Code', '')
            if error_code == '404':
                try:
                    if self.config.region == 'us-east-1':
                        self._client.create_bucket(Bucket=bucket)
                    else:
                        self._client.create_bucket(
                            Bucket=bucket,
                            CreateBucketConfiguration={
                                'LocationConstraint': self.config.region
                            }
                        )
                    print(f"[S3] Created bucket '{bucket}'")
                    return True
                except ClientError as create_error:
                    print(f"[S3] Failed to create bucket: {create_error}")
                    return False
            else:
                print(f"[S3] Error checking bucket: {e}")
                return False

    def list_buckets(self) -> List[str]:
        """List all buckets."""
        try:
            response = self._client.list_buckets()
            return [b['Name'] for b in response.get('Buckets', [])]
        except ClientError as e:
            print(f"[S3] Error listing buckets: {e}")
            return []

    # =========================================================================
    # Object Operations
    # =========================================================================

    def list_objects(
        self,
        prefix: str = "",
        bucket: Optional[str] = None,
        max_keys: int = 1000
    ) -> List[S3Object]:
        """
        List objects in bucket.

        Args:
            prefix: Filter by key prefix.
            bucket: Bucket name. Uses default if not specified.
            max_keys: Maximum number of keys to return.

        Returns:
            List of S3Object instances.
        """
        bucket = bucket or self.config.bucket
        objects = []

        try:
            paginator = self._client.get_paginator('list_objects_v2')
            pages = paginator.paginate(
                Bucket=bucket,
                Prefix=prefix,
                PaginationConfig={'MaxItems': max_keys}
            )

            for page in pages:
                for obj in page.get('Contents', []):
                    objects.append(S3Object(
                        key=obj['Key'],
                        size=obj['Size'],
                        last_modified=obj['LastModified'],
                        etag=obj['ETag'].strip('"'),
                        storage_class=obj.get('StorageClass', 'STANDARD'),
                    ))

            print(f"[S3] Found {len(objects)} objects with prefix '{prefix}'")
            return objects

        except ClientError as e:
            print(f"[S3] Error listing objects: {e}")
            return []

    def upload_file(
        self,
        source: Path,
        key: str,
        bucket: Optional[str] = None,
        metadata: Optional[Dict[str, str]] = None,
        content_type: Optional[str] = None
    ) -> bool:
        """
        Upload a file to S3.

        Args:
            source: Local file path.
            key: S3 object key (path in bucket).
            bucket: Bucket name. Uses default if not specified.
            metadata: Optional metadata dict.
            content_type: Optional content type.

        Returns:
            True on success, False on error.
        """
        bucket = bucket or self.config.bucket
        source = Path(source)

        if not source.exists():
            print(f"[S3] Error: Source file not found: {source}")
            return False

        try:
            extra_args = {}
            if metadata:
                extra_args['Metadata'] = metadata
            if content_type:
                extra_args['ContentType'] = content_type

            print(f"[S3] Uploading {source} -> s3://{bucket}/{key}")
            self._client.upload_file(
                str(source),
                bucket,
                key,
                ExtraArgs=extra_args if extra_args else None
            )
            print(f"[S3] Upload complete: {source.stat().st_size} bytes")
            return True

        except ClientError as e:
            print(f"[S3] Upload error: {e}")
            return False

    def upload_fileobj(
        self,
        fileobj: BinaryIO,
        key: str,
        bucket: Optional[str] = None,
        metadata: Optional[Dict[str, str]] = None
    ) -> bool:
        """
        Upload a file-like object to S3.

        Args:
            fileobj: File-like object (must be opened in binary mode).
            key: S3 object key.
            bucket: Bucket name.
            metadata: Optional metadata.

        Returns:
            True on success.
        """
        bucket = bucket or self.config.bucket

        try:
            extra_args = {'Metadata': metadata} if metadata else None
            self._client.upload_fileobj(fileobj, bucket, key, ExtraArgs=extra_args)
            print(f"[S3] Uploaded stream to s3://{bucket}/{key}")
            return True
        except ClientError as e:
            print(f"[S3] Upload error: {e}")
            return False

    def download_file(
        self,
        key: str,
        destination: Path,
        bucket: Optional[str] = None
    ) -> bool:
        """
        Download a file from S3.

        Args:
            key: S3 object key.
            destination: Local file path.
            bucket: Bucket name.

        Returns:
            True on success.
        """
        bucket = bucket or self.config.bucket
        destination = Path(destination)

        try:
            # Ensure parent directory exists
            destination.parent.mkdir(parents=True, exist_ok=True)

            print(f"[S3] Downloading s3://{bucket}/{key} -> {destination}")
            self._client.download_file(bucket, key, str(destination))
            print(f"[S3] Download complete: {destination.stat().st_size} bytes")
            return True

        except ClientError as e:
            print(f"[S3] Download error: {e}")
            return False

    def download_fileobj(
        self,
        key: str,
        fileobj: BinaryIO,
        bucket: Optional[str] = None
    ) -> bool:
        """
        Download S3 object to a file-like object.

        Args:
            key: S3 object key.
            fileobj: File-like object (must be opened in binary mode).
            bucket: Bucket name.

        Returns:
            True on success.
        """
        bucket = bucket or self.config.bucket

        try:
            self._client.download_fileobj(bucket, key, fileobj)
            return True
        except ClientError as e:
            print(f"[S3] Download error: {e}")
            return False

    def delete_object(self, key: str, bucket: Optional[str] = None) -> bool:
        """
        Delete an object from S3.

        Args:
            key: S3 object key.
            bucket: Bucket name.

        Returns:
            True on success.
        """
        bucket = bucket or self.config.bucket

        try:
            self._client.delete_object(Bucket=bucket, Key=key)
            print(f"[S3] Deleted s3://{bucket}/{key}")
            return True
        except ClientError as e:
            print(f"[S3] Delete error: {e}")
            return False

    def object_exists(self, key: str, bucket: Optional[str] = None) -> bool:
        """
        Check if an object exists.

        Args:
            key: S3 object key.
            bucket: Bucket name.

        Returns:
            True if exists.
        """
        bucket = bucket or self.config.bucket

        try:
            self._client.head_object(Bucket=bucket, Key=key)
            return True
        except ClientError:
            return False

    def get_object_metadata(
        self,
        key: str,
        bucket: Optional[str] = None
    ) -> Optional[S3Object]:
        """
        Get object metadata without downloading.

        Args:
            key: S3 object key.
            bucket: Bucket name.

        Returns:
            S3Object with metadata or None.
        """
        bucket = bucket or self.config.bucket

        try:
            response = self._client.head_object(Bucket=bucket, Key=key)
            return S3Object(
                key=key,
                size=response['ContentLength'],
                last_modified=response['LastModified'],
                etag=response['ETag'].strip('"'),
                metadata=response.get('Metadata', {}),
            )
        except ClientError:
            return None

    def generate_presigned_url(
        self,
        key: str,
        bucket: Optional[str] = None,
        expiration: int = 3600,
        method: str = 'get_object'
    ) -> Optional[str]:
        """
        Generate a presigned URL for temporary access.

        Args:
            key: S3 object key.
            bucket: Bucket name.
            expiration: URL expiration in seconds.
            method: S3 method ('get_object' or 'put_object').

        Returns:
            Presigned URL or None.
        """
        bucket = bucket or self.config.bucket

        try:
            url = self._client.generate_presigned_url(
                method,
                Params={'Bucket': bucket, 'Key': key},
                ExpiresIn=expiration
            )
            return url
        except ClientError as e:
            print(f"[S3] Error generating presigned URL: {e}")
            return None

    # =========================================================================
    # Sync Operations
    # =========================================================================

    def sync_directory(
        self,
        source_dir: Path,
        prefix: str,
        bucket: Optional[str] = None,
        delete: bool = False
    ) -> int:
        """
        Sync a local directory to S3.

        Args:
            source_dir: Local directory path.
            prefix: S3 key prefix.
            bucket: Bucket name.
            delete: If True, delete S3 objects not in source.

        Returns:
            Number of files uploaded.
        """
        bucket = bucket or self.config.bucket
        source_dir = Path(source_dir)

        if not source_dir.is_dir():
            print(f"[S3] Error: Not a directory: {source_dir}")
            return 0

        uploaded = 0
        for file_path in source_dir.rglob("*"):
            if file_path.is_file():
                relative = file_path.relative_to(source_dir)
                key = f"{prefix}/{relative}".replace("\\", "/")

                # Check if upload needed (compare size/etag)
                remote = self.get_object_metadata(key, bucket)
                if remote and remote.size == file_path.stat().st_size:
                    continue

                if self.upload_file(file_path, key, bucket):
                    uploaded += 1

        print(f"[S3] Synced {uploaded} files to s3://{bucket}/{prefix}")
        return uploaded

    def download_directory(
        self,
        prefix: str,
        destination: Path,
        bucket: Optional[str] = None
    ) -> int:
        """
        Download all objects with prefix to local directory.

        Args:
            prefix: S3 key prefix.
            destination: Local directory.
            bucket: Bucket name.

        Returns:
            Number of files downloaded.
        """
        bucket = bucket or self.config.bucket
        destination = Path(destination)
        destination.mkdir(parents=True, exist_ok=True)

        objects = self.list_objects(prefix, bucket)
        downloaded = 0

        for obj in objects:
            relative = obj.key[len(prefix):].lstrip("/")
            local_path = destination / relative

            if self.download_file(obj.key, local_path, bucket):
                downloaded += 1

        print(f"[S3] Downloaded {downloaded} files to {destination}")
        return downloaded


# Convenience function for quick access
def get_s3_client(config: Optional[S3Config] = None) -> S3Client:
    """Get S3 client instance."""
    return S3Client(config)
