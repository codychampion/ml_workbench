"""
Backblaze B2 Client Module
==========================
Provides secure, rate-limited access to B2 cloud storage.

Phase 1: Mocked implementation using local manifest and files
Phase 2/3: Full B2 SDK integration with encryption

Security Features:
- Rate limiting to prevent API abuse
- Simulated encrypted transfers (Phase 1)
- Real AES-256 encryption (Phase 2/3)
"""

import json
import shutil
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Protocol
import hashlib

# PHASE 2/3 TODO: Uncomment for real B2 integration
# from b2sdk.v2 import B2Api, InMemoryAccountInfo


@dataclass
class FileInfo:
    """Represents a file in the B2 bucket (or mocked local storage)."""

    file_name: str
    file_id: str
    size_bytes: int
    upload_timestamp: str
    content_sha256: str
    metadata: Dict[str, str] = field(default_factory=dict)


class B2ClientProtocol(Protocol):
    """Protocol defining the B2 client interface."""

    def list_files(self, prefix: str = "") -> List[FileInfo]: ...
    def download_file(self, file_name: str, destination: Path) -> bool: ...
    def upload_file(self, source: Path, destination_name: str) -> Optional[FileInfo]: ...


class RateLimiter:
    """Simple rate limiter for API calls."""

    def __init__(self, max_requests_per_minute: int = 100):
        self.max_requests = max_requests_per_minute
        self.request_times: List[float] = []

    def acquire(self) -> bool:
        """
        Attempt to acquire a rate limit slot.
        Returns True if allowed, False if rate limited.
        """
        now = time.time()
        # Remove requests older than 1 minute
        self.request_times = [t for t in self.request_times if now - t < 60]

        if len(self.request_times) >= self.max_requests:
            wait_time = 60 - (now - self.request_times[0])
            print(f"[RateLimiter] Rate limited. Wait {wait_time:.1f}s before retry.")
            return False

        self.request_times.append(now)
        return True

    def wait_if_needed(self) -> None:
        """Block until a rate limit slot is available."""
        while not self.acquire():
            time.sleep(1)


class MockedB2Client:
    """
    Mocked B2 client for Phase 1 local development.

    Uses a local JSON manifest to simulate B2's file listing API,
    and copies files from a local directory to simulate downloads.
    This allows testing the full pipeline without network calls or credentials.
    """

    def __init__(
        self,
        manifest_path: Path = Path(".b2_local_manifest.json"),
        local_data_dir: Path = Path("./data/raw"),
        max_requests_per_minute: int = 100
    ):
        self.manifest_path = Path(manifest_path)
        self.local_data_dir = Path(local_data_dir)
        self.rate_limiter = RateLimiter(max_requests_per_minute)

        # Ensure directories exist
        self.local_data_dir.mkdir(parents=True, exist_ok=True)

        # Initialize manifest if it doesn't exist
        if not self.manifest_path.exists():
            self._create_default_manifest()

    def _create_default_manifest(self) -> None:
        """Create a default manifest file with sample entries."""
        default_manifest = {
            "bucket_name": "mlops-data-bucket-mock",
            "files": [
                {
                    "file_name": "datasets/sample_images/image_001.png",
                    "file_id": "mock_file_001",
                    "size_bytes": 1024,
                    "upload_timestamp": datetime.now().isoformat(),
                    "content_sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
                    "metadata": {"type": "training_data", "project": "flux-comfyui"}
                },
                {
                    "file_name": "datasets/sample_images/image_002.png",
                    "file_id": "mock_file_002",
                    "size_bytes": 2048,
                    "upload_timestamp": datetime.now().isoformat(),
                    "content_sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
                    "metadata": {"type": "training_data", "project": "flux-comfyui"}
                },
                {
                    "file_name": "models/base_model_v1.pth",
                    "file_id": "mock_file_003",
                    "size_bytes": 10485760,
                    "upload_timestamp": datetime.now().isoformat(),
                    "content_sha256": "abc123def456",
                    "metadata": {"type": "model_checkpoint", "version": "1.0"}
                }
            ],
            "last_updated": datetime.now().isoformat()
        }

        with open(self.manifest_path, 'w') as f:
            json.dump(default_manifest, f, indent=2)

        print(f"[MockedB2Client] Created default manifest at {self.manifest_path}")

    def list_files(self, prefix: str = "") -> List[FileInfo]:
        """
        List files from the local manifest (simulates B2 list_file_names API).

        This demonstrates the API rate-limit guardrail - in production,
        B2's API has rate limits that must be respected.
        """
        # Simulate rate limiting
        self.rate_limiter.wait_if_needed()

        print(f"[MockedB2Client] Listing files with prefix: '{prefix}'")
        print("[MockedB2Client] (Reading from local manifest - no API call)")

        if not self.manifest_path.exists():
            print(f"[MockedB2Client] WARNING: Manifest not found at {self.manifest_path}")
            return []

        with open(self.manifest_path, 'r') as f:
            manifest = json.load(f)

        files = []
        for file_data in manifest.get("files", []):
            if prefix == "" or file_data["file_name"].startswith(prefix):
                files.append(FileInfo(
                    file_name=file_data["file_name"],
                    file_id=file_data["file_id"],
                    size_bytes=file_data["size_bytes"],
                    upload_timestamp=file_data["upload_timestamp"],
                    content_sha256=file_data["content_sha256"],
                    metadata=file_data.get("metadata", {})
                ))

        print(f"[MockedB2Client] Found {len(files)} files matching prefix")
        return files

    def download_file(self, file_name: str, destination: Path) -> bool:
        """
        Download a file (simulates B2 download with encrypted transfer).

        In Phase 1, this copies from the local data directory.
        Demonstrates the intended security workflow.
        """
        # Simulate rate limiting
        self.rate_limiter.wait_if_needed()

        print(f"[MockedB2Client] Downloading: {file_name}")
        print("[MockedB2Client] Simulating encrypted transfer...")

        # In Phase 1, look for the file in local_data_dir
        # Map the B2 path to local path
        local_source = self.local_data_dir / Path(file_name).name

        # Also check if full path exists
        if not local_source.exists():
            local_source = self.local_data_dir / file_name

        if not local_source.exists():
            # Create a placeholder file for testing
            print(f"[MockedB2Client] Source not found locally, creating placeholder")
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(f"# Placeholder for {file_name}\n# Created by MockedB2Client")
            return True

        # Simulate transfer delay (would be network latency in production)
        time.sleep(0.1)

        # Copy file
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(local_source, destination)

        print(f"[MockedB2Client] Simulating encrypted transfer... COMPLETE")
        print(f"[MockedB2Client] Saved to: {destination}")

        # PHASE 2/3 TODO: Implement actual AES-256 decryption
        # decrypted_data = decrypt_aes256(encrypted_data, self.encryption_key)

        return True

    def upload_file(self, source: Path, destination_name: str) -> Optional[FileInfo]:
        """
        Upload a file (simulates B2 upload with encryption).

        In Phase 1, this copies to local_data_dir and updates manifest.
        """
        # Simulate rate limiting
        self.rate_limiter.wait_if_needed()

        if not source.exists():
            print(f"[MockedB2Client] ERROR: Source file not found: {source}")
            return None

        print(f"[MockedB2Client] Uploading: {source} -> {destination_name}")
        print("[MockedB2Client] Simulating encrypted transfer...")

        # Calculate hash
        content_hash = hashlib.sha256(source.read_bytes()).hexdigest()

        # Copy to local storage
        dest_path = self.local_data_dir / destination_name
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, dest_path)

        # Update manifest
        file_info = FileInfo(
            file_name=destination_name,
            file_id=f"mock_{content_hash[:8]}",
            size_bytes=source.stat().st_size,
            upload_timestamp=datetime.now().isoformat(),
            content_sha256=content_hash,
            metadata={"source": str(source)}
        )

        self._add_to_manifest(file_info)

        print(f"[MockedB2Client] Simulating encrypted transfer... COMPLETE")
        print(f"[MockedB2Client] File ID: {file_info.file_id}")

        # PHASE 2/3 TODO: Implement actual AES-256 encryption before upload
        # encrypted_data = encrypt_aes256(data, self.encryption_key)

        return file_info

    def _add_to_manifest(self, file_info: FileInfo) -> None:
        """Add or update a file entry in the manifest."""
        manifest = {"bucket_name": "mlops-data-bucket-mock", "files": []}

        if self.manifest_path.exists():
            with open(self.manifest_path, 'r') as f:
                manifest = json.load(f)

        # Remove existing entry if present
        manifest["files"] = [
            f for f in manifest["files"]
            if f["file_name"] != file_info.file_name
        ]

        # Add new entry
        manifest["files"].append({
            "file_name": file_info.file_name,
            "file_id": file_info.file_id,
            "size_bytes": file_info.size_bytes,
            "upload_timestamp": file_info.upload_timestamp,
            "content_sha256": file_info.content_sha256,
            "metadata": file_info.metadata
        })

        manifest["last_updated"] = datetime.now().isoformat()

        with open(self.manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)


# PHASE 2/3 TODO: Implement real B2 client
class B2Client:
    """
    Real B2 client for production use.

    Phase 1: Alias to MockedB2Client
    Phase 2/3: Full B2 SDK implementation
    """

    def __new__(cls, *args, **kwargs):
        """
        Factory that returns MockedB2Client in Phase 1.

        PHASE 2/3 TODO: Remove this override and implement real B2 integration:

        def __init__(self, key_id: str, app_key: str, bucket_name: str):
            self.info = InMemoryAccountInfo()
            self.b2_api = B2Api(self.info)
            self.b2_api.authorize_account("production", key_id, app_key)
            self.bucket = self.b2_api.get_bucket_by_name(bucket_name)
        """
        print("[B2Client] Phase 1: Using MockedB2Client")
        return MockedB2Client(*args, **kwargs)


# PHASE 2/3 TODO: Add encryption utilities
# class EncryptionManager:
#     """Handles AES-256 encryption/decryption for secure data transfer."""
#
#     def __init__(self, key: bytes):
#         self.key = key
#
#     def encrypt(self, data: bytes) -> bytes:
#         """Encrypt data using AES-256-GCM."""
#         pass
#
#     def decrypt(self, encrypted_data: bytes) -> bytes:
#         """Decrypt data using AES-256-GCM."""
#         pass
