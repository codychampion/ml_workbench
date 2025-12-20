"""
AIM Model Registry Extension
=============================
Extends AIM with model registration, versioning, and S3 storage integration.

This provides a self-hosted alternative to MLflow Model Registry or W&B Artifacts.

Usage:
    from utils.model_registry import ModelRegistry, register_model, load_model

    # Register a model after training
    registry = ModelRegistry()
    model_info = registry.register(
        model_path="./outputs/model.pt",
        name="my-captioner",
        version="1.0.0",
        metrics={"accuracy": 0.95, "loss": 0.05},
        aim_run=run,  # Optional: link to AIM experiment
    )

    # Load a model
    model_path = registry.load("my-captioner", version="latest")
"""

import os
import json
import hashlib
import shutil
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any, List, Union

# AIM integration
try:
    from aim import Run
    AIM_AVAILABLE = True
except ImportError:
    AIM_AVAILABLE = False
    Run = None

# S3 integration
try:
    from data_transfer import S3Client
    S3_AVAILABLE = True
except ImportError:
    S3_AVAILABLE = False


@dataclass
class ModelVersion:
    """Represents a registered model version."""
    name: str
    version: str
    path: str  # S3 path or local path
    created_at: str
    metrics: Dict[str, float]
    params: Dict[str, Any]
    tags: List[str]
    aim_run_hash: Optional[str]
    checksum: str
    size_bytes: int
    stage: str  # "development", "staging", "production", "archived"
    description: str


@dataclass
class ModelInfo:
    """Metadata about a registered model."""
    name: str
    latest_version: str
    versions: List[str]
    created_at: str
    updated_at: str
    description: str
    tags: List[str]


class ModelRegistry:
    """
    Self-hosted model registry backed by S3 (MinIO) and AIM.

    Storage structure in S3:
        mlops-models/
        ├── registry/
        │   └── index.json          # Global model index
        ├── models/
        │   └── {model_name}/
        │       ├── metadata.json   # Model metadata
        │       └── versions/
        │           └── {version}/
        │               ├── model.pt (or model directory)
        │               └── version.json
    """

    def __init__(
        self,
        bucket: str = "mlops-models",
        local_cache: str = "./outputs/model_cache",
        s3_client: Optional['S3Client'] = None,
    ):
        """
        Initialize the model registry.

        Args:
            bucket: S3 bucket for model storage
            local_cache: Local cache directory for downloaded models
            s3_client: Optional pre-configured S3 client
        """
        self.bucket = bucket
        self.local_cache = Path(local_cache)
        self.local_cache.mkdir(parents=True, exist_ok=True)

        # Initialize S3 client
        if s3_client:
            self.s3 = s3_client
        elif S3_AVAILABLE:
            self.s3 = S3Client()
        else:
            self.s3 = None
            print("[ModelRegistry] Warning: S3 not available, using local storage only")

        # Local index for offline access
        self._index_path = self.local_cache / "index.json"
        self._index = self._load_local_index()

    def _load_local_index(self) -> Dict:
        """Load the local model index."""
        if self._index_path.exists():
            return json.loads(self._index_path.read_text())
        return {"models": {}, "updated_at": None}

    def _save_local_index(self) -> None:
        """Save the local model index."""
        self._index["updated_at"] = datetime.now().isoformat()
        self._index_path.write_text(json.dumps(self._index, indent=2))

    def _compute_checksum(self, path: Path) -> str:
        """Compute SHA256 checksum of a file or directory."""
        sha256 = hashlib.sha256()

        if path.is_file():
            sha256.update(path.read_bytes())
        else:
            # Hash all files in directory
            for file in sorted(path.rglob("*")):
                if file.is_file():
                    sha256.update(file.read_bytes())

        return sha256.hexdigest()[:16]

    def _get_size(self, path: Path) -> int:
        """Get size of file or directory in bytes."""
        if path.is_file():
            return path.stat().st_size
        return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())

    def register(
        self,
        model_path: Union[str, Path],
        name: str,
        version: Optional[str] = None,
        metrics: Optional[Dict[str, float]] = None,
        params: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
        description: str = "",
        aim_run: Optional[Run] = None,
        stage: str = "development",
    ) -> ModelVersion:
        """
        Register a model in the registry.

        Args:
            model_path: Path to model file or directory
            name: Model name (e.g., "blip-captioner-lora")
            version: Version string (auto-generated if not provided)
            metrics: Training metrics (accuracy, loss, etc.)
            params: Training parameters/hyperparameters
            tags: Model tags for filtering
            description: Human-readable description
            aim_run: Optional AIM run to link this model to
            stage: Deployment stage

        Returns:
            ModelVersion object with registration info
        """
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        # Auto-generate version if not provided
        if version is None:
            existing = self._index.get("models", {}).get(name, {}).get("versions", [])
            version = f"v{len(existing) + 1}"

        # Compute metadata
        checksum = self._compute_checksum(model_path)
        size_bytes = self._get_size(model_path)
        created_at = datetime.now().isoformat()

        # Get AIM run hash if available
        aim_run_hash = None
        if aim_run and AIM_AVAILABLE:
            aim_run_hash = aim_run.hash

        # Create version info
        model_version = ModelVersion(
            name=name,
            version=version,
            path=f"models/{name}/versions/{version}",
            created_at=created_at,
            metrics=metrics or {},
            params=params or {},
            tags=tags or [],
            aim_run_hash=aim_run_hash,
            checksum=checksum,
            size_bytes=size_bytes,
            stage=stage,
            description=description,
        )

        # Upload to S3 if available
        if self.s3:
            self._upload_model(model_path, model_version)
        else:
            # Store locally
            local_version_path = self.local_cache / name / version
            local_version_path.mkdir(parents=True, exist_ok=True)
            if model_path.is_file():
                shutil.copy(model_path, local_version_path / model_path.name)
            else:
                shutil.copytree(model_path, local_version_path / "model", dirs_exist_ok=True)

        # Update index
        if name not in self._index["models"]:
            self._index["models"][name] = {
                "created_at": created_at,
                "versions": [],
                "latest_version": None,
                "description": description,
                "tags": tags or [],
            }

        self._index["models"][name]["versions"].append(version)
        self._index["models"][name]["latest_version"] = version
        self._index["models"][name]["updated_at"] = created_at
        self._save_local_index()

        # Log to AIM if run provided
        if aim_run and AIM_AVAILABLE:
            aim_run["registered_model"] = {
                "name": name,
                "version": version,
                "path": model_version.path,
                "checksum": checksum,
                "stage": stage,
            }

        print(f"[ModelRegistry] Registered: {name}@{version}")
        print(f"[ModelRegistry] Checksum: {checksum}")
        print(f"[ModelRegistry] Size: {size_bytes / 1024 / 1024:.2f} MB")

        return model_version

    def _upload_model(self, local_path: Path, model_version: ModelVersion) -> None:
        """Upload model to S3."""
        s3_base = model_version.path

        # Upload model files
        if local_path.is_file():
            s3_key = f"{s3_base}/{local_path.name}"
            self.s3.upload_file(local_path, s3_key)
        else:
            for file in local_path.rglob("*"):
                if file.is_file():
                    relative = file.relative_to(local_path)
                    s3_key = f"{s3_base}/model/{relative}"
                    self.s3.upload_file(file, s3_key)

        # Upload version metadata
        version_json = f"{s3_base}/version.json"
        version_data = asdict(model_version)

        # Write to temp file and upload
        temp_path = self.local_cache / "temp_version.json"
        temp_path.write_text(json.dumps(version_data, indent=2))
        self.s3.upload_file(temp_path, version_json)
        temp_path.unlink()

    def load(
        self,
        name: str,
        version: str = "latest",
        download: bool = True,
    ) -> Path:
        """
        Load a model from the registry.

        Args:
            name: Model name
            version: Version string or "latest"
            download: Whether to download from S3 if not cached

        Returns:
            Path to the model file/directory
        """
        # Resolve "latest" version
        if version == "latest":
            model_info = self._index.get("models", {}).get(name)
            if not model_info:
                raise ValueError(f"Model not found: {name}")
            version = model_info["latest_version"]

        # Check local cache
        local_path = self.local_cache / name / version
        if local_path.exists():
            print(f"[ModelRegistry] Using cached: {name}@{version}")
            return local_path

        # Download from S3
        if download and self.s3:
            print(f"[ModelRegistry] Downloading: {name}@{version}")
            local_path.mkdir(parents=True, exist_ok=True)

            s3_prefix = f"models/{name}/versions/{version}/"
            objects = self.s3.list_objects(prefix=s3_prefix)

            for obj in objects:
                relative_key = obj.key[len(s3_prefix):]
                local_file = local_path / relative_key
                local_file.parent.mkdir(parents=True, exist_ok=True)
                self.s3.download_file(obj.key, local_file)

            return local_path

        raise FileNotFoundError(f"Model not found: {name}@{version}")

    def list_models(self) -> List[ModelInfo]:
        """List all registered models."""
        models = []
        for name, info in self._index.get("models", {}).items():
            models.append(ModelInfo(
                name=name,
                latest_version=info.get("latest_version"),
                versions=info.get("versions", []),
                created_at=info.get("created_at"),
                updated_at=info.get("updated_at"),
                description=info.get("description", ""),
                tags=info.get("tags", []),
            ))
        return models

    def list_versions(self, name: str) -> List[str]:
        """List all versions of a model."""
        model_info = self._index.get("models", {}).get(name)
        if not model_info:
            return []
        return model_info.get("versions", [])

    def get_version_info(self, name: str, version: str = "latest") -> Optional[ModelVersion]:
        """Get detailed info about a model version."""
        if version == "latest":
            model_info = self._index.get("models", {}).get(name)
            if not model_info:
                return None
            version = model_info["latest_version"]

        # Try to load from S3
        if self.s3:
            try:
                version_json = f"models/{name}/versions/{version}/version.json"
                local_path = self.local_cache / "temp_version_info.json"
                self.s3.download_file(version_json, local_path)
                data = json.loads(local_path.read_text())
                local_path.unlink()
                return ModelVersion(**data)
            except Exception:
                pass

        return None

    def promote(self, name: str, version: str, stage: str) -> None:
        """
        Promote a model version to a new stage.

        Args:
            name: Model name
            version: Version to promote
            stage: New stage ("staging", "production", "archived")
        """
        version_info = self.get_version_info(name, version)
        if not version_info:
            raise ValueError(f"Version not found: {name}@{version}")

        version_info.stage = stage

        # Update in S3
        if self.s3:
            version_json = f"models/{name}/versions/{version}/version.json"
            temp_path = self.local_cache / "temp_version.json"
            temp_path.write_text(json.dumps(asdict(version_info), indent=2))
            self.s3.upload_file(temp_path, version_json)
            temp_path.unlink()

        print(f"[ModelRegistry] Promoted {name}@{version} to {stage}")

    def delete_version(self, name: str, version: str) -> None:
        """Delete a model version."""
        # Remove from index
        if name in self._index.get("models", {}):
            versions = self._index["models"][name].get("versions", [])
            if version in versions:
                versions.remove(version)
                self._save_local_index()

        # Remove from local cache
        local_path = self.local_cache / name / version
        if local_path.exists():
            shutil.rmtree(local_path)

        # Remove from S3
        if self.s3:
            s3_prefix = f"models/{name}/versions/{version}/"
            objects = self.s3.list_objects(prefix=s3_prefix)
            for obj in objects:
                self.s3.delete_object(obj.key)

        print(f"[ModelRegistry] Deleted: {name}@{version}")

    def sync_from_s3(self) -> None:
        """Sync the local index from S3."""
        if not self.s3:
            return

        try:
            s3_index_path = "registry/index.json"
            local_index = self.local_cache / "s3_index.json"
            self.s3.download_file(s3_index_path, local_index)

            s3_index = json.loads(local_index.read_text())

            # Merge with local index
            for name, info in s3_index.get("models", {}).items():
                if name not in self._index["models"]:
                    self._index["models"][name] = info
                else:
                    # Merge versions
                    local_versions = set(self._index["models"][name].get("versions", []))
                    s3_versions = set(info.get("versions", []))
                    self._index["models"][name]["versions"] = list(local_versions | s3_versions)

            self._save_local_index()
            print("[ModelRegistry] Synced from S3")
        except Exception as e:
            print(f"[ModelRegistry] Sync failed: {e}")

    def sync_to_s3(self) -> None:
        """Sync the local index to S3."""
        if not self.s3:
            return

        try:
            s3_index_path = "registry/index.json"
            temp_path = self.local_cache / "temp_index.json"
            temp_path.write_text(json.dumps(self._index, indent=2))
            self.s3.upload_file(temp_path, s3_index_path)
            temp_path.unlink()
            print("[ModelRegistry] Synced to S3")
        except Exception as e:
            print(f"[ModelRegistry] Sync failed: {e}")


# Convenience functions
_registry: Optional[ModelRegistry] = None


def get_registry() -> ModelRegistry:
    """Get or create the global model registry instance."""
    global _registry
    if _registry is None:
        _registry = ModelRegistry()
    return _registry


def register_model(
    model_path: Union[str, Path],
    name: str,
    version: Optional[str] = None,
    metrics: Optional[Dict[str, float]] = None,
    aim_run: Optional[Run] = None,
    **kwargs
) -> ModelVersion:
    """Register a model (convenience function)."""
    return get_registry().register(
        model_path=model_path,
        name=name,
        version=version,
        metrics=metrics,
        aim_run=aim_run,
        **kwargs
    )


def load_model(name: str, version: str = "latest") -> Path:
    """Load a model (convenience function)."""
    return get_registry().load(name, version)


def list_models() -> List[ModelInfo]:
    """List all models (convenience function)."""
    return get_registry().list_models()
