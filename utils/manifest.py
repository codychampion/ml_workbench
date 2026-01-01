"""
Dataset Provenance Manifests.
=============================
Create lightweight JSON manifests for collections and annotations with git
traceability and optional Hydra config snapshots. Designed for syncing into the
knowledge base and Khoj search.
"""

import json
import os
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from utils.git_utils import get_git_info


def _default_manifest_dir() -> Path:
    """
    Get default manifest directory from environment or use default path.

    Returns:
        Path to manifest directory
    """
    return Path(os.environ.get("MANIFEST_DIR", "./knowledge/datasets/manifests"))


def _slugify(text: str) -> str:
    """
    Convert text to slug format (lowercase, alphanumeric + hyphens/underscores).

    Args:
        text: Text to slugify

    Returns:
        Slugified text suitable for filenames
    """
    return "".join(c if c.isalnum() or c in "-_" else "-" for c in text.lower()).strip("-")


@dataclass
class CollectionManifest:
    id: str
    type: str
    name: str
    created_at: str
    source: Dict[str, Any]
    output_dir: str
    counts: Dict[str, Any]
    git: Dict[str, Any]
    config: Optional[Dict[str, Any]] = None


@dataclass
class AnnotationManifest:
    id: str
    type: str
    name: str
    created_at: str
    parent_collection_id: Optional[str]
    input_dir: str
    output_dir: str
    params: Dict[str, Any]
    counts: Dict[str, Any]
    git: Dict[str, Any]
    config: Optional[Dict[str, Any]] = None


def write_manifest(manifest: Dict[str, Any], manifest_dir: Optional[Path] = None, sidecar: Optional[Path] = None) -> Path:
    """
    Write manifest to knowledge base and optionally create sidecar copy.

    Args:
        manifest: Manifest dictionary to write
        manifest_dir: Directory for manifest (defaults to knowledge/datasets/manifests)
        sidecar: Optional path for sidecar manifest in data directory

    Returns:
        Path to written manifest file
    """
    manifest_dir = manifest_dir or _default_manifest_dir()
    manifest_dir.mkdir(parents=True, exist_ok=True)
    target = manifest_dir / f"{manifest['id']}.json"
    target.write_text(json.dumps(manifest, indent=2))
    if sidecar:
        sidecar.parent.mkdir(parents=True, exist_ok=True)
        sidecar.write_text(json.dumps(manifest, indent=2))
    return target


def record_collection_manifest(
    name: str,
    output_dir: Path,
    source: Dict[str, Any],
    counts: Dict[str, Any],
    cfg: Optional[Any] = None,
) -> Path:
    """
    Create and write a collection manifest for dataset provenance.

    Manifests track:
    - Collection source and parameters
    - Sample counts and statistics
    - Git state for reproducibility
    - Optional Hydra config snapshot

    Args:
        name: Human-readable collection name
        output_dir: Directory where data was collected
        source: Source information (type, URL, parameters)
        counts: Sample counts and statistics
        cfg: Optional Hydra DictConfig for config snapshot

    Returns:
        Path to written manifest file
    """
    ts = datetime.now()
    manifest_id = f"col-{ts.strftime('%Y%m%d-%H%M%S')}-{_slugify(name) or 'collection'}"
    config_snapshot = None
    if cfg is not None:
        try:
            from omegaconf import OmegaConf
            config_snapshot = OmegaConf.to_container(cfg, resolve=True)
        except Exception:
            pass
    manifest = CollectionManifest(
        id=manifest_id,
        type="collection",
        name=name,
        created_at=ts.isoformat(),
        source=source,
        output_dir=str(output_dir),
        counts=counts,
        git=get_git_info(),
        config=config_snapshot,
    )
    sidecar = output_dir / "collection_manifest.json"
    return write_manifest(asdict(manifest), sidecar=sidecar)


def find_parent_collection_id(path: Path) -> Optional[str]:
    """
    Walk up directory tree looking for a collection_manifest.json.

    This allows annotation/processing steps to link back to their source collection.

    Args:
        path: Starting directory path

    Returns:
        Collection ID if found, None otherwise
    """
    for candidate in [path] + list(path.parents):
        manifest_path = candidate / "collection_manifest.json"
        if manifest_path.exists():
            try:
                data = json.loads(manifest_path.read_text())
                return data.get("id")
            except Exception:
                continue
    return None


def record_annotation_manifest(
    name: str,
    input_dir: Path,
    output_dir: Path,
    params: Dict[str, Any],
    counts: Dict[str, Any],
    parent_collection_id: Optional[str] = None,
    cfg: Optional[Any] = None,
) -> Path:
    """
    Create and write an annotation manifest for processing provenance.

    Annotation manifests track data transformations (captioning, labeling, etc.)
    and link back to source collections for full data lineage.

    Args:
        name: Human-readable annotation task name
        input_dir: Directory containing input data
        output_dir: Directory where annotations/outputs were written
        params: Processing parameters (model, settings, etc.)
        counts: Processing statistics (successful, failed, etc.)
        parent_collection_id: Optional ID of source collection
        cfg: Optional Hydra DictConfig for config snapshot

    Returns:
        Path to written manifest file
    """
    ts = datetime.now()
    manifest_id = f"ann-{ts.strftime('%Y%m%d-%H%M%S')}-{_slugify(name) or 'annotation'}"
    config_snapshot = None
    if cfg is not None:
        try:
            from omegaconf import OmegaConf
            config_snapshot = OmegaConf.to_container(cfg, resolve=True)
        except Exception:
            pass
    manifest = AnnotationManifest(
        id=manifest_id,
        type="annotation",
        name=name,
        created_at=ts.isoformat(),
        parent_collection_id=parent_collection_id,
        input_dir=str(input_dir),
        output_dir=str(output_dir),
        params=params,
        counts=counts,
        git=get_git_info(),
        config=config_snapshot,
    )
    sidecar = output_dir / "annotation_manifest.json"
    return write_manifest(asdict(manifest), sidecar=sidecar)
