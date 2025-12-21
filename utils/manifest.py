#!/usr/bin/env python3
"""
Dataset Manifest System
=======================
Track collections and annotations with unique IDs and relationships.

Collections are raw data downloads.
Annotations are processing runs on collections (e.g., different caption models).

One collection can have multiple annotation datasets.

Usage:
    from utils.manifest import create_collection_manifest, create_annotation_manifest

    # Create collection manifest
    manifest = create_collection_manifest(
        output_dir="./data/collected/earthporn",
        source="reddit",
        source_url="https://reddit.com/r/earthporn",
        metadata={"subreddit": "earthporn", "limit": 100}
    )

    # Create annotation manifest
    annotation = create_annotation_manifest(
        collection_id=manifest["id"],
        annotation_type="caption",
        model="blip-base",
        output_dir="./data/annotated/earthporn_blip",
        metadata={"num_captions": 95}
    )
"""

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
import hashlib
import os

# Try to use Hydra config if available
try:
    from omegaconf import DictConfig, OmegaConf
    HYDRA_AVAILABLE = True
except ImportError:
    HYDRA_AVAILABLE = False


def get_manifest_config(cfg: Optional[Any] = None) -> Dict[str, Any]:
    """
    Get manifest configuration from Hydra config or environment variables.

    Args:
        cfg: Optional Hydra config (DictConfig). If None, uses env vars.

    Returns:
        Dict with manifest configuration
    """
    if cfg and HYDRA_AVAILABLE and hasattr(cfg, 'infrastructure'):
        manifest_cfg = cfg.infrastructure.manifest
        return {
            'auto_create': manifest_cfg.get('auto_create', True),
            'git_tracking': manifest_cfg.get('git_tracking', True),
            'version': manifest_cfg.get('version', '1.0'),
            'id_prefix': {
                'collection': manifest_cfg.id_prefix.get('collection', 'col-'),
                'annotation': manifest_cfg.id_prefix.get('annotation', 'ann-'),
            }
        }
    else:
        # Fallback to environment variables
        return {
            'auto_create': os.getenv('MANIFEST_AUTO_CREATE', 'true').lower() == 'true',
            'git_tracking': os.getenv('MANIFEST_GIT_TRACKING', 'true').lower() == 'true',
            'version': os.getenv('MANIFEST_VERSION', '1.0'),
            'id_prefix': {
                'collection': os.getenv('MANIFEST_COLLECTION_PREFIX', 'col-'),
                'annotation': os.getenv('MANIFEST_ANNOTATION_PREFIX', 'ann-'),
            }
        }


def generate_id(prefix: str = "", cfg: Optional[Any] = None) -> str:
    """
    Generate a unique ID with optional prefix.

    Args:
        prefix: ID prefix (e.g., 'col-', 'ann-'). If empty, uses config defaults.
        cfg: Optional Hydra config for reading prefix from config

    Returns:
        Unique ID in format: prefix-YYYYMMDD-hash
    """
    unique = str(uuid.uuid4())[:8]
    timestamp = datetime.now().strftime("%Y%m%d")
    return f"{prefix}{timestamp}-{unique}" if prefix else f"{timestamp}-{unique}"


def hash_params(params: Dict[str, Any]) -> str:
    """Create a deterministic hash of parameters."""
    # Sort keys for consistent hashing
    sorted_json = json.dumps(params, sort_keys=True)
    return hashlib.sha256(sorted_json.encode()).hexdigest()[:12]


# =============================================================================
# Collection Manifests
# =============================================================================

def create_collection_manifest(
    output_dir: Path,
    source: str,
    source_url: str,
    metadata: Optional[Dict[str, Any]] = None,
    manifest_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Create a collection manifest.

    Args:
        output_dir: Directory where collection is stored
        source: Source type (reddit, url, filesystem, etc.)
        source_url: Original source URL
        metadata: Additional metadata (subreddit, filters, etc.)
        manifest_id: Optional custom ID (otherwise auto-generated)

    Returns:
        Collection manifest dict
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Count files
    image_extensions = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp"}
    video_extensions = {".mp4", ".webm", ".mov", ".avi"}

    all_files = list(output_dir.rglob("*")) if output_dir.exists() else []
    images = [f for f in all_files if f.suffix.lower() in image_extensions]
    videos = [f for f in all_files if f.suffix.lower() in video_extensions]

    # Generate or use provided ID
    collection_id = manifest_id or generate_id("col-")

    manifest = {
        "id": collection_id,
        "type": "collection",
        "version": "1.0",
        "created_at": datetime.now().isoformat(),

        # Source info
        "source": {
            "type": source,
            "url": source_url,
            "metadata": metadata or {}
        },

        # Storage
        "storage": {
            "path": str(output_dir.absolute()),
            "total_files": len(all_files),
            "images": len(images),
            "videos": len(videos)
        },

        # Traceability
        "git": get_git_info(),

        # Relationships
        "annotations": []  # Will be populated as annotations are created
    }

    # Save manifest
    manifest_file = output_dir / "collection_manifest.json"
    manifest_file.write_text(json.dumps(manifest, indent=2))

    print(f"📦 Collection manifest created: {collection_id}")
    print(f"   Path: {manifest_file}")

    return manifest


def load_collection_manifest(path: Path) -> Optional[Dict[str, Any]]:
    """Load a collection manifest from directory or file."""
    path = Path(path)

    if path.is_dir():
        manifest_file = path / "collection_manifest.json"
    else:
        manifest_file = path

    if not manifest_file.exists():
        return None

    return json.loads(manifest_file.read_text())


def update_collection_manifest(
    collection_path: Path,
    updates: Dict[str, Any]
) -> Dict[str, Any]:
    """Update an existing collection manifest."""
    manifest = load_collection_manifest(collection_path)
    if not manifest:
        raise ValueError(f"No collection manifest found at {collection_path}")

    manifest.update(updates)
    manifest["updated_at"] = datetime.now().isoformat()

    manifest_file = Path(collection_path) / "collection_manifest.json"
    manifest_file.write_text(json.dumps(manifest, indent=2))

    return manifest


# =============================================================================
# Annotation Manifests
# =============================================================================

def create_annotation_manifest(
    collection_id: str,
    annotation_type: str,
    model: str,
    output_dir: Path,
    metadata: Optional[Dict[str, Any]] = None,
    collection_path: Optional[Path] = None,
    manifest_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Create an annotation manifest.

    Args:
        collection_id: ID of parent collection
        annotation_type: Type of annotation (caption, label, segmentation, etc.)
        model: Model/tool used for annotation
        output_dir: Directory where annotations are stored
        metadata: Additional metadata (accuracy, parameters, etc.)
        collection_path: Path to collection (for updating parent manifest)
        manifest_id: Optional custom ID (otherwise auto-generated)

    Returns:
        Annotation manifest dict
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate annotation-specific ID
    annotation_id = manifest_id or generate_id("ann-")

    # Count annotation files
    annotation_files = []
    if output_dir.exists():
        # Common annotation file patterns
        patterns = ["*.txt", "*.json", "*.xml", "*.jsonl"]
        for pattern in patterns:
            annotation_files.extend(output_dir.rglob(pattern))

    manifest = {
        "id": annotation_id,
        "type": "annotation",
        "version": "1.0",
        "created_at": datetime.now().isoformat(),

        # Parent relationship
        "collection": {
            "id": collection_id,
            "path": str(collection_path.absolute()) if collection_path else None
        },

        # Annotation details
        "annotation": {
            "type": annotation_type,
            "model": model,
            "metadata": metadata or {}
        },

        # Storage
        "storage": {
            "path": str(output_dir.absolute()),
            "annotation_files": len(annotation_files)
        },

        # Traceability
        "git": get_git_info()
    }

    # Save manifest
    manifest_file = output_dir / "annotation_manifest.json"
    manifest_file.write_text(json.dumps(manifest, indent=2))

    print(f"🏷️  Annotation manifest created: {annotation_id}")
    print(f"   Parent collection: {collection_id}")
    print(f"   Path: {manifest_file}")

    # Update parent collection manifest if path provided
    if collection_path:
        try:
            collection_manifest = load_collection_manifest(collection_path)
            if collection_manifest:
                if "annotations" not in collection_manifest:
                    collection_manifest["annotations"] = []

                collection_manifest["annotations"].append({
                    "id": annotation_id,
                    "type": annotation_type,
                    "model": model,
                    "path": str(output_dir.absolute()),
                    "created_at": datetime.now().isoformat()
                })

                manifest_file = Path(collection_path) / "collection_manifest.json"
                manifest_file.write_text(json.dumps(collection_manifest, indent=2))
                print(f"   ✓ Updated parent collection manifest")
        except Exception as e:
            print(f"   ⚠ Could not update parent manifest: {e}")

    return manifest


def load_annotation_manifest(path: Path) -> Optional[Dict[str, Any]]:
    """Load an annotation manifest from directory or file."""
    path = Path(path)

    if path.is_dir():
        manifest_file = path / "annotation_manifest.json"
    else:
        manifest_file = path

    if not manifest_file.exists():
        return None

    return json.loads(manifest_file.read_text())


# =============================================================================
# Manifest Discovery
# =============================================================================

def find_all_collections(root_dir: Path = Path("./data")) -> List[Dict[str, Any]]:
    """Find all collection manifests in a directory tree."""
    root_dir = Path(root_dir)
    collections = []

    for manifest_file in root_dir.rglob("collection_manifest.json"):
        try:
            manifest = json.loads(manifest_file.read_text())
            manifest["_manifest_path"] = str(manifest_file)
            collections.append(manifest)
        except Exception as e:
            print(f"Error loading {manifest_file}: {e}")

    return collections


def find_all_annotations(root_dir: Path = Path("./data")) -> List[Dict[str, Any]]:
    """Find all annotation manifests in a directory tree."""
    root_dir = Path(root_dir)
    annotations = []

    for manifest_file in root_dir.rglob("annotation_manifest.json"):
        try:
            manifest = json.loads(manifest_file.read_text())
            manifest["_manifest_path"] = str(manifest_file)
            annotations.append(manifest)
        except Exception as e:
            print(f"Error loading {manifest_file}: {e}")

    return annotations


def get_collection_tree(root_dir: Path = Path("./data")) -> Dict[str, Any]:
    """Get a hierarchical view of collections and their annotations."""
    collections = find_all_collections(root_dir)
    annotations = find_all_annotations(root_dir)

    # Build tree
    tree = {}
    for collection in collections:
        col_id = collection["id"]
        tree[col_id] = {
            **collection,
            "annotations": []
        }

    # Add annotations to their parent collections
    for annotation in annotations:
        col_id = annotation.get("collection", {}).get("id")
        if col_id and col_id in tree:
            tree[col_id]["annotations"].append(annotation)

    return tree


# =============================================================================
# Git Integration
# =============================================================================

def get_git_info() -> Dict[str, Any]:
    """Get current git state for traceability."""
    import subprocess

    def run_git(args: List[str]) -> str:
        try:
            result = subprocess.run(
                ["git"] + args,
                capture_output=True,
                text=True,
                timeout=10,
                cwd=Path(__file__).parent.parent
            )
            return result.stdout.strip() if result.returncode == 0 else ""
        except Exception:
            return ""

    commit = run_git(["rev-parse", "HEAD"])
    commit_short = run_git(["rev-parse", "--short", "HEAD"])
    branch = run_git(["rev-parse", "--abbrev-ref", "HEAD"])

    return {
        "commit": commit,
        "commit_short": commit_short,
        "branch": branch,
        "dirty": bool(run_git(["status", "--porcelain"]))
    }


# =============================================================================
# CLI
# =============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Dataset manifest utilities")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # List collections
    list_parser = subparsers.add_parser("list", help="List all collections and annotations")
    list_parser.add_argument("--root", type=Path, default=Path("./data"))

    # Show collection tree
    tree_parser = subparsers.add_parser("tree", help="Show collection hierarchy")
    tree_parser.add_argument("--root", type=Path, default=Path("./data"))

    args = parser.parse_args()

    if args.command == "list":
        collections = find_all_collections(args.root)
        annotations = find_all_annotations(args.root)

        print(f"\n📦 Collections ({len(collections)}):")
        for col in collections:
            print(f"  {col['id']}: {col['source']['type']} - {col['storage']['images']} images")

        print(f"\n🏷️  Annotations ({len(annotations)}):")
        for ann in annotations:
            print(f"  {ann['id']}: {ann['annotation']['type']} using {ann['annotation']['model']}")
            print(f"    → collection: {ann['collection']['id']}")

    elif args.command == "tree":
        tree = get_collection_tree(args.root)

        print("\n📦 Collection Tree:\n")
        for col_id, data in tree.items():
            print(f"  {col_id}")
            print(f"    Source: {data['source']['type']}")
            print(f"    Images: {data['storage']['images']}")
            print(f"    Annotations: {len(data['annotations'])}")
            for ann in data["annotations"]:
                print(f"      └─ {ann['id']}: {ann['annotation']['type']} ({ann['annotation']['model']})")
            print()


if __name__ == "__main__":
    main()
