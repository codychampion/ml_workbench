"""
Dataset Provenance Manifests.
=============================
Create lightweight JSON manifests for collections and annotations with git
traceability and optional Hydra config snapshots. Designed for syncing into the
knowledge base and Khoj search.
"""

import json
import os
import subprocess
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


def _run_git(args: list[str]) -> str:
    try:
        result = subprocess.run(
            ["git"] + args,
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.stdout.strip() if result.returncode == 0 else ""
    except Exception:
        return ""


def get_git_info() -> Dict[str, Any]:
    commit = _run_git(["rev-parse", "HEAD"])
    branch = _run_git(["rev-parse", "--abbrev-ref", "HEAD"])
    commit_message = _run_git(["log", "-1", "--format=%s"])
    commit_author = _run_git(["log", "-1", "--format=%an <%ae>"])
    commit_short = _run_git(["rev-parse", "--short", "HEAD"])
    dirty = bool(_run_git(["status", "--porcelain"]))
    return {
        "commit": commit,
        "commit_short": commit_short,
        "branch": branch,
        "message": commit_message,
        "author": commit_author,
        "dirty": dirty,
    }


def _default_manifest_dir() -> Path:
    return Path(os.environ.get("MANIFEST_DIR", "./knowledge/datasets/manifests"))


def _slugify(text: str) -> str:
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
    """Walk up directories looking for a collection_manifest.json."""
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
