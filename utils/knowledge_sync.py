#!/usr/bin/env python3
"""
Knowledge Base Sync Utilities
==============================
Auto-generate Obsidian markdown files from MLOps services:
- AIM experiments → experiments/*.md
- FiftyOne datasets → datasets/*.md
- Zotero papers → papers/*.md
- Model registry → models/*.md

Usage:
    python -m utils.knowledge_sync                    # Sync all
    python -m utils.knowledge_sync --source aim       # Sync AIM only
    python -m utils.knowledge_sync --source fiftyone  # Sync FiftyOne only
    python -m utils.knowledge_sync --watch            # Watch for changes
"""

import os
import json
import argparse
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List

# Try to use Hydra config if available
try:
    from omegaconf import DictConfig, OmegaConf
    HYDRA_AVAILABLE = True
except ImportError:
    HYDRA_AVAILABLE = False

# Default configuration (fallback when Hydra not available)
KNOWLEDGE_DIR = Path(os.environ.get("KNOWLEDGE_DIR", "./knowledge"))
AIM_REPO = os.environ.get("AIM_REPO", "./outputs/aim")
DATA_DIR = Path(os.environ.get("DATA_DIR", "./data"))

# Service URLs
ZOTERO_URL = os.environ.get("ZOTERO_URL", "http://localhost:8085")
FIFTYONE_URL = os.environ.get("FIFTYONE_URL", "http://localhost:5151")

# Git repository URL (for commit links)
REPO_URL = os.environ.get("REPO_URL", "https://github.com/your-org/ml_workbench")


def get_knowledge_config(cfg: Optional[Any] = None) -> Dict[str, Any]:
    """
    Get knowledge base configuration from Hydra config or environment variables.

    Args:
        cfg: Optional Hydra config (DictConfig). If None, uses env vars.

    Returns:
        Dict with knowledge base configuration
    """
    if cfg and HYDRA_AVAILABLE and hasattr(cfg, 'infrastructure'):
        knowledge_cfg = cfg.infrastructure.knowledge
        return {
            'vault_dir': Path(knowledge_cfg.get('vault_dir', './knowledge')),
            'zotero_url': knowledge_cfg.zotero.get('url', ZOTERO_URL),
            'fiftyone_url': os.getenv('FIFTYONE_URL', 'http://localhost:5151'),
            'aim_repo': cfg.infrastructure.aim.get('repo', AIM_REPO),
            'data_dir': Path(cfg.paths.get('data', {}).get('collected', './data/collected')).parent,
            'sync_sources': knowledge_cfg.sync.get('sources', ['all']),
        }
    else:
        # Fallback to environment variables
        return {
            'vault_dir': KNOWLEDGE_DIR,
            'zotero_url': ZOTERO_URL,
            'fiftyone_url': FIFTYONE_URL,
            'aim_repo': AIM_REPO,
            'data_dir': DATA_DIR,
            'sync_sources': ['all'],
        }


def ensure_dirs():
    """Ensure knowledge directories exist."""
    for subdir in ["papers", "experiments", "datasets", "models", "results", "collections", "annotations"]:
        (KNOWLEDGE_DIR / subdir).mkdir(parents=True, exist_ok=True)


# =============================================================================
# Git Utilities
# =============================================================================

def get_git_info() -> Dict[str, Any]:
    """Get current git state for traceability."""
    def run_git(args: List[str]) -> str:
        try:
            result = subprocess.run(
                ["git"] + args,
                capture_output=True,
                text=True,
                timeout=10
            )
            return result.stdout.strip() if result.returncode == 0 else ""
        except Exception:
            return ""

    commit = run_git(["rev-parse", "HEAD"])
    commit_short = run_git(["rev-parse", "--short", "HEAD"])
    branch = run_git(["rev-parse", "--abbrev-ref", "HEAD"])
    commit_message = run_git(["log", "-1", "--format=%s"])
    commit_author = run_git(["log", "-1", "--format=%an <%ae>"])

    # Check if working directory is dirty
    dirty = bool(run_git(["status", "--porcelain"]))

    # Get list of files changed in last commit
    modified_files = run_git(["diff-tree", "--no-commit-id", "--name-only", "-r", "HEAD"])
    modified_list = modified_files.split("\n") if modified_files else []

    return {
        "commit": commit,
        "commit_short": commit_short,
        "branch": branch,
        "commit_message": commit_message,
        "commit_author": commit_author,
        "dirty": dirty,
        "modified_files": modified_list,
        "repo_url": REPO_URL,
        "commit_url": f"{REPO_URL}/commit/{commit}" if commit else "",
        "diff_from_main": f"{REPO_URL}/compare/main...{commit}" if commit else "",
    }


def get_git_info_for_commit(commit_hash: str) -> Dict[str, Any]:
    """Get git info for a specific commit."""
    def run_git(args: List[str]) -> str:
        try:
            result = subprocess.run(
                ["git"] + args,
                capture_output=True,
                text=True,
                timeout=10
            )
            return result.stdout.strip() if result.returncode == 0 else ""
        except Exception:
            return ""

    commit_short = commit_hash[:8] if len(commit_hash) >= 8 else commit_hash
    commit_message = run_git(["log", "-1", "--format=%s", commit_hash])
    commit_author = run_git(["log", "-1", "--format=%an <%ae>", commit_hash])
    modified_files = run_git(["diff-tree", "--no-commit-id", "--name-only", "-r", commit_hash])
    modified_list = modified_files.split("\n") if modified_files else []

    # Try to find which branch this commit is on
    branch = run_git(["branch", "--contains", commit_hash, "--format=%(refname:short)"])
    branch = branch.split("\n")[0] if branch else "unknown"

    return {
        "commit": commit_hash,
        "commit_short": commit_short,
        "branch": branch,
        "commit_message": commit_message,
        "commit_author": commit_author,
        "modified_files": modified_list,
        "repo_url": REPO_URL,
        "commit_url": f"{REPO_URL}/commit/{commit_hash}",
        "diff_from_main": f"{REPO_URL}/compare/main...{commit_hash}",
    }


def slugify(text: str) -> str:
    """Convert text to filename-safe slug."""
    return "".join(c if c.isalnum() or c in "-_" else "-" for c in text.lower()).strip("-")


# =============================================================================
# AIM Sync
# =============================================================================

def sync_aim_experiments():
    """Sync AIM experiments to knowledge base."""
    try:
        from aim import Repo
    except ImportError:
        print("AIM not installed, skipping experiment sync")
        return []

    try:
        repo = Repo(AIM_REPO)
    except Exception as e:
        print(f"Could not open AIM repo: {e}")
        return []

    synced = []
    for run in repo.iter_runs():
        try:
            filepath = create_experiment_from_aim(run)
            synced.append(filepath)
        except Exception as e:
            print(f"Error syncing run {run.hash}: {e}")

    print(f"Synced {len(synced)} experiments from AIM")
    return synced


def create_experiment_from_aim(run) -> Path:
    """Create experiment markdown from AIM run."""
    run_hash = run.hash[:8]
    experiment_name = run.experiment or "default"
    created = datetime.fromtimestamp(run.created_at)

    # Get metrics
    metrics = {}
    try:
        for metric_name in run.metrics():
            metric = run.get_metric(metric_name)
            if metric and hasattr(metric, 'values'):
                values = list(metric.values)
                if values:
                    metrics[metric_name] = values[-1]
    except:
        pass

    # Get params
    params = {}
    try:
        params = dict(run.get("hparams", {}))
    except:
        pass

    # Get git info - try from run metadata first, then fall back to current
    git_info = {}
    try:
        # AIM stores git info if available
        git_info = dict(run.get("git", {}))
    except:
        pass

    if not git_info.get("commit"):
        # Try to get from run's stored git hash
        try:
            git_hash = run.get("git_hash", "")
            if git_hash:
                git_info = get_git_info_for_commit(git_hash)
        except:
            pass

    # Fall back to current git state if nothing found
    if not git_info.get("commit"):
        git_info = get_git_info()
        git_info["note"] = "Using current git state - commit at run time not recorded"

    filename = f"{created.strftime('%Y%m%d')}-{slugify(experiment_name)}-{run_hash}.md"
    filepath = KNOWLEDGE_DIR / "experiments" / filename

    # Format modified files for display
    modified_files_str = "\n".join(git_info.get("modified_files", []))

    content = f"""---
type: experiment
name: "{experiment_name}"
aim_run_hash: "{run.hash}"
aim_url: "http://localhost:43800/runs/{run.hash}"
date: {created.strftime('%Y-%m-%d')}
status: completed
hyperparameters: {json.dumps(params)}
final_metrics: {json.dumps(metrics)}

# Code Traceability
git:
  commit: "{git_info.get('commit', '')}"
  commit_short: "{git_info.get('commit_short', '')}"
  branch: "{git_info.get('branch', '')}"
  commit_url: "{git_info.get('commit_url', '')}"
  commit_message: "{git_info.get('commit_message', '')}"
  author: "{git_info.get('commit_author', '')}"
  dirty: {str(git_info.get('dirty', False)).lower()}
  diff_from_main: "{git_info.get('diff_from_main', '')}"

tags:
  - experiment
  - auto-generated
---

# Experiment: {experiment_name}

**Run ID:** `{run.hash}`
**Date:** {created.strftime('%Y-%m-%d %H:%M')}

## Metrics

| Metric | Value |
|--------|-------|
"""
    for k, v in metrics.items():
        val = f"{v:.4f}" if isinstance(v, float) else str(v)
        content += f"| {k} | {val} |\n"

    content += """
## Hyperparameters

| Parameter | Value |
|-----------|-------|
"""
    for k, v in params.items():
        content += f"| {k} | `{v}` |\n"

    content += f"""
## Code Changes

**Commit:** [`{git_info.get('commit_short', 'unknown')}`]({git_info.get('commit_url', '')}) on branch `{git_info.get('branch', 'unknown')}`
**Message:** {git_info.get('commit_message', 'N/A')}
**Author:** {git_info.get('commit_author', 'N/A')}

### Modified Files
```
{modified_files_str or 'No file changes recorded'}
```

### Reproduce
```bash
# Checkout exact code state
git checkout {git_info.get('commit', 'HEAD')}

# Run experiment
python -m pipelines.train.train_lora experiment={experiment_name}
```

## Notes

*Auto-synced from AIM on {datetime.now().strftime('%Y-%m-%d %H:%M')}*

## Related
- [[]]
"""

    filepath.write_text(content)
    return filepath


# =============================================================================
# Zotero Sync
# =============================================================================

def sync_zotero_papers():
    """Sync papers from Zotero service."""
    try:
        import requests
        response = requests.get(f"{ZOTERO_URL}/api/papers", timeout=10)
        if response.status_code != 200:
            print(f"Zotero API error: {response.status_code}")
            return []
        papers = response.json().get("papers", [])
    except Exception as e:
        print(f"Could not connect to Zotero: {e}")
        return []

    synced = []
    for paper in papers:
        try:
            filepath = create_paper_from_zotero(paper)
            synced.append(filepath)
        except Exception as e:
            print(f"Error syncing paper {paper.get('id')}: {e}")

    print(f"Synced {len(synced)} papers from Zotero")
    return synced


def create_paper_from_zotero(paper: Dict[str, Any]) -> Path:
    """Create paper markdown from Zotero entry."""
    paper_id = paper.get("id", "unknown")
    title = paper.get("title", "Untitled")
    authors = paper.get("authors", "Unknown")
    year = paper.get("year", "")
    doi = paper.get("doi", "")
    abstract = paper.get("abstract", "")
    tags = json.loads(paper.get("tags", "[]")) if isinstance(paper.get("tags"), str) else paper.get("tags", [])

    filename = f"{year}-{slugify(title)[:50]}.md"
    filepath = KNOWLEDGE_DIR / "papers" / filename

    content = f"""---
type: paper
title: "{title}"
authors: "{authors}"
year: {year}
doi: "{doi}"
status: unread
tags:
  - paper
  - auto-generated
{chr(10).join(f'  - {t}' for t in tags)}
---

# {title}

**Authors:** {authors}
**Year:** {year}
{"**DOI:** [" + doi + "](https://doi.org/" + doi + ")" if doi else ""}

## Abstract

{abstract or "*No abstract available*"}

## Summary

*Add your summary here*

## Key Contributions

1.
2.

## Notes

*Auto-synced from Zotero on {datetime.now().strftime('%Y-%m-%d %H:%M')}*

## Related
- [[]]
"""

    filepath.write_text(content)
    return filepath


# =============================================================================
# FiftyOne Sync
# =============================================================================

def sync_fiftyone_datasets():
    """Sync FiftyOne datasets to knowledge base."""
    try:
        import fiftyone as fo
        datasets = fo.list_datasets()
    except ImportError:
        print("FiftyOne not installed, skipping dataset sync")
        return []
    except Exception as e:
        print(f"FiftyOne error: {e}")
        return []

    synced = []
    for name in datasets:
        try:
            dataset = fo.load_dataset(name)
            filepath = create_dataset_from_fiftyone(dataset)
            synced.append(filepath)
        except Exception as e:
            print(f"Error syncing dataset {name}: {e}")

    print(f"Synced {len(synced)} datasets from FiftyOne")
    return synced


def create_dataset_from_fiftyone(dataset) -> Path:
    """Create dataset markdown from FiftyOne dataset."""
    name = dataset.name
    num_samples = len(dataset)
    media_type = getattr(dataset, 'media_type', 'unknown')

    # Get field info
    fields = list(dataset.get_field_schema().keys()) if hasattr(dataset, 'get_field_schema') else []

    filename = f"{slugify(name)}.md"
    filepath = KNOWLEDGE_DIR / "datasets" / filename

    content = f"""---
type: dataset
name: "{name}"
fiftyone_name: "{name}"
fiftyone_url: "http://localhost:5151/datasets/{name}"
total_samples: {num_samples}
modality: {media_type}
tags:
  - dataset
  - auto-generated
---

# Dataset: {name}

**Samples:** {num_samples}
**Media Type:** {media_type}

## Fields

{chr(10).join(f'- `{f}`' for f in fields)}

## Usage

```python
import fiftyone as fo
dataset = fo.load_dataset("{name}")
print(dataset)
```

## Notes

*Auto-synced from FiftyOne on {datetime.now().strftime('%Y-%m-%d %H:%M')}*

## Related
- [[]]
"""

    filepath.write_text(content)
    return filepath


# =============================================================================
# Collection & Annotation Sync
# =============================================================================

def sync_collections():
    """Sync collection manifests to knowledge base."""
    try:
        from utils.manifest import find_all_collections
    except ImportError:
        print("Manifest utilities not available, skipping collection sync")
        return []

    collections = find_all_collections(DATA_DIR)
    synced = []

    for collection in collections:
        try:
            filepath = create_collection_markdown(collection)
            synced.append(filepath)
        except Exception as e:
            print(f"Error syncing collection {collection.get('id')}: {e}")

    print(f"Synced {len(synced)} collections")
    return synced


def create_collection_markdown(manifest: Dict[str, Any]) -> Path:
    """Create markdown file from collection manifest."""
    collection_id = manifest["id"]
    created = datetime.fromisoformat(manifest["created_at"])
    source = manifest.get("source", {})
    storage = manifest.get("storage", {})
    git = manifest.get("git", {})
    annotations = manifest.get("annotations", [])

    filename = f"{slugify(collection_id)}.md"
    filepath = KNOWLEDGE_DIR / "collections" / filename

    # Format source metadata
    source_metadata = source.get("metadata", {})
    source_meta_str = "\n".join(f"  - **{k}**: `{v}`" for k, v in source_metadata.items())

    # Format annotations
    if annotations:
        annotations_str = "\n".join(
            f"- `{ann['id']}`: {ann['type']} using **{ann['model']}** ({ann.get('created_at', 'N/A')})"
            for ann in annotations
        )
    else:
        annotations_str = "*No annotations yet*"

    content = f"""---
type: collection
collection_id: "{collection_id}"
source_type: "{source.get('type', 'unknown')}"
source_url: "{source.get('url', '')}"
date: {created.strftime('%Y-%m-%d')}
total_images: {storage.get('images', 0)}
total_videos: {storage.get('videos', 0)}
total_files: {storage.get('total_files', 0)}

# Code Traceability
git:
  commit: "{git.get('commit', '')}"
  commit_short: "{git.get('commit_short', '')}"
  branch: "{git.get('branch', '')}"
  dirty: {str(git.get('dirty', False)).lower()}

tags:
  - collection
  - {source.get('type', 'data')}
  - auto-generated
---

# Collection: {collection_id}

**ID:** `{collection_id}`
**Created:** {created.strftime('%Y-%m-%d %H:%M')}
**Source:** {source.get('type', 'unknown')} - {source.get('url', 'N/A')}

## Contents

- **Images:** {storage.get('images', 0)}
- **Videos:** {storage.get('videos', 0)}
- **Total Files:** {storage.get('total_files', 0)}
- **Path:** `{storage.get('path', 'N/A')}`

## Source Details

{source_meta_str or '*No additional metadata*'}

## Annotations

{annotations_str}

## Code State

**Commit:** `{git.get('commit_short', 'unknown')}` on branch `{git.get('branch', 'unknown')}`
**Working Directory:** {"⚠️ Modified (dirty)" if git.get('dirty') else "✓ Clean"}

## Usage

```bash
# View collection
ls {storage.get('path', '.')}

# Caption this collection
docker-compose run --rm annotate python -m pipelines.annotate.caption \\
    --input-dir {storage.get('path', '.')} \\
    --model blip-base

# Create FiftyOne dataset
docker-compose run --rm annotate python -m pipelines.annotate.create_dataset \\
    --input-dir {storage.get('path', '.')} \\
    --name {collection_id}
```

## Notes

*Auto-synced from manifest on {datetime.now().strftime('%Y-%m-%d %H:%M')}*

## Related
- [[]]
"""

    filepath.write_text(content)
    return filepath


def sync_annotations():
    """Sync annotation manifests to knowledge base."""
    try:
        from utils.manifest import find_all_annotations
    except ImportError:
        print("Manifest utilities not available, skipping annotation sync")
        return []

    annotations = find_all_annotations(DATA_DIR)
    synced = []

    for annotation in annotations:
        try:
            filepath = create_annotation_markdown(annotation)
            synced.append(filepath)
        except Exception as e:
            print(f"Error syncing annotation {annotation.get('id')}: {e}")

    print(f"Synced {len(synced)} annotations")
    return synced


def create_annotation_markdown(manifest: Dict[str, Any]) -> Path:
    """Create markdown file from annotation manifest."""
    annotation_id = manifest["id"]
    created = datetime.fromisoformat(manifest["created_at"])
    collection = manifest.get("collection", {})
    annotation = manifest.get("annotation", {})
    storage = manifest.get("storage", {})
    git = manifest.get("git", {})

    filename = f"{slugify(annotation_id)}.md"
    filepath = KNOWLEDGE_DIR / "annotations" / filename

    # Format annotation metadata
    ann_metadata = annotation.get("metadata", {})
    ann_meta_str = "\n".join(f"  - **{k}**: `{v}`" for k, v in ann_metadata.items())

    content = f"""---
type: annotation
annotation_id: "{annotation_id}"
collection_id: "{collection.get('id', 'unknown')}"
annotation_type: "{annotation.get('type', 'unknown')}"
model: "{annotation.get('model', 'unknown')}"
date: {created.strftime('%Y-%m-%d')}
total_annotations: {storage.get('annotation_files', 0)}

# Code Traceability
git:
  commit: "{git.get('commit', '')}"
  commit_short: "{git.get('commit_short', '')}"
  branch: "{git.get('branch', '')}"
  dirty: {str(git.get('dirty', False)).lower()}

tags:
  - annotation
  - {annotation.get('type', 'caption')}
  - auto-generated
---

# Annotation: {annotation_id}

**ID:** `{annotation_id}`
**Created:** {created.strftime('%Y-%m-%d %H:%M')}
**Type:** {annotation.get('type', 'unknown')}
**Model:** `{annotation.get('model', 'unknown')}`

## Parent Collection

**Collection ID:** `{collection.get('id', 'unknown')}`
**Path:** `{collection.get('path', 'N/A')}`

See: [[{slugify(collection.get('id', 'unknown'))}]]

## Details

- **Annotation Files:** {storage.get('annotation_files', 0)}
- **Storage Path:** `{storage.get('path', 'N/A')}`

## Metadata

{ann_meta_str or '*No additional metadata*'}

## Code State

**Commit:** `{git.get('commit_short', 'unknown')}` on branch `{git.get('branch', 'unknown')}`
**Working Directory:** {"⚠️ Modified (dirty)" if git.get('dirty') else "✓ Clean"}

## Usage

```bash
# View annotations
ls {storage.get('path', '.')}

# Train on this dataset
docker-compose run --rm train python -m pipelines.train.finetune \\
    --dataset {storage.get('path', '.')} \\
    --epochs 3
```

## Notes

*Auto-synced from manifest on {datetime.now().strftime('%Y-%m-%d %H:%M')}*

## Related
- [[{slugify(collection.get('id', 'unknown'))}]]
"""

    filepath.write_text(content)
    return filepath


# =============================================================================
# Main
# =============================================================================

def sync_all():
    """Sync all sources."""
    ensure_dirs()
    sync_aim_experiments()
    sync_zotero_papers()
    sync_fiftyone_datasets()
    sync_collections()
    sync_annotations()


def main():
    parser = argparse.ArgumentParser(description="Sync MLOps services to knowledge base")
    parser.add_argument("--source", choices=["aim", "zotero", "fiftyone", "collections", "annotations", "all"], default="all")
    parser.add_argument("--watch", action="store_true", help="Watch for changes")
    parser.add_argument("--interval", type=int, default=60, help="Watch interval in seconds")

    args = parser.parse_args()
    ensure_dirs()

    if args.watch:
        import time
        print(f"Watching for changes (interval: {args.interval}s)...")
        while True:
            try:
                sync_all()
                time.sleep(args.interval)
            except KeyboardInterrupt:
                break
    else:
        if args.source == "all":
            sync_all()
        elif args.source == "aim":
            sync_aim_experiments()
        elif args.source == "zotero":
            sync_zotero_papers()
        elif args.source == "fiftyone":
            sync_fiftyone_datasets()
        elif args.source == "collections":
            sync_collections()
        elif args.source == "annotations":
            sync_annotations()


if __name__ == "__main__":
    main()
