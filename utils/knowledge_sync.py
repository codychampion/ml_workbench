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

# Configuration
KNOWLEDGE_DIR = Path(os.environ.get("KNOWLEDGE_DIR", "./knowledge"))
AIM_REPO = os.environ.get("AIM_REPO", "./outputs/aim")

# Service URLs
ZOTERO_URL = os.environ.get("ZOTERO_URL", "http://localhost:8085")
FIFTYONE_URL = os.environ.get("FIFTYONE_URL", "http://localhost:5151")

# Git repository URL (for commit links)
REPO_URL = os.environ.get("REPO_URL", "https://github.com/your-org/ml_workbench")


def ensure_dirs():
    """Ensure knowledge directories exist."""
    for subdir in ["papers", "experiments", "datasets", "models", "results"]:
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
# Main
# =============================================================================

def sync_all():
    """Sync all sources."""
    ensure_dirs()
    sync_aim_experiments()
    sync_zotero_papers()
    sync_fiftyone_datasets()


def main():
    parser = argparse.ArgumentParser(description="Sync MLOps services to knowledge base")
    parser.add_argument("--source", choices=["aim", "zotero", "fiftyone", "all"], default="all")
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


if __name__ == "__main__":
    main()
