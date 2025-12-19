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
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List

# Configuration
KNOWLEDGE_DIR = Path(os.environ.get("KNOWLEDGE_DIR", "./knowledge"))
AIM_REPO = os.environ.get("AIM_REPO", "./outputs/aim")

# Service URLs
ZOTERO_URL = os.environ.get("ZOTERO_URL", "http://localhost:8085")
FIFTYONE_URL = os.environ.get("FIFTYONE_URL", "http://localhost:5151")


def ensure_dirs():
    """Ensure knowledge directories exist."""
    for subdir in ["papers", "experiments", "datasets", "models", "results"]:
        (KNOWLEDGE_DIR / subdir).mkdir(parents=True, exist_ok=True)


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

    filename = f"{created.strftime('%Y%m%d')}-{slugify(experiment_name)}-{run_hash}.md"
    filepath = KNOWLEDGE_DIR / "experiments" / filename

    content = f"""---
type: experiment
name: "{experiment_name}"
aim_run_hash: "{run.hash}"
aim_url: "http://localhost:43800/runs/{run.hash}"
date: {created.strftime('%Y-%m-%d')}
status: completed
hyperparameters: {json.dumps(params)}
final_metrics: {json.dumps(metrics)}
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
