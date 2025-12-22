#!/usr/bin/env python3
"""
Knowledge Watcher - Auto-generate markdown when needed.

Monitors:
- AIM runs → knowledge/experiments/runs/
- PDFs in papers/pdfs/ → knowledge/papers/notes/
- Manifests in datasets/manifests/ → knowledge/datasets/cards/

Usage:
    python scripts/watcher.py              # Run once
    python scripts/watcher.py --daemon     # Run continuously
    python scripts/watcher.py --interval 60
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from aim import Repo
    AIM_AVAILABLE = True
except ImportError:
    AIM_AVAILABLE = False

# Paths
KNOWLEDGE_DIR = Path(os.environ.get("KNOWLEDGE_DIR", "./knowledge"))
AIM_REPO = Path(os.environ.get("AIM_REPO", "./outputs/aim"))
RUNS_DIR = KNOWLEDGE_DIR / "experiments" / "runs"
PAPERS_PDF_DIR = KNOWLEDGE_DIR / "papers" / "pdfs"
PAPERS_NOTES_DIR = KNOWLEDGE_DIR / "papers" / "notes"
DATASETS_MANIFESTS_DIR = KNOWLEDGE_DIR / "datasets" / "manifests"
DATASETS_CARDS_DIR = KNOWLEDGE_DIR / "datasets" / "cards"
STATE_FILE = KNOWLEDGE_DIR / ".watcher_state.json"


def load_state() -> dict:
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {"seen_runs": [], "seen_pdfs": [], "seen_manifests": [], "last_check": None}


def save_state(state: dict):
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(state, indent=2))


def fmt_table(data: dict) -> str:
    if not data:
        return "*None*"
    lines = ["| Key | Value |", "|-----|-------|"]
    for k, v in sorted(data.items()):
        v = f"{v:.4f}" if isinstance(v, float) else str(v)[:50]
        lines.append(f"| {k} | {v} |")
    return "\n".join(lines)


# === AIM Runs ===
def export_aim_run(run, output_dir: Path) -> Path:
    run_hash = run.hash[:8]
    exp = run.experiment or "default"
    created = datetime.fromtimestamp(run.created_at).strftime("%Y-%m-%d %H:%M")
    date = datetime.now().strftime("%Y-%m-%d")

    metrics, params = {}, {}
    try:
        for m in run.metrics():
            vals = list(run.get_metric(m).values)
            if vals:
                metrics[m] = vals[-1]
    except Exception:
        pass
    try:
        params = dict(run.get("hparams", {}))
    except Exception:
        pass

    tags = ", ".join(f'"{t}"' for t in list(run.tags)[:5]) if hasattr(run, 'tags') else ""

    content = f'''---
type: run-summary
run_id: "{run_hash}"
exp_id: "{exp}"
created: "{date}"
tags: [{tags}]
---

# Run: {run_hash}

**Experiment:** {exp} | **Created:** {created}

## Metrics
{fmt_table(metrics)}

## Parameters
{fmt_table(params)}

## Observations

## Next Steps
- [ ]
'''
    out = output_dir / f"{date}-{exp}-{run_hash}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(content)
    return out


def check_aim_runs(state: dict) -> list:
    if not AIM_AVAILABLE or not AIM_REPO.exists():
        return []
    exported = []
    try:
        repo = Repo(str(AIM_REPO))
        for run in repo.iter_runs():
            if run.hash not in state["seen_runs"]:
                path = export_aim_run(run, RUNS_DIR)
                state["seen_runs"].append(run.hash)
                exported.append(("aim", path))
    except Exception as e:
        print(f"[watcher] AIM error: {e}")
    return exported


# === PDFs ===
def create_paper_note(pdf_path: Path, output_dir: Path) -> Path:
    name = pdf_path.stem
    date = datetime.now().strftime("%Y-%m-%d")

    content = f'''---
type: paper
title: "{name}"
authors: []
year:
url:
pdf: "[[papers/pdfs/{pdf_path.name}]]"
tags: []
---

# {name}

## Summary

## Key Ideas
-

## Method

## Results

## Relevance

## Related
'''
    out = output_dir / f"{name}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    if not out.exists():  # Don't overwrite existing notes
        out.write_text(content)
        return out
    return None


def check_pdfs(state: dict) -> list:
    if not PAPERS_PDF_DIR.exists():
        return []
    exported = []
    for pdf in PAPERS_PDF_DIR.glob("*.pdf"):
        if pdf.name not in state["seen_pdfs"]:
            path = create_paper_note(pdf, PAPERS_NOTES_DIR)
            state["seen_pdfs"].append(pdf.name)
            if path:
                exported.append(("pdf", path))
    return exported


# === Dataset Manifests ===
def create_dataset_card(manifest_path: Path, output_dir: Path) -> Path:
    name = manifest_path.stem
    date = datetime.now().strftime("%Y-%m-%d")

    # Try to read manifest for stats
    stats = {}
    try:
        data = json.loads(manifest_path.read_text())
        if isinstance(data, list):
            stats["samples"] = len(data)
        elif isinstance(data, dict):
            stats = {k: v for k, v in data.items() if isinstance(v, (int, float, str))}
    except Exception:
        pass

    content = f'''---
type: dataset
dataset_id: "{name}"
created: "{date}"
version: "1.0"
manifest: "[[datasets/manifests/{manifest_path.name}]]"
tags: []
---

# Dataset: {name}

## Overview

## Location
- **Path:** s3://mlops-data/datasets/{name}/
- **Manifest:** datasets/manifests/{manifest_path.name}

## Statistics
{fmt_table(stats) if stats else "| Stat | Value |\n|------|-------|\n| Samples | |\n| Classes | |"}

## Splits
| Split | Samples | Notes |
|-------|---------|-------|
| train | | |
| val | | |
| test | | |

## Used In
'''
    out = output_dir / f"{name}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    if not out.exists():
        out.write_text(content)
        return out
    return None


def check_manifests(state: dict) -> list:
    if not DATASETS_MANIFESTS_DIR.exists():
        return []
    exported = []
    for manifest in DATASETS_MANIFESTS_DIR.glob("*.json"):
        if manifest.name not in state["seen_manifests"]:
            path = create_dataset_card(manifest, DATASETS_CARDS_DIR)
            state["seen_manifests"].append(manifest.name)
            if path:
                exported.append(("manifest", path))
    # Also check YAML manifests
    for manifest in DATASETS_MANIFESTS_DIR.glob("*.yaml"):
        if manifest.name not in state["seen_manifests"]:
            path = create_dataset_card(manifest, DATASETS_CARDS_DIR)
            state["seen_manifests"].append(manifest.name)
            if path:
                exported.append(("manifest", path))
    return exported


# === Main ===
def check_once(state: dict) -> list:
    """Run all checks once. Returns list of (type, path) tuples."""
    results = []
    results.extend(check_aim_runs(state))
    results.extend(check_pdfs(state))
    results.extend(check_manifests(state))
    state["last_check"] = datetime.now().isoformat()
    save_state(state)
    return results


def run_daemon(interval: int = 60):
    print(f"[watcher] Starting (interval: {interval}s)")
    print(f"[watcher] Watching: AIM={AIM_REPO}, PDFs={PAPERS_PDF_DIR}, Manifests={DATASETS_MANIFESTS_DIR}")

    state = load_state()
    results = check_once(state)
    for typ, path in results:
        print(f"[watcher] {typ}: {path.name}")

    try:
        while True:
            time.sleep(interval)
            results = check_once(state)
            for typ, path in results:
                print(f"[watcher] {typ}: {path.name}")
    except KeyboardInterrupt:
        print("\n[watcher] Stopped")


def main():
    parser = argparse.ArgumentParser(description="Knowledge watcher")
    parser.add_argument("--daemon", action="store_true", help="Run continuously")
    parser.add_argument("--interval", type=int, default=60, help="Check interval (seconds)")
    parser.add_argument("--reset", action="store_true", help="Reset state (re-process all)")
    args = parser.parse_args()

    if args.reset:
        if STATE_FILE.exists():
            STATE_FILE.unlink()
        print("[watcher] State reset")

    if args.daemon:
        run_daemon(args.interval)
    else:
        state = load_state()
        results = check_once(state)
        for typ, path in results:
            print(f"[watcher] {typ}: {path.name}")
        print(f"[watcher] Total: {len(results)} files")


if __name__ == "__main__":
    main()
