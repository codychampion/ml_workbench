#!/usr/bin/env python3
"""
AIM Report Ingestion for Knowledge Base
========================================
Exports AIM experiment data and creates Obsidian-compatible markdown notes
for ingestion into Khoj AI assistant.

Usage:
    python -m utils.aim_ingestion                    # Export all experiments
    python -m utils.aim_ingestion --run HASH         # Export specific run
    python -m utils.aim_ingestion --since 7d         # Export last 7 days
    python -m utils.aim_ingestion --watch            # Watch for new experiments
"""

import os
import json
import time
import argparse
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, Any, List

try:
    from aim import Repo
    AIM_AVAILABLE = True
except ImportError:
    AIM_AVAILABLE = False

# Configuration
AIM_REPO_PATH = os.environ.get("AIM_REPO", "/workspace/outputs/aim")
KNOWLEDGE_DIR = Path(os.environ.get("KNOWLEDGE_DIR", "./knowledge"))
OBSIDIAN_VAULT = Path(os.environ.get("OBSIDIAN_VAULT", "./knowledge/experiments"))


def ensure_dirs():
    """Ensure output directories exist."""
    OBSIDIAN_VAULT.mkdir(parents=True, exist_ok=True)
    (KNOWLEDGE_DIR / "aim_exports").mkdir(parents=True, exist_ok=True)


def format_metrics_table(metrics: Dict[str, Any]) -> str:
    """Format metrics as markdown table."""
    if not metrics:
        return "*No metrics recorded*"

    lines = ["| Metric | Value |", "|--------|-------|"]
    for key, value in sorted(metrics.items()):
        if isinstance(value, float):
            value = f"{value:.4f}"
        lines.append(f"| {key} | {value} |")
    return "\n".join(lines)


def format_params_table(params: Dict[str, Any]) -> str:
    """Format parameters as markdown table."""
    if not params:
        return "*No parameters recorded*"

    lines = ["| Parameter | Value |", "|-----------|-------|"]
    for key, value in sorted(params.items()):
        if isinstance(value, dict):
            value = json.dumps(value, indent=2)
        lines.append(f"| {key} | `{value}` |")
    return "\n".join(lines)


def export_run_to_markdown(run) -> str:
    """Export a single AIM run to Obsidian-compatible markdown."""
    run_hash = run.hash
    experiment = run.experiment or "default"
    created_at = datetime.fromtimestamp(run.created_at).strftime("%Y-%m-%d %H:%M:%S")

    # Get final metrics (last value of each tracked metric)
    final_metrics = {}
    try:
        for metric_name in run.metrics():
            metric = run.get_metric(metric_name)
            if metric and hasattr(metric, 'values'):
                values = list(metric.values)
                if values:
                    final_metrics[metric_name] = values[-1]
    except Exception:
        pass

    # Get hyperparameters
    params = {}
    try:
        params = dict(run.get("hparams", {}))
    except Exception:
        pass

    # Get tags
    tags = []
    try:
        tags = list(run.tags) if hasattr(run, 'tags') else []
    except Exception:
        pass

    # Build markdown content
    tag_str = " ".join(f"#{t}" for t in tags) if tags else "#experiment"

    content = f"""---
type: experiment
experiment: {experiment}
run_hash: {run_hash}
created: {created_at}
tags: [{", ".join(tags)}]
---

# Experiment: {experiment}

**Run ID:** `{run_hash}`
**Created:** {created_at}
**Tags:** {tag_str}

## Final Metrics

{format_metrics_table(final_metrics)}

## Hyperparameters

{format_params_table(params)}

## Notes

*Add your observations and insights here.*

## Related

- [[experiments-index|All Experiments]]
- [[{experiment}-experiments|{experiment} Experiments]]

---
*Auto-generated from AIM on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}*
"""
    return content


def export_run(run, output_dir: Path = None) -> Path:
    """Export a single run to markdown file."""
    output_dir = output_dir or OBSIDIAN_VAULT

    run_hash = run.hash[:8]
    experiment = run.experiment or "default"
    created_date = datetime.fromtimestamp(run.created_at).strftime("%Y%m%d")

    filename = f"{created_date}-{experiment}-{run_hash}.md"
    filepath = output_dir / filename

    content = export_run_to_markdown(run)
    filepath.write_text(content)

    return filepath


def export_all_runs(repo_path: str = None, since: Optional[timedelta] = None) -> List[Path]:
    """Export all runs from AIM repo."""
    if not AIM_AVAILABLE:
        print("ERROR: aim package not installed. Run: pip install aim")
        return []

    repo_path = repo_path or AIM_REPO_PATH
    ensure_dirs()

    try:
        repo = Repo(repo_path)
    except Exception as e:
        print(f"ERROR: Could not open AIM repo at {repo_path}: {e}")
        return []

    exported = []
    cutoff_time = None
    if since:
        cutoff_time = (datetime.now() - since).timestamp()

    for run in repo.iter_runs():
        if cutoff_time and run.created_at < cutoff_time:
            continue

        try:
            filepath = export_run(run)
            exported.append(filepath)
            print(f"Exported: {filepath.name}")
        except Exception as e:
            print(f"ERROR exporting run {run.hash}: {e}")

    # Create index file
    create_experiments_index(exported)

    return exported


def create_experiments_index(exported_files: List[Path]):
    """Create an index file for all experiments."""
    index_path = OBSIDIAN_VAULT / "experiments-index.md"

    content = """---
type: index
tags: [experiments, aim, mlops]
---

# Experiments Index

This is an auto-generated index of all ML experiments tracked in AIM.

## Recent Experiments

"""

    # Sort by filename (which includes date)
    for filepath in sorted(exported_files, reverse=True)[:20]:
        name = filepath.stem
        content += f"- [[{name}]]\n"

    content += f"""
## Statistics

- **Total Experiments:** {len(exported_files)}
- **Last Updated:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Links

- [AIM Dashboard](http://localhost:43800)
- [[paper-library|Paper Library]]
"""

    index_path.write_text(content)
    print(f"Created index: {index_path}")


def export_single_run(run_hash: str, repo_path: str = None) -> Optional[Path]:
    """Export a specific run by hash."""
    if not AIM_AVAILABLE:
        print("ERROR: aim package not installed")
        return None

    repo_path = repo_path or AIM_REPO_PATH
    ensure_dirs()

    try:
        repo = Repo(repo_path)
        run = repo.get_run(run_hash)
        if run:
            return export_run(run)
        else:
            print(f"Run not found: {run_hash}")
            return None
    except Exception as e:
        print(f"ERROR: {e}")
        return None


def watch_for_new_runs(repo_path: str = None, interval: int = 60):
    """Watch for new runs and export them."""
    if not AIM_AVAILABLE:
        print("ERROR: aim package not installed")
        return

    repo_path = repo_path or AIM_REPO_PATH
    ensure_dirs()

    seen_hashes = set()

    # Initialize with existing runs
    try:
        repo = Repo(repo_path)
        for run in repo.iter_runs():
            seen_hashes.add(run.hash)
        print(f"Initialized with {len(seen_hashes)} existing runs")
    except Exception as e:
        print(f"ERROR initializing: {e}")
        return

    print(f"Watching for new experiments (checking every {interval}s)...")

    while True:
        try:
            time.sleep(interval)
            repo = Repo(repo_path)

            for run in repo.iter_runs():
                if run.hash not in seen_hashes:
                    filepath = export_run(run)
                    seen_hashes.add(run.hash)
                    print(f"New experiment: {filepath.name}")
        except KeyboardInterrupt:
            print("\nStopped watching.")
            break
        except Exception as e:
            print(f"ERROR: {e}")


def parse_duration(duration_str: str) -> timedelta:
    """Parse duration string like '7d', '24h', '30m'."""
    unit = duration_str[-1].lower()
    value = int(duration_str[:-1])

    if unit == 'd':
        return timedelta(days=value)
    elif unit == 'h':
        return timedelta(hours=value)
    elif unit == 'm':
        return timedelta(minutes=value)
    else:
        raise ValueError(f"Unknown duration unit: {unit}")


def main():
    parser = argparse.ArgumentParser(description="Export AIM experiments to knowledge base")
    parser.add_argument("--repo", default=AIM_REPO_PATH, help="AIM repo path")
    parser.add_argument("--output", default=str(OBSIDIAN_VAULT), help="Output directory")
    parser.add_argument("--run", help="Export specific run by hash")
    parser.add_argument("--since", help="Export runs since (e.g., 7d, 24h)")
    parser.add_argument("--watch", action="store_true", help="Watch for new runs")
    parser.add_argument("--interval", type=int, default=60, help="Watch interval in seconds")

    args = parser.parse_args()

    global OBSIDIAN_VAULT
    OBSIDIAN_VAULT = Path(args.output)

    if args.watch:
        watch_for_new_runs(args.repo, args.interval)
    elif args.run:
        filepath = export_single_run(args.run, args.repo)
        if filepath:
            print(f"Exported to: {filepath}")
    else:
        since = parse_duration(args.since) if args.since else None
        exported = export_all_runs(args.repo, since)
        print(f"\nExported {len(exported)} experiments")


if __name__ == "__main__":
    main()
