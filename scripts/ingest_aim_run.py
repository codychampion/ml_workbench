#!/usr/bin/env python3
"""Ingest AIM run to knowledge base as markdown note."""

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

try:
    from aim import Repo
    AIM_AVAILABLE = True
except ImportError:
    AIM_AVAILABLE = False

TEMPLATE = '''---
type: run-summary
run_id: "{run_hash}"
exp_id: "{experiment}"
created: "{date}"
status: completed
tags: [{tags}]
---

# Run: {run_hash}

**Experiment:** {experiment}
**Created:** {created_at}

## Metrics
{metrics_table}

## Parameters
{params_table}

## Observations

## Next Steps
- [ ]
'''

def format_table(data, headers=("Key", "Value")):
    if not data:
        return "*None*"
    lines = [f"| {headers[0]} | {headers[1]} |", "|-----|-------|"]
    for k, v in sorted(data.items()):
        v = f"{v:.4f}" if isinstance(v, float) else str(v)[:50]
        lines.append(f"| {k} | {v} |")
    return "\n".join(lines)

def export_run(run, output_dir):
    run_hash = run.hash[:8]
    experiment = run.experiment or "default"
    created_at = datetime.fromtimestamp(run.created_at).strftime("%Y-%m-%d %H:%M")
    date = datetime.now().strftime("%Y-%m-%d")

    # Get metrics
    metrics = {}
    try:
        for m in run.metrics():
            vals = list(run.get_metric(m).values)
            if vals:
                metrics[m] = vals[-1]
    except Exception:
        pass

    # Get params
    params = {}
    try:
        params = dict(run.get("hparams", {}))
    except Exception:
        pass

    tags = ", ".join(f'"{t}"' for t in list(run.tags)[:5]) if hasattr(run, 'tags') else ""

    content = TEMPLATE.format(
        run_hash=run_hash, experiment=experiment, date=date, created_at=created_at,
        tags=tags, metrics_table=format_table(metrics), params_table=format_table(params)
    )

    out_path = output_dir / f"{date}-{experiment}-{run_hash}.md"
    out_path.write_text(content)
    return out_path

def main():
    parser = argparse.ArgumentParser(description="Ingest AIM run to knowledge base")
    parser.add_argument("--run", help="Run hash (default: latest)")
    parser.add_argument("--repo", default=os.environ.get("AIM_REPO", "./outputs/aim"), help="AIM repo path")
    parser.add_argument("--output", default="./knowledge/experiments/runs", help="Output directory")
    args = parser.parse_args()

    if not AIM_AVAILABLE:
        print("ERROR: aim not installed")
        return

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        repo = Repo(args.repo)
        if args.run:
            run = repo.get_run(args.run)
            if run:
                path = export_run(run, output_dir)
                print(f"Exported: {path}")
        else:
            runs = list(repo.iter_runs())
            if runs:
                path = export_run(runs[-1], output_dir)
                print(f"Exported latest: {path}")
            else:
                print("No runs found")
    except Exception as e:
        print(f"ERROR: {e}")

if __name__ == "__main__":
    main()
