#!/usr/bin/env python3
"""Build and register model as Docker image."""

import argparse
import subprocess
import json
from datetime import datetime
from pathlib import Path

REGISTRY = "localhost:5000"

TEMPLATE = '''---
type: model-registry
model_id: "{model_id}"
version: "{version}"
image_digest: "{digest}"
created: "{date}"
stage: development
run_id: "{run_id}"
tags: []
---

# Model: {model_id}

## Registry Info
- **Image:** {registry}/{model_id}:{version}
- **Digest:** {digest}
- **Run:** [[experiments/runs/{run_id}]]

## Metrics
{metrics_table}

## Usage
```bash
docker pull {registry}/{model_id}:{version}
docker run --rm {registry}/{model_id}:{version} --input /data
```

## Changelog
- {date}: Initial version
'''

def run_cmd(cmd):
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return result.returncode == 0, result.stdout.strip(), result.stderr.strip()

def main():
    parser = argparse.ArgumentParser(description="Register model image")
    parser.add_argument("model_id", help="Model identifier")
    parser.add_argument("--version", default="latest", help="Version tag")
    parser.add_argument("--dockerfile", default="Dockerfile.model", help="Dockerfile path")
    parser.add_argument("--context", default=".", help="Build context")
    parser.add_argument("--run-id", default="", help="Associated run ID")
    parser.add_argument("--metrics", default="{}", help="JSON metrics")
    parser.add_argument("--output", default="./knowledge/models/registry", help="Output directory")
    parser.add_argument("--push", action="store_true", help="Push to registry")
    args = parser.parse_args()

    image = f"{REGISTRY}/{args.model_id}:{args.version}"
    date = datetime.now().strftime("%Y-%m-%d")

    # Build
    print(f"Building {image}...")
    ok, out, err = run_cmd(f"docker build -t {image} -f {args.dockerfile} {args.context}")
    if not ok:
        print(f"Build failed: {err}")
        return

    # Get digest
    ok, digest, _ = run_cmd(f"docker inspect --format='{{{{.Id}}}}' {image}")
    digest = digest.replace("sha256:", "")[:12] if ok else "unknown"

    # Push
    if args.push:
        print(f"Pushing {image}...")
        ok, out, err = run_cmd(f"docker push {image}")
        if not ok:
            print(f"Push failed: {err}")

    # Write registry note
    try:
        metrics = json.loads(args.metrics)
    except:
        metrics = {}

    metrics_table = "\n".join(f"| {k} | {v} |" for k, v in metrics.items()) or "| | |"

    content = TEMPLATE.format(
        model_id=args.model_id, version=args.version, digest=digest, date=date,
        run_id=args.run_id or "N/A", registry=REGISTRY, metrics_table=metrics_table
    )

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.model_id}-{args.version}.md"
    out_path.write_text(content)

    print(f"Registered: {out_path}")
    print(f"Image: {image}")
    print(f"Digest: {digest}")

if __name__ == "__main__":
    main()
