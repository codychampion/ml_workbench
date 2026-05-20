#!/usr/bin/env python3
"""Build and register a model Docker image."""

import argparse
import json
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Sequence

REGISTRY = "localhost:5000"
SAFE_IMAGE_PART = re.compile(r"^[a-z0-9][a-z0-9._/-]{0,127}$")
SAFE_TAG = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_.-]{0,127}$")

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
| Metric | Value |
|---|---|
{metrics_table}

## Usage
```bash
docker pull {registry}/{model_id}:{version}
docker run --rm {registry}/{model_id}:{version} --input /data
```

## Changelog
- {date}: Initial version
'''


def validate_image_part(value: str, label: str) -> str:
    """Validate a Docker repository/name component used in generated commands."""
    if not SAFE_IMAGE_PART.fullmatch(value):
        raise ValueError(
            f"Invalid {label}: {value!r}. Use lowercase letters, numbers, '.', '_', '-', and '/'."
        )
    return value


def validate_tag(value: str) -> str:
    """Validate a Docker tag."""
    if not SAFE_TAG.fullmatch(value):
        raise ValueError(
            f"Invalid version tag: {value!r}. Use letters, numbers, '.', '_', or '-'."
        )
    return value


def run_cmd(cmd: Sequence[str]) -> tuple[bool, str, str]:
    """Run a command without shell interpolation and capture output."""
    result = subprocess.run(cmd, shell=False, capture_output=True, text=True, check=False)
    return result.returncode == 0, result.stdout.strip(), result.stderr.strip()


def parse_metrics(raw: str) -> dict:
    """Parse metric JSON and require an object at the top level."""
    try:
        metrics = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid metrics JSON: {exc}") from exc

    if not isinstance(metrics, dict):
        raise ValueError("--metrics must be a JSON object, for example: '{\"accuracy\": 0.91}'")
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(description="Register model image")
    parser.add_argument("model_id", help="Model identifier")
    parser.add_argument("--version", default="latest", help="Version tag")
    parser.add_argument("--dockerfile", default="Dockerfile.model", help="Dockerfile path")
    parser.add_argument("--context", default=".", help="Build context")
    parser.add_argument("--run-id", default="", help="Associated run ID")
    parser.add_argument("--metrics", default="{}", help="JSON metrics object")
    parser.add_argument("--output", default="./knowledge/models/registry", help="Output directory")
    parser.add_argument("--push", action="store_true", help="Push to registry")
    args = parser.parse_args()

    try:
        model_id = validate_image_part(args.model_id, "model_id")
        version = validate_tag(args.version)
        metrics = parse_metrics(args.metrics)
    except ValueError as exc:
        parser.error(str(exc))

    dockerfile = Path(args.dockerfile)
    context = Path(args.context)
    if not dockerfile.exists():
        parser.error(f"Dockerfile does not exist: {dockerfile}")
    if not context.exists():
        parser.error(f"Build context does not exist: {context}")

    image = f"{REGISTRY}/{model_id}:{version}"
    date = datetime.now().strftime("%Y-%m-%d")

    print(f"Building {image}...")
    ok, _, err = run_cmd(["docker", "build", "-t", image, "-f", str(dockerfile), str(context)])
    if not ok:
        print(f"Build failed: {err}", file=sys.stderr)
        return 1

    ok, digest, err = run_cmd(["docker", "inspect", "--format={{.Id}}", image])
    if not ok:
        print(f"Warning: could not inspect image digest: {err}", file=sys.stderr)
        digest = "unknown"
    else:
        digest = digest.replace("sha256:", "")[:12]

    if args.push:
        print(f"Pushing {image}...")
        ok, _, err = run_cmd(["docker", "push", image])
        if not ok:
            print(f"Push failed: {err}", file=sys.stderr)
            return 1

    metrics_table = "\n".join(f"| {key} | {value} |" for key, value in metrics.items()) or "| _none_ | _none_ |"

    content = TEMPLATE.format(
        model_id=model_id,
        version=version,
        digest=digest,
        date=date,
        run_id=args.run_id or "N/A",
        registry=REGISTRY,
        metrics_table=metrics_table,
    )

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{model_id.replace('/', '_')}-{version}.md"
    out_path.write_text(content)

    print(f"Registered: {out_path}")
    print(f"Image: {image}")
    print(f"Digest: {digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
