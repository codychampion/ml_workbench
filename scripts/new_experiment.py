#!/usr/bin/env python3
"""Create a new experiment plan note."""

import argparse
from datetime import datetime
from pathlib import Path
import uuid

TEMPLATE = '''---
type: experiment-plan
exp_id: "{exp_id}"
created: "{date}"
status: draft
tags: [{tags}]
---

# Experiment: {title}

## Hypothesis

## Dataset
- **ID:**
- **Path:** s3://mlops-data/datasets/

## Method

## Success Criteria
-

## Runs
| Run ID | Status | Key Result | Notes |
|--------|--------|------------|-------|

## Conclusions

## Next Steps
- [ ]
'''

def main():
    parser = argparse.ArgumentParser(description="Create experiment plan note")
    parser.add_argument("title", help="Experiment title")
    parser.add_argument("--tags", default="", help="Comma-separated tags")
    parser.add_argument("--output", default="./knowledge/experiments/plans", help="Output directory")
    args = parser.parse_args()

    exp_id = f"exp-{datetime.now().strftime('%Y%m%d')}-{uuid.uuid4().hex[:6]}"
    date = datetime.now().strftime("%Y-%m-%d")
    tags = ", ".join(f'"{t.strip()}"' for t in args.tags.split(",") if t.strip())

    content = TEMPLATE.format(exp_id=exp_id, date=date, title=args.title, tags=tags)
    out_path = Path(args.output) / f"{exp_id}.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(content)

    print(f"Created: {out_path}")
    print(f"Experiment ID: {exp_id}")

if __name__ == "__main__":
    main()
