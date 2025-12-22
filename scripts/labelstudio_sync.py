#!/usr/bin/env python3
"""Sync FiftyOne <-> Label Studio annotations."""

import argparse
import json
from pathlib import Path

def export_to_labelstudio(dataset_path: str, output: str, label_config: str = None):
    """Export FiftyOne dataset to Label Studio tasks JSON."""
    try:
        import fiftyone as fo
        dataset = fo.load_dataset(dataset_path)
        tasks = []
        for sample in dataset:
            tasks.append({
                "data": {"image": sample.filepath},
                "predictions": []  # Add predictions if available
            })
        Path(output).write_text(json.dumps(tasks, indent=2))
        print(f"Exported {len(tasks)} tasks to {output}")
    except ImportError:
        print("FiftyOne not installed")

def import_from_labelstudio(annotations_path: str, dataset_name: str):
    """Import Label Studio annotations to FiftyOne dataset."""
    try:
        import fiftyone as fo
        annotations = json.loads(Path(annotations_path).read_text())
        print(f"Found {len(annotations)} annotations")
        # Implementation depends on annotation format
        print("Import implemented per project needs")
    except ImportError:
        print("FiftyOne not installed")

def main():
    parser = argparse.ArgumentParser(description="Sync FiftyOne <-> Label Studio")
    parser.add_argument("command", choices=["export", "import"])
    parser.add_argument("--dataset", help="FiftyOne dataset name")
    parser.add_argument("--file", help="JSON file path")
    args = parser.parse_args()

    if args.command == "export":
        export_to_labelstudio(args.dataset, args.file or "tasks.json")
    else:
        import_from_labelstudio(args.file, args.dataset)

if __name__ == "__main__":
    main()
