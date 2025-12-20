#!/usr/bin/env python3
"""
Create FiftyOne Dataset from Collected Data
============================================
Creates a FiftyOne dataset from collected and captioned images
for visualization and quality review.

Usage:
    python create_dataset.py --input-dir ./data/collected --name my-dataset
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    import fiftyone as fo
except ImportError:
    print("FiftyOne not installed. Run: pip install fiftyone")
    sys.exit(1)


def create_dataset(
    input_dir: Path,
    dataset_name: str,
    include_captions: bool = True,
    launch_app: bool = False
) -> fo.Dataset:
    """
    Create a FiftyOne dataset from collected images.

    Args:
        input_dir: Directory containing images (and optional .txt captions)
        dataset_name: Name for the dataset
        include_captions: Whether to include captions as labels
        launch_app: Whether to launch FiftyOne app after creation
    """
    input_dir = Path(input_dir)

    # Find all images
    image_extensions = {".jpg", ".jpeg", ".png", ".webp", ".gif"}
    images = [
        f for f in input_dir.rglob("*")
        if f.suffix.lower() in image_extensions
        and not f.name.startswith(".")
    ]

    if not images:
        print(f"No images found in {input_dir}")
        return None

    print(f"\n{'='*60}")
    print(f"Creating FiftyOne Dataset: {dataset_name}")
    print(f"{'='*60}")
    print(f"Images found: {len(images)}")

    # Delete existing dataset if exists
    if dataset_name in fo.list_datasets():
        print(f"Deleting existing dataset: {dataset_name}")
        fo.delete_dataset(dataset_name)

    # Create dataset
    dataset = fo.Dataset(dataset_name)
    dataset.persistent = True

    samples = []
    captions_found = 0

    for image_path in images:
        sample = fo.Sample(filepath=str(image_path))

        # Add source metadata
        sample["source_dir"] = str(image_path.parent.name)
        sample["filename"] = image_path.name

        # Look for caption file
        if include_captions:
            caption_file = image_path.with_suffix(".txt")
            if caption_file.exists():
                caption = caption_file.read_text().strip()
                sample["caption"] = caption
                captions_found += 1

        # Look for gallery-dl metadata
        json_file = image_path.with_suffix(".json")
        if json_file.exists():
            try:
                with open(json_file) as f:
                    metadata = json.load(f)
                sample["title"] = metadata.get("title", "")
                sample["author"] = metadata.get("author", "")
                sample["subreddit"] = metadata.get("subreddit", "")
                sample["score"] = metadata.get("score", 0)
                sample["url"] = metadata.get("url", "")
            except:
                pass

        samples.append(sample)

    dataset.add_samples(samples)

    print(f"Dataset created: {len(dataset)} samples")
    print(f"Captions found: {captions_found}")

    # Print summary by source
    if dataset.count("source_dir") > 0:
        print("\nSamples by source:")
        for source in dataset.distinct("source_dir"):
            count = len(dataset.match(fo.ViewField("source_dir") == source))
            print(f"  {source}: {count}")

    # Launch app if requested
    if launch_app:
        print(f"\nLaunching FiftyOne app...")
        print("Open http://localhost:5151 in your browser")
        session = fo.launch_app(dataset, port=5151, address="0.0.0.0")
        session.wait()

    return dataset


def main():
    parser = argparse.ArgumentParser(
        description="Create FiftyOne dataset from collected images"
    )

    parser.add_argument(
        "--input-dir", "-i",
        type=Path,
        default=Path("./data/collected"),
        help="Directory containing images (default: ./data/collected)"
    )
    parser.add_argument(
        "--name", "-n",
        type=str,
        default=f"collected_{datetime.now().strftime('%Y%m%d')}",
        help="Dataset name"
    )
    parser.add_argument(
        "--no-captions",
        action="store_true",
        help="Don't include captions in dataset"
    )
    parser.add_argument(
        "--launch",
        action="store_true",
        help="Launch FiftyOne app after creation"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List existing datasets and exit"
    )

    args = parser.parse_args()

    if args.list:
        datasets = fo.list_datasets()
        print("Existing FiftyOne datasets:")
        for name in datasets:
            ds = fo.load_dataset(name)
            print(f"  {name}: {len(ds)} samples")
        return

    dataset = create_dataset(
        input_dir=args.input_dir,
        dataset_name=args.name,
        include_captions=not args.no_captions,
        launch_app=args.launch
    )

    if dataset:
        print(f"\nDataset '{args.name}' created successfully!")
        print("\nTo view later:")
        print(f"  python -c \"import fiftyone as fo; fo.load_dataset('{args.name}').launch()\"")


if __name__ == "__main__":
    main()
