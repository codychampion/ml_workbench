#!/usr/bin/env python3
"""
Hugging Face Dataset Downloader
===============================
Download image datasets from the Hugging Face Hub to the local data/ directory.

Usage (CLI):
    python -m pipelines.collect.collect_hf --dataset laion/laion400m --split train --limit 50

Usage (Hydra):
    python -m pipelines.collect.collect_hf --hydra pipeline=collect_hf collect_hf.dataset=laion/laion400m
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Any, Dict

# No-op decorators (Prefect removed)
def flow(*args, **kwargs):
    def decorator(fn):
        return fn
    return decorator if not args or callable(args[0]) else decorator

def task(*args, **kwargs):
    def decorator(fn):
        return fn
    return decorator if not args or callable(args[0]) else decorator

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config import get_config
from utils.hydra_aim import init_aim_from_hydra
from utils.manifest import record_collection_manifest

# Optional Hydra
try:
    import hydra
    from omegaconf import DictConfig
except ImportError:
    hydra = None
    DictConfig = Any

# HF datasets
try:
    from datasets import load_dataset
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False


@task(name="download-hf-dataset", retries=2, retry_delay_seconds=30)
def download_dataset(
    dataset: str,
    split: str,
    limit: int,
    image_column: str,
    id_column: Optional[str],
    output_dir: Path,
) -> Dict[str, Any]:
    """
    Download images and metadata from a HuggingFace dataset split.

    Args:
        dataset: HuggingFace dataset name (e.g., "laion/laion400m")
        split: Dataset split to download (train, test, validation)
        limit: Maximum number of samples to download
        image_column: Name of the image column in the dataset
        id_column: Optional ID column for naming files
        output_dir: Base output directory for downloaded data

    Returns:
        Dictionary with download statistics and paths
    """
    if not HF_AVAILABLE:
        raise RuntimeError(
            "datasets package not installed. Install with: pip install datasets"
        )

    print(f"\n{'='*60}")
    print(f"Downloading HuggingFace Dataset: {dataset}")
    print(f"{'='*60}")
    print(f"Split: {split} | Limit: {limit} | Image Column: {image_column}")

    # Load dataset with error handling
    try:
        ds = load_dataset(dataset, split=split, streaming=False)
    except Exception as e:
        raise ValueError(
            f"Failed to load dataset '{dataset}' split '{split}': {e}\n"
            f"Check dataset name and split are valid on HuggingFace Hub"
        ) from e

    target_dir = output_dir / dataset.replace("/", "_") / split
    target_dir.mkdir(parents=True, exist_ok=True)

    meta_path = target_dir / "metadata.jsonl"
    saved = 0
    skipped = 0

    try:
        with meta_path.open("w", encoding="utf-8") as meta_f:
            for ex in ds:
                if saved >= limit:
                    break
                if image_column not in ex or ex[image_column] is None:
                    skipped += 1
                    continue
                img_obj = ex[image_column]
                try:
                    pil_img = img_obj.to_pil()
                except Exception:
                    try:
                        pil_img = img_obj["bytes"].to_pil()
                    except Exception as e:
                        print(f"  Warning: Failed to process image at index {saved + skipped}: {e}")
                        skipped += 1
                        continue

                sample_id = ex.get(id_column) if id_column else ex.get("id", saved)
                fname = f"{sample_id if sample_id is not None else saved:08d}.png"
                out_path = target_dir / fname

                try:
                    pil_img.save(out_path)
                except Exception as e:
                    print(f"  Warning: Failed to save image {fname}: {e}")
                    skipped += 1
                    continue

                # Strip image payload from metadata
                ex_meta = {k: v for k, v in ex.items() if k != image_column}
                meta_f.write(json.dumps({"id": sample_id, "file": fname, **ex_meta}) + "\n")

                saved += 1
                if saved % 25 == 0:
                    print(f"[HF] Saved {saved} samples...")

        if saved == 0:
            raise ValueError(
                f"No samples saved. Check that '{image_column}' column exists and contains valid images"
            )

    except IOError as e:
        raise IOError(
            f"Failed to write files to {target_dir}: {e}\n"
            f"Check disk space and write permissions"
        ) from e

    print(f"\n{'='*60}")
    print(f"Download Complete: {saved} samples ({skipped} skipped)")
    print(f"Output: {target_dir}")
    print(f"{'='*60}")

    return {
        "samples": saved,
        "output_dir": str(target_dir),
        "dataset": dataset,
        "split": split,
    }


def main():
    """CLI interface for HuggingFace dataset collection."""
    parser = argparse.ArgumentParser(description="Download a Hugging Face image dataset")
    parser.add_argument("--dataset", required=True, help="Dataset name (e.g., laion/laion400m)")
    parser.add_argument("--split", default="train", help="Dataset split (default: train)")
    parser.add_argument("--limit", type=int, default=100, help="Max samples to download")
    parser.add_argument("--image-column", default="image", help="Image column name (default: image)")
    parser.add_argument("--id-column", default=None, help="Optional ID column to use for filenames")
    parser.add_argument("--output-dir", type=Path, default=Path("./data/collected"), help="Output base dir")
    parser.add_argument(
        "--hydra",
        action="store_true",
        help="Use Hydra config (conf/pipeline/collect_hf.yaml) instead of CLI flags"
    )
    args = parser.parse_args()

    result = download_dataset(
        dataset=args.dataset,
        split=args.split,
        limit=args.limit,
        image_column=args.image_column,
        id_column=args.id_column,
        output_dir=args.output_dir,
    )

    # Write collection manifest
    record_collection_manifest(
        name=f"hf-{args.dataset.replace('/', '-')}",
        output_dir=Path(result["output_dir"]),
        source={
            "type": "huggingface",
            "dataset": args.dataset,
            "split": args.split,
            "image_column": args.image_column,
            "id_column": args.id_column,
        },
        counts={"samples": result["samples"]},
        cfg=None,
    )

    print("\n" + "=" * 60)
    print("Collection Complete!")
    print("=" * 60)
    print(f"Output directory: {result['output_dir']}")
    print("\nNext steps:")
    print("  1. Caption images: python -m pipelines.annotate.caption")
    print("  2. View in FiftyOne: python -m pipelines.annotate.create_dataset")


if __name__ == "__main__":
    # Support Hydra-driven runs with --hydra flag
    if hydra is not None and "--hydra" in sys.argv:
        # Remove the flag so Hydra doesn't treat it as an unknown argument
        sys.argv = [arg for arg in sys.argv if arg != "--hydra"]

        @hydra.main(version_base="1.2", config_path="../../conf", config_name="config")
        def hydra_main(cfg: "DictConfig"):  # type: ignore[misc]
            hf_cfg = cfg.get("pipeline", {}).get("collect_hf", {})
            source = hf_cfg.get("source", {})
            output_cfg = hf_cfg.get("output", {})

            dataset = source.get("dataset")
            split = source.get("split", "train")
            limit = source.get("limit", 100)
            image_column = source.get("image_column", "image")
            id_column = source.get("id_column")
            output_dir = Path(output_cfg.get("dir", "./data/collected"))

            result = download_dataset(
                dataset=dataset,
                split=split,
                limit=limit,
                image_column=image_column,
                id_column=id_column,
                output_dir=output_dir,
            )

            # Log config + git state into AIM
            init_aim_from_hydra(cfg, run_name=f"collect-hf-{dataset.replace('/', '-')}", experiment="collect")

            # Write collection manifest
            record_collection_manifest(
                name=f"hf-{dataset.replace('/', '-')}",
                output_dir=Path(result["output_dir"]),
                source={
                    "type": "huggingface",
                    "dataset": dataset,
                    "split": split,
                    "image_column": image_column,
                    "id_column": id_column,
                },
                counts={"samples": result["samples"]},
                cfg=cfg,
            )

        hydra_main()
    else:
        main()
