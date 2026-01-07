#!/usr/bin/env python3
"""
CUAD Dataset Collector
======================
Download the Contract Understanding Atticus Dataset (CUAD) from Hugging Face Hub
for legal contract analysis and MAKER-based experiments.

Usage:
    python -m pipelines.collect.collect_cuad --split train --limit 1000
    python -m pipelines.collect.collect_cuad --hydra  # uses conf/pipeline/collect_cuad.yaml
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Any, Dict, List

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from utils.decorators import flow, task
from utils.hydra_aim import init_aim_from_hydra
from utils.manifest import record_collection_manifest

# Optional Hydra integration
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


@task(name="download-cuad", retries=2, retry_delay_seconds=30)
def download_cuad_task(
    dataset: str,
    split: str,
    limit: int,
    output_dir: Path,
    text_column: str = "text",
) -> Dict[str, Any]:
    """
    Download and process CUAD dataset from HuggingFace.

    Args:
        dataset: HuggingFace dataset identifier
        split: Dataset split to download
        limit: Maximum samples (-1 for all)
        output_dir: Base output directory
        text_column: Name of text column in dataset

    Returns:
        Dictionary with download statistics

    Raises:
        RuntimeError: If datasets package not installed
        ValueError: If dataset/split not found
    """
    if not HF_AVAILABLE:
        raise RuntimeError(
            "datasets package not installed. Install with: pip install datasets"
        )

    print(f"\n{'='*60}")
    print(f"Downloading CUAD Dataset")
    print(f"{'='*60}")
    print(f"Dataset: {dataset} | Split: {split} | Limit: {limit}")

    # Load dataset with error handling
    try:
        ds = load_dataset(dataset, split=split, streaming=False)
    except Exception as e:
        raise ValueError(
            f"Failed to load dataset '{dataset}' split '{split}': {e}\n"
            f"Check dataset name and split are valid on HuggingFace Hub"
        ) from e

    # Create output directory
    target_dir = output_dir / split
    target_dir.mkdir(parents=True, exist_ok=True)

    # Process samples
    output_file = target_dir / "cuad_contracts.jsonl"
    stats = {
        "total_samples": 0,
        "clause_categories": set(),
        "total_text_length": 0,
    }

    print(f"\nProcessing samples...")
    try:
        with output_file.open("w", encoding="utf-8") as f:
            for idx, sample in enumerate(ds):
                if limit > 0 and idx >= limit:
                    break

                # Validate text column exists
                if text_column not in sample:
                    print(f"  Warning: Sample {idx} missing '{text_column}' column, skipping")
                    continue

                # Extract sample data
                sample_data = {
                    "id": idx,
                    "text": sample.get(text_column, ""),
                    "text_length": len(sample.get(text_column, "")),
                    "timestamp": datetime.now().isoformat(),
                }

                # Add clause annotations
                for key, value in sample.items():
                    if key != text_column:
                        sample_data[key] = value
                        if value:  # Track present categories
                            stats["clause_categories"].add(key)

                f.write(json.dumps(sample_data, ensure_ascii=False) + "\n")
                stats["total_samples"] += 1
                stats["total_text_length"] += sample_data["text_length"]

                if (idx + 1) % 100 == 0:
                    print(f"  Processed {idx + 1} samples...")

        if stats["total_samples"] == 0:
            raise ValueError(
                f"No samples processed. Check that '{text_column}' column exists in dataset"
            )

    except IOError as e:
        raise IOError(
            f"Failed to write to output file {output_file}: {e}\n"
            f"Check disk space and write permissions"
        ) from e

    # Save metadata
    metadata = {
        "dataset": dataset,
        "split": split,
        "download_date": datetime.now().isoformat(),
        "total_samples": stats["total_samples"],
        "clause_categories": sorted(list(stats["clause_categories"])),
        "num_categories": len(stats["clause_categories"]),
        "avg_text_length": stats["total_text_length"] / max(stats["total_samples"], 1),
        "output_file": str(output_file),
    }

    metadata_file = target_dir / "metadata.json"
    with metadata_file.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"Download Complete")
    print(f"{'='*60}")
    print(f"Samples: {stats['total_samples']}")
    print(f"Categories: {len(stats['clause_categories'])}")
    print(f"Output: {target_dir}")
    print(f"Metadata: {metadata_file}")

    return {
        "samples": stats["total_samples"],
        "categories": len(stats["clause_categories"]),
        "output_dir": str(target_dir),
    }


@task(name="prepare-maker-experiment")
def prepare_maker_experiment_task(
    cuad_dir: Path,
    output_dir: Path,
    voting_threshold: int = 3,
    granularity: str = "clause_level",
) -> int:
    """Prepare CUAD data for MAKER-style decomposition experiments."""
    print(f"\n{'='*60}")
    print(f"Preparing MAKER Experiment")
    print(f"{'='*60}")
    print(f"Granularity: {granularity} | Voting threshold (k): {voting_threshold}")

    input_file = cuad_dir / "cuad_contracts.jsonl"
    if not input_file.exists():
        raise FileNotFoundError(f"CUAD data not found at {input_file}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Decompose into individual tasks
    tasks = []
    with input_file.open("r", encoding="utf-8") as f:
        for line in f:
            sample = json.loads(line)

            # Create subtasks for each clause category
            for key in sample.keys():
                if key not in ["id", "text", "text_length", "timestamp"]:
                    task = {
                        "contract_id": sample["id"],
                        "text": sample["text"],
                        "clause_category": key,
                        "label": sample[key],
                        "task_type": "clause_identification",
                        "voting_threshold": voting_threshold,
                    }
                    tasks.append(task)

    # Save decomposed tasks
    output_file = output_dir / "maker_tasks.jsonl"
    with output_file.open("w", encoding="utf-8") as f:
        for task in tasks:
            f.write(json.dumps(task, ensure_ascii=False) + "\n")

    # Save experiment config
    config_file = output_dir / "maker_config.json"
    with config_file.open("w", encoding="utf-8") as f:
        json.dump({
            "voting_threshold": voting_threshold,
            "granularity": granularity,
            "total_tasks": len(tasks),
            "created_at": datetime.now().isoformat(),
        }, f, indent=2)

    print(f"\nCreated {len(tasks)} subtasks")
    print(f"Tasks: {output_file}")
    print(f"Config: {config_file}")

    return len(tasks)


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Download CUAD dataset for contract review experiments"
    )
    parser.add_argument(
        "--dataset",
        default="theatticusproject/cuad",
        help="Dataset name (default: theatticusproject/cuad)"
    )
    parser.add_argument(
        "--split",
        default="train",
        help="Dataset split (default: train)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=1000,
        help="Max samples to download (-1 for all, default: 1000)"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./data/collected/cuad"),
        help="Output directory (default: ./data/collected/cuad)"
    )
    parser.add_argument(
        "--prepare-maker",
        action="store_true",
        help="Prepare data for MAKER experiments"
    )
    parser.add_argument(
        "--voting-threshold",
        type=int,
        default=3,
        help="Voting threshold k for MAKER (default: 3)"
    )
    parser.add_argument(
        "--hydra",
        action="store_true",
        help="Use Hydra config (conf/pipeline/collect_cuad.yaml)"
    )

    args = parser.parse_args()

    if not args.hydra:
        # CLI mode
        result = download_cuad_task(
            dataset=args.dataset,
            split=args.split,
            limit=args.limit,
            output_dir=args.output_dir,
        )

        # Record manifest
        record_collection_manifest(
            name="cuad-collection",
            output_dir=args.output_dir / args.split,
            source={
                "type": "huggingface",
                "dataset": args.dataset,
                "split": args.split,
                "limit": args.limit,
            },
            counts={
                "samples": result["samples"],
                "categories": result["categories"],
            },
            cfg=None,
        )

        if args.prepare_maker:
            maker_dir = args.output_dir.parent / "cuad_maker_experiment"
            num_tasks = prepare_maker_experiment_task(
                cuad_dir=args.output_dir / args.split,
                output_dir=maker_dir,
                voting_threshold=args.voting_threshold,
            )

        print("\n" + "=" * 60)
        print("CUAD Collection Complete!")
        print("=" * 60)
        print(f"Output: {args.output_dir}")
        if args.prepare_maker:
            print(f"MAKER tasks: {maker_dir}")
        print("\nNext steps:")
        print("  1. Review experiment plan: cat knowledge/experiments/plans/cuad_maker_plan.md")
        print("  2. Run MAKER experiment: python -m pipelines.evaluate.cuad_maker")


@flow(name="cuad-collection-pipeline", log_prints=True)
def run_collection_flow(
    dataset: str,
    split: str,
    limit: int,
    output_dir: Path,
    prepare_maker: bool = False,
    voting_threshold: int = 3,
) -> Dict[str, Any]:
    """
    Flow for CUAD dataset collection.

    (Prefect removed; run directly with python -m pipelines.collect.collect_cuad)
    """
    result = download_cuad_task(
        dataset=dataset,
        split=split,
        limit=limit,
        output_dir=output_dir,
    )

    if prepare_maker:
        maker_dir = output_dir.parent / "cuad_maker_experiment"
        num_tasks = prepare_maker_experiment_task(
            cuad_dir=output_dir / split,
            output_dir=maker_dir,
            voting_threshold=voting_threshold,
        )
        result["maker_tasks"] = num_tasks

    return result


if __name__ == "__main__":
    # Support Hydra-driven runs with --hydra flag
    if hydra is not None and "--hydra" in sys.argv:
        # Remove flag for Hydra
        sys.argv = [arg for arg in sys.argv if arg != "--hydra"]

        @hydra.main(version_base="1.2", config_path="../../conf", config_name="config")
        def hydra_main(cfg: "DictConfig"):  # type: ignore[misc]
            cuad_cfg = cfg.get("pipeline", {}).get("collect_cuad", {})
            source = cuad_cfg.get("source", {})
            filter_cfg = cuad_cfg.get("filter", {})
            output_cfg = cuad_cfg.get("output", {})
            maker_cfg = cuad_cfg.get("maker", {})

            dataset = source.get("dataset", "theatticusproject/cuad")
            split = source.get("split", "train")
            limit = source.get("limit", 1000)
            text_column = filter_cfg.get("text_column", "text")
            output_dir = Path(output_cfg.get("dir", "./data/collected/cuad"))
            prepare_maker = maker_cfg.get("enabled", False)
            voting_threshold = maker_cfg.get("voting_threshold", 3)

            # Download dataset
            result = download_cuad_task(
                dataset=dataset,
                split=split,
                limit=limit,
                output_dir=output_dir,
                text_column=text_column,
            )

            # Initialize AIM tracking
            init_aim_from_hydra(cfg, run_name=f"collect-cuad-{split}", experiment="collect")

            # Record manifest
            record_collection_manifest(
                name="cuad-collection",
                output_dir=output_dir / split,
                source={
                    "type": "huggingface",
                    "dataset": dataset,
                    "split": split,
                    "limit": limit,
                },
                counts={
                    "samples": result["samples"],
                    "categories": result["categories"],
                },
                cfg=cfg,
            )

            # Prepare MAKER experiment if enabled
            if prepare_maker:
                maker_dir = output_dir.parent / "cuad_maker_experiment"
                num_tasks = prepare_maker_experiment_task(
                    cuad_dir=output_dir / split,
                    output_dir=maker_dir,
                    voting_threshold=voting_threshold,
                    granularity=maker_cfg.get("granularity", "clause_level"),
                )
                print(f"\nMAKER experiment prepared with {num_tasks} tasks")

        hydra_main()
    else:
        main()
