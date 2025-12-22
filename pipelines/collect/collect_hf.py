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
import os
from pathlib import Path
from typing import Optional

# Optional Hydra
try:
    import hydra
    from omegaconf import DictConfig
except ImportError:
    hydra = None
    DictConfig = None

# HF datasets
try:
    from datasets import load_dataset
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False


def download_dataset(
    dataset: str,
    split: str,
    limit: int,
    image_column: str,
    id_column: Optional[str],
    output_dir: Path,
) -> int:
    """Download images and metadata from a HF dataset split."""
    if not HF_AVAILABLE:
        raise RuntimeError("datasets package not installed. pip install datasets")

    ds = load_dataset(dataset, split=split, streaming=False)
    target_dir = output_dir / dataset.replace("/", "_") / split
    target_dir.mkdir(parents=True, exist_ok=True)

    meta_path = target_dir / "metadata.jsonl"
    saved = 0

    with meta_path.open("w", encoding="utf-8") as meta_f:
        for ex in ds:
            if saved >= limit:
                break
            if image_column not in ex or ex[image_column] is None:
                continue
            img_obj = ex[image_column]
            try:
                pil_img = img_obj.to_pil()
            except Exception:
                try:
                    pil_img = img_obj["bytes"].to_pil()
                except Exception:
                    continue

            sample_id = ex.get(id_column) if id_column else ex.get("id", saved)
            fname = f"{sample_id if sample_id is not None else saved:08d}.png"
            out_path = target_dir / fname
            pil_img.save(out_path)

            # Strip image payload from metadata
            ex_meta = {k: v for k, v in ex.items() if k != image_column}
            meta_f.write(json.dumps({"id": sample_id, "file": fname, **ex_meta}) + "\n")

            saved += 1
            if saved % 25 == 0:
                print(f"[HF] Saved {saved} samples...")

    print(f"[HF] Collected {saved} samples to {target_dir}")
    return saved


def main_cli():
    parser = argparse.ArgumentParser(description="Download a Hugging Face image dataset")
    parser.add_argument("--dataset", required=True, help="Dataset name (e.g., laion/laion400m)")
    parser.add_argument("--split", default="train", help="Dataset split (default: train)")
    parser.add_argument("--limit", type=int, default=100, help="Max samples to download")
    parser.add_argument("--image-column", default="image", help="Image column name (default: image)")
    parser.add_argument("--id-column", default=None, help="Optional ID column to use for filenames")
    parser.add_argument("--output-dir", type=Path, default=Path("./data/collected"), help="Output base dir")
    args = parser.parse_args()

    download_dataset(
        dataset=args.dataset,
        split=args.split,
        limit=args.limit,
        image_column=args.image_column,
        id_column=args.id_column,
        output_dir=args.output_dir,
    )


def main_hydra(cfg: DictConfig):
    hf_cfg = cfg.get("collect_hf", {})
    output_root = Path(hf_cfg.get("output_dir", "./data/collected"))
    download_dataset(
        dataset=hf_cfg.get("dataset"),
        split=hf_cfg.get("split", "train"),
        limit=hf_cfg.get("limit", 100),
        image_column=hf_cfg.get("image_column", "image"),
        id_column=hf_cfg.get("id_column"),
        output_dir=output_root,
    )


if __name__ == "__main__":
    if hydra is not None and "--hydra" in os.sys.argv:
        @hydra.main(config_path="../../conf", config_name="config")
        def hydra_entry(cfg: DictConfig):  # type: ignore[misc]
            main_hydra(cfg)
        hydra_entry()
    else:
        main_cli()
