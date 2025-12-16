#!/usr/bin/env python3
"""
Spotlight Server Startup Script
================================
Starts the Renumics Spotlight server with configurable options.

This server provides:
- Interactive data exploration UI
- Embedding visualization
- Automatic data quality analysis
- Integration with local datasets
"""

import os
import sys
from pathlib import Path

# Configuration from environment
HOST = os.environ.get("SPOTLIGHT_HOST", "0.0.0.0")
PORT = int(os.environ.get("SPOTLIGHT_PORT", "8000"))
DATA_DIR = Path(os.environ.get("SPOTLIGHT_DATA_DIR", "/workspace/data"))
OUTPUT_DIR = Path(os.environ.get("SPOTLIGHT_OUTPUT_DIR", "/workspace/outputs"))


def find_datasets():
    """Find available datasets for exploration."""
    datasets = []

    # Look for parquet files
    for parquet_file in DATA_DIR.rglob("*.parquet"):
        datasets.append(("parquet", parquet_file))

    # Look for CSV files
    for csv_file in DATA_DIR.rglob("*.csv"):
        datasets.append(("csv", csv_file))

    # Look for H5 files (embeddings)
    for h5_file in DATA_DIR.rglob("*.h5"):
        datasets.append(("h5", h5_file))

    return datasets


def create_sample_dataset():
    """Create a sample dataset if none exists."""
    import pandas as pd
    import numpy as np

    sample_dir = DATA_DIR / "sample"
    sample_dir.mkdir(parents=True, exist_ok=True)

    sample_file = sample_dir / "sample_dataset.parquet"
    if not sample_file.exists():
        print("[Spotlight] Creating sample dataset...")

        # Create sample data
        n_samples = 100
        np.random.seed(42)

        df = pd.DataFrame({
            "id": range(n_samples),
            "category": np.random.choice(["cat", "dog", "bird"], n_samples),
            "score": np.random.rand(n_samples),
            "embedding_x": np.random.randn(n_samples),
            "embedding_y": np.random.randn(n_samples),
            "text": [f"Sample text {i}" for i in range(n_samples)],
        })

        df.to_parquet(sample_file)
        print(f"[Spotlight] Sample dataset created: {sample_file}")

    return sample_file


def main():
    """Start the Spotlight server."""
    from renumics import spotlight
    import pandas as pd

    print("=" * 60)
    print("Renumics Spotlight - Data Exploration Server")
    print("=" * 60)
    print(f"Host: {HOST}")
    print(f"Port: {PORT}")
    print(f"Data directory: {DATA_DIR}")
    print()

    # Find or create dataset
    datasets = find_datasets()

    if datasets:
        print(f"[Spotlight] Found {len(datasets)} datasets:")
        for dtype, path in datasets[:5]:
            print(f"  - [{dtype}] {path}")
        if len(datasets) > 5:
            print(f"  ... and {len(datasets) - 5} more")

        # Load the first parquet or csv dataset
        dataset_file = None
        for dtype, path in datasets:
            if dtype in ("parquet", "csv"):
                dataset_file = path
                break

        if dataset_file:
            print(f"\n[Spotlight] Loading: {dataset_file}")
            if str(dataset_file).endswith(".parquet"):
                df = pd.read_parquet(dataset_file)
            else:
                df = pd.read_csv(dataset_file)
        else:
            df = pd.read_parquet(create_sample_dataset())
    else:
        print("[Spotlight] No datasets found, creating sample...")
        df = pd.read_parquet(create_sample_dataset())

    print(f"[Spotlight] Dataset shape: {df.shape}")
    print(f"[Spotlight] Columns: {list(df.columns)}")
    print()

    # Start Spotlight
    print(f"[Spotlight] Starting server at http://{HOST}:{PORT}")
    print("=" * 60)

    spotlight.show(
        df,
        host=HOST,
        port=PORT,
        no_browser=True,
        wait=True
    )


if __name__ == "__main__":
    main()
