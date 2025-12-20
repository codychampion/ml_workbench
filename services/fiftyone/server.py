#!/usr/bin/env python3
"""
FiftyOne Server Launcher
========================
Launches FiftyOne App as a persistent service for data visualization.
Automatically loads datasets from the configured directory.
"""

import os
import sys
import time
from pathlib import Path

# Configure FiftyOne to use external MongoDB before importing
database_uri = os.getenv("FIFTYONE_DATABASE_URI")
if database_uri:
    os.environ["FIFTYONE_DATABASE_URI"] = database_uri
    print(f"[FiftyOne] Using external MongoDB: {database_uri.split('@')[-1] if '@' in database_uri else database_uri}")

import fiftyone as fo
import fiftyone.zoo as foz


class FiftyOneConfig:
    """FiftyOne server configuration."""
    ADDRESS = os.getenv("FIFTYONE_APP_ADDRESS", "0.0.0.0")
    PORT = int(os.getenv("FIFTYONE_APP_PORT", "5151"))
    DATABASE_URI = os.getenv("FIFTYONE_DATABASE_URI")
    DATASET_DIR = Path(os.getenv("FIFTYONE_DEFAULT_DATASET_DIR", "/workspace/fiftyone/datasets"))
    AUTO_LOAD_QUICKSTART = os.getenv("FIFTYONE_AUTO_LOAD_QUICKSTART", "true").lower() == "true"


def ensure_sample_dataset():
    """Ensure a sample dataset exists for demonstration."""
    dataset_name = "quickstart-sample"

    if dataset_name in fo.list_datasets():
        print(f"[FiftyOne] Loading existing dataset: {dataset_name}")
        return fo.load_dataset(dataset_name)

    if FiftyOneConfig.AUTO_LOAD_QUICKSTART:
        print(f"[FiftyOne] Downloading quickstart dataset...")
        try:
            dataset = foz.load_zoo_dataset(
                "quickstart",
                max_samples=50,
                shuffle=True,
                seed=42,
                dataset_name=dataset_name
            )
            dataset.persistent = True
            print(f"[FiftyOne] Created dataset with {len(dataset)} samples")
            return dataset
        except Exception as e:
            print(f"[FiftyOne] Warning: Could not load quickstart dataset: {e}")
            print("[FiftyOne] Creating empty placeholder dataset...")
            dataset = fo.Dataset(dataset_name)
            dataset.persistent = True
            return dataset

    return None


def list_available_datasets():
    """List all available datasets."""
    datasets = fo.list_datasets()
    print(f"\n[FiftyOne] Available datasets ({len(datasets)}):")
    for name in datasets:
        ds = fo.load_dataset(name)
        print(f"  - {name}: {len(ds)} samples")
    print()


def main():
    """Launch FiftyOne server."""
    print("=" * 60)
    print("FiftyOne Visualization Server")
    print("=" * 60)
    print(f"Address: {FiftyOneConfig.ADDRESS}")
    print(f"Port: {FiftyOneConfig.PORT}")
    if FiftyOneConfig.DATABASE_URI:
        db_display = FiftyOneConfig.DATABASE_URI.split('@')[-1] if '@' in FiftyOneConfig.DATABASE_URI else FiftyOneConfig.DATABASE_URI
        print(f"Database: {db_display}")
    print(f"Datasets: {FiftyOneConfig.DATASET_DIR}")
    print("=" * 60)

    # Ensure directories exist
    FiftyOneConfig.DATASET_DIR.mkdir(parents=True, exist_ok=True)

    # Load or create sample dataset
    dataset = ensure_sample_dataset()
    list_available_datasets()

    # Launch the app
    print(f"\n[FiftyOne] Starting app at http://{FiftyOneConfig.ADDRESS}:{FiftyOneConfig.PORT}")
    print("[FiftyOne] Press Ctrl+C to stop\n")

    try:
        # Launch with the sample dataset if available
        # Use remote=True to indicate headless mode (no browser expected)
        if dataset is not None:
            session = fo.launch_app(
                dataset,
                address=FiftyOneConfig.ADDRESS,
                port=FiftyOneConfig.PORT,
                remote=True,  # Headless server mode
                auto=False    # Don't open browser
            )
        else:
            session = fo.launch_app(
                address=FiftyOneConfig.ADDRESS,
                port=FiftyOneConfig.PORT,
                remote=True,
                auto=False
            )

        print(f"[FiftyOne] Server started successfully on port {FiftyOneConfig.PORT}")

        # Keep the server running with a simple loop
        # This is more reliable than session.wait() in headless environments
        while True:
            time.sleep(60)

    except KeyboardInterrupt:
        print("\n[FiftyOne] Shutting down...")
    except Exception as e:
        print(f"\n[FiftyOne] Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
