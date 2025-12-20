#!/usr/bin/env python3
"""
LoRA Training Pipeline
======================
Phase 1: Symbolic training for pipeline validation (CPU-only)
Phase 2/3: Full LoRA training with PEFT and GPU acceleration

This script demonstrates:
1. Dataset loading from B2 storage (mocked)
2. Model preparation for LoRA training
3. Training loop with AIM logging
4. Checkpoint saving and upload
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any

import numpy as np
import torch
import torch.nn as nn

# Prefect for workflow orchestration
try:
    from prefect import flow, task
    PREFECT_AVAILABLE = True
except ImportError:
    PREFECT_AVAILABLE = False
    # Provide no-op decorators when Prefect is not available
    def flow(*args, **kwargs):
        def decorator(fn):
            return fn
        return decorator if not args or callable(args[0]) else decorator
    def task(*args, **kwargs):
        def decorator(fn):
            return fn
        return decorator if not args or callable(args[0]) else decorator

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config import get_config
from data_transfer import B2Client

# AIM for experiment tracking
try:
    from aim import Run
    AIM_AVAILABLE = True
except ImportError:
    AIM_AVAILABLE = False
    print("[Warning] AIM not installed, metrics logging disabled")


class SymbolicLoRAModel(nn.Module):
    """
    Symbolic LoRA model for Phase 1 testing.

    Demonstrates the LoRA training workflow without actual large model weights.

    PHASE 2/3 TODO: Replace with actual LoRA implementation:
    from peft import LoraConfig, get_peft_model
    from diffusers import StableDiffusionPipeline

    model = StableDiffusionPipeline.from_pretrained("...")
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
        lora_dropout=0.1,
    )
    model = get_peft_model(model, lora_config)
    """

    def __init__(self, input_dim: int = 768, hidden_dim: int = 256, lora_rank: int = 8):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.lora_rank = lora_rank

        # Base model (frozen in real LoRA)
        self.base_layer = nn.Linear(input_dim, hidden_dim)
        self.base_layer.requires_grad = False

        # LoRA adapters (A and B matrices)
        self.lora_A = nn.Linear(input_dim, lora_rank, bias=False)
        self.lora_B = nn.Linear(lora_rank, hidden_dim, bias=False)

        # Initialize LoRA weights
        nn.init.kaiming_uniform_(self.lora_A.weight)
        nn.init.zeros_(self.lora_B.weight)

        # Output projection
        self.output = nn.Linear(hidden_dim, input_dim)

        print(f"[SymbolicLoRA] Initialized with rank={lora_rank}")
        print(f"[SymbolicLoRA] Base params: {sum(p.numel() for p in self.base_layer.parameters()):,}")
        print(f"[SymbolicLoRA] LoRA params: {sum(p.numel() for p in [self.lora_A.weight, self.lora_B.weight]):,}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Base model output (frozen)
        base_out = self.base_layer(x)

        # LoRA adapter output
        lora_out = self.lora_B(self.lora_A(x))

        # Combine
        combined = base_out + lora_out
        return self.output(combined)

    def get_lora_state_dict(self) -> Dict[str, torch.Tensor]:
        """Get only the LoRA parameters for saving."""
        return {
            "lora_A": self.lora_A.state_dict(),
            "lora_B": self.lora_B.state_dict(),
        }


class SymbolicDataset(torch.utils.data.Dataset):
    """Symbolic dataset for Phase 1 testing."""

    def __init__(self, size: int = 100, input_dim: int = 768):
        self.size = size
        self.input_dim = input_dim
        self.data = torch.randn(size, input_dim)
        self.targets = torch.randn(size, input_dim)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]


@task(name="init-aim-tracking")
def init_aim(config: Dict[str, Any], run_name: Optional[str] = None) -> Optional[Run]:
    """Initialize AIM for experiment tracking."""
    if not AIM_AVAILABLE:
        return None

    app_config = get_config()

    run = Run(
        repo=str(app_config.aim.repo),
        experiment=app_config.aim.experiment,
    )

    # Set run name and metadata
    run_name = run_name or f"lora-train-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    run.name = run_name

    # Log hyperparameters
    run["hparams"] = config

    # Add tags
    run.add_tag("lora")
    run.add_tag("training")
    run.add_tag("phase1")
    run.add_tag("cpu")

    print(f"[AIM] Initialized run: {run.name}")
    print(f"[AIM] Repo: {app_config.aim.repo}")
    return run


@task(name="train-epoch")
def train_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: str,
    epoch: int
) -> float:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0

    for batch_idx, (data, target) in enumerate(dataloader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if batch_idx % 10 == 0:
            print(f"[Train] Epoch {epoch} | Batch {batch_idx}/{len(dataloader)} | Loss: {loss.item():.4f}")

    return total_loss / len(dataloader)


@flow(name="lora-training-pipeline", log_prints=True)
def run_training_pipeline(
    output_dir: Path,
    num_epochs: int = 10,
    batch_size: int = 16,
    learning_rate: float = 1e-4,
    lora_rank: int = 8,
    dataset_size: int = 100,
    seed: int = 42
):
    """Run the LoRA training pipeline."""
    config = get_config()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set seed for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = config.compute.device
    print(f"[Pipeline] Device: {device}")

    # Training configuration
    train_config = {
        "num_epochs": num_epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "lora_rank": lora_rank,
        "dataset_size": dataset_size,
        "seed": seed,
        "device": device,
        "phase": "1-local-core"
    }

    # Step 1: Initialize AIM tracking
    run = init_aim(train_config)

    # Step 2: Initialize B2 client for data loading (mocked)
    b2_client = B2Client()
    available_files = b2_client.list_files(prefix="datasets/lora")
    print(f"[Pipeline] Found {len(available_files)} training files in storage")

    # Step 3: Create dataset and dataloader
    print("\n[Pipeline] Creating symbolic dataset...")
    dataset = SymbolicDataset(size=dataset_size)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True
    )

    # Step 4: Initialize model
    print("\n[Pipeline] Initializing LoRA model...")
    model = SymbolicLoRAModel(lora_rank=lora_rank).to(device)

    # Step 5: Setup training
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=learning_rate
    )
    criterion = nn.MSELoss()

    # Log model info (AIM doesn't have watch, log param counts instead)
    if run:
        run["model"] = {
            "type": "SymbolicLoRAModel",
            "lora_rank": lora_rank,
            "total_params": sum(p.numel() for p in model.parameters()),
            "trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
        }

    # Step 6: Training loop
    print("\n[Pipeline] Starting training...")
    best_loss = float("inf")

    for epoch in range(1, num_epochs + 1):
        print(f"\n{'='*40}")
        print(f"Epoch {epoch}/{num_epochs}")
        print(f"{'='*40}")

        train_loss = train_epoch(model, dataloader, optimizer, criterion, device, epoch)

        # Log metrics to AIM
        if run:
            run.track(train_loss, name="loss", epoch=epoch, context={"subset": "train"})
            run.track(optimizer.param_groups[0]["lr"], name="learning_rate", epoch=epoch)

        print(f"[Epoch {epoch}] Average Loss: {train_loss:.4f}")

        # Save best checkpoint
        if train_loss < best_loss:
            best_loss = train_loss
            checkpoint_path = output_dir / "best_lora.pt"
            torch.save({
                "epoch": epoch,
                "lora_state_dict": model.get_lora_state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": train_loss,
                "config": train_config,
            }, checkpoint_path)
            print(f"[Checkpoint] Saved best model to {checkpoint_path}")

            # Upload to B2 (mocked)
            b2_client.upload_file(
                source=checkpoint_path,
                destination_name=f"checkpoints/lora/best_lora_epoch{epoch}.pt"
            )

    # Save final checkpoint
    final_checkpoint_path = output_dir / "final_lora.pt"
    torch.save({
        "epoch": num_epochs,
        "lora_state_dict": model.get_lora_state_dict(),
        "config": train_config,
    }, final_checkpoint_path)

    # Log final summary to AIM
    if run:
        run["summary"] = {
            "best_loss": best_loss,
            "final_loss": train_loss,
            "total_epochs": num_epochs,
            "checkpoint_path": str(final_checkpoint_path),
        }
        run.close()

    print(f"\n{'='*60}")
    print("Training Complete!")
    print(f"{'='*60}")
    print(f"Best Loss: {best_loss:.4f}")
    print(f"Checkpoints: {output_dir}")
    if run:
        print(f"AIM Run: {run.name}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="LoRA Training Pipeline (Phase 1: Symbolic)"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./outputs/lora-checkpoints"),
        help="Directory to save checkpoints"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Training batch size"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate"
    )
    parser.add_argument(
        "--lora-rank",
        type=int,
        default=8,
        help="LoRA rank"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("LoRA Training Pipeline")
    print("Phase 1: Local Core (CPU-only, Symbolic Model)")
    print("=" * 60)

    run_training_pipeline(
        output_dir=args.output_dir,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        lora_rank=args.lora_rank,
        seed=args.seed
    )


if __name__ == "__main__":
    main()


# PHASE 2/3 TODO: Prefect workflow integration
# from prefect import flow, task
#
# @task(retries=3)
# def prepare_dataset_task(dataset_config: dict):
#     """Download and prepare training data from B2."""
#     pass
#
# @task
# def train_lora_task(model_config: dict, dataset, epochs: int):
#     """Run LoRA training with checkpointing."""
#     pass
#
# @task
# def upload_checkpoint_task(checkpoint_path: Path):
#     """Upload trained LoRA to B2 storage."""
#     pass
#
# @flow(name="lora-training-pipeline")
# def lora_training_flow(config: dict):
#     dataset = prepare_dataset_task(config["dataset"])
#     checkpoint = train_lora_task(config["model"], dataset, config["epochs"])
#     upload_checkpoint_task(checkpoint)


# PHASE 2/3 TODO: SkyPilot integration
# sky_config = """
# resources:
#   accelerators: A100:1
#   use_spot: true
#   disk_size: 256
#
# setup: |
#   pip install -r requirements.txt
#   pip install peft xformers
#
# run: |
#   python -m pipelines.train.train_lora --epochs 100 --device cuda
# """
