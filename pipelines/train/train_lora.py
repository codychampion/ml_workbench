#!/usr/bin/env python3
"""
LoRA Training Pipeline
======================
Symbolic LoRA training for pipeline validation.
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config import get_config
from utils.storage import get_s3_client

try:
    from aim import Run
    AIM_AVAILABLE = True
except ImportError:
    AIM_AVAILABLE = False

# Optional Hydra integration for config-driven runs
try:
    import hydra
    from omegaconf import DictConfig
except ImportError:
    hydra = None
    DictConfig = Any


class SymbolicLoRAModel(nn.Module):
    """Symbolic LoRA model for testing."""

    def __init__(self, input_dim: int = 768, hidden_dim: int = 256, lora_rank: int = 8):
        super().__init__()
        self.base_layer = nn.Linear(input_dim, hidden_dim)
        self.base_layer.requires_grad = False
        self.lora_A = nn.Linear(input_dim, lora_rank, bias=False)
        self.lora_B = nn.Linear(lora_rank, hidden_dim, bias=False)
        nn.init.kaiming_uniform_(self.lora_A.weight)
        nn.init.zeros_(self.lora_B.weight)
        self.output = nn.Linear(hidden_dim, input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.output(self.base_layer(x) + self.lora_B(self.lora_A(x)))

    def get_lora_state_dict(self) -> Dict[str, torch.Tensor]:
        return {"lora_A": self.lora_A.state_dict(), "lora_B": self.lora_B.state_dict()}


class SymbolicDataset(torch.utils.data.Dataset):
    """Symbolic dataset for testing."""

    def __init__(self, size: int = 100, input_dim: int = 768):
        self.data = torch.randn(size, input_dim)
        self.targets = torch.randn(size, input_dim)

    def __len__(self): return len(self.data)
    def __getitem__(self, idx): return self.data[idx], self.targets[idx]


def train_epoch(model, dataloader, optimizer, criterion, device, epoch) -> float:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    for batch_idx, (data, target) in enumerate(dataloader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        loss = criterion(model(data), target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        if batch_idx % 10 == 0:
            print(f"[Train] Epoch {epoch} | Batch {batch_idx}/{len(dataloader)} | Loss: {loss.item():.4f}")
    return total_loss / len(dataloader)


def run_training(output_dir: Path, num_epochs: int = 10, batch_size: int = 16,
                 learning_rate: float = 1e-4, lora_rank: int = 8, dataset_size: int = 100, seed: int = 42):
    """Run the LoRA training pipeline."""
    config = get_config()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(seed)
    np.random.seed(seed)
    device = config.compute.device

    train_config = {
        "num_epochs": num_epochs, "batch_size": batch_size, "learning_rate": learning_rate,
        "lora_rank": lora_rank, "dataset_size": dataset_size, "seed": seed, "device": device,
    }

    # Initialize AIM
    run = None
    if AIM_AVAILABLE:
        run = Run(repo=str(config.aim.repo), experiment=config.aim.experiment)
        run.name = f"lora-train-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        run["hparams"] = train_config

    # Initialize S3 client
    s3 = get_s3_client("mlops-models")

    # Create model and data
    dataset = SymbolicDataset(size=dataset_size)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model = SymbolicLoRAModel(lora_rank=lora_rank).to(device)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=learning_rate)
    criterion = nn.MSELoss()

    # Training loop
    best_loss = float("inf")
    for epoch in range(1, num_epochs + 1):
        train_loss = train_epoch(model, dataloader, optimizer, criterion, device, epoch)
        if run:
            run.track(train_loss, name="loss", epoch=epoch)
        print(f"[Epoch {epoch}] Loss: {train_loss:.4f}")

        if train_loss < best_loss:
            best_loss = train_loss
            checkpoint_path = output_dir / "best_lora.pt"
            torch.save({"epoch": epoch, "lora_state_dict": model.get_lora_state_dict(),
                       "loss": train_loss, "config": train_config}, checkpoint_path)
            if s3:
                s3.upload_file(checkpoint_path, f"checkpoints/lora/best_lora_epoch{epoch}.pt")

    if run:
        run["summary"] = {"best_loss": best_loss, "final_loss": train_loss}
        run.close()

    print(f"\nTraining Complete! Best Loss: {best_loss:.4f}")


def main():
    parser = argparse.ArgumentParser(description="LoRA Training Pipeline")
    parser.add_argument("--output-dir", type=Path, default=Path("./outputs/lora-checkpoints"))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_training(output_dir=args.output_dir, num_epochs=args.epochs, batch_size=args.batch_size,
                 learning_rate=args.lr, lora_rank=args.lora_rank, seed=args.seed)


def _run_from_hydra(cfg: "DictConfig"):
    """
    Launch training using Hydra config files so experiments stay reproducible.
    """
    train_cfg = cfg.get("train", {})
    lora_cfg = cfg.get("lora", {})
    paths_cfg = cfg.get("paths", {})
    compute_cfg = cfg.get("compute", {})

    output_dir = Path(paths_cfg.get("checkpoints", "./outputs/lora-checkpoints"))

    run_training(
        output_dir=output_dir,
        num_epochs=train_cfg.get("epochs", 10),
        batch_size=train_cfg.get("batch_size", 16),
        learning_rate=train_cfg.get("learning_rate", 1e-4),
        lora_rank=lora_cfg.get("rank", 8),
        dataset_size=train_cfg.get("dataset_size", 100),
        seed=compute_cfg.get("seed", 42),
    )


if hydra is not None:
    @hydra.main(config_path="../../conf", config_name="config")
    def hydra_main(cfg: "DictConfig"):  # type: ignore[misc]
        _run_from_hydra(cfg)


if __name__ == "__main__":
    if hydra is not None and "--hydra" in sys.argv:
        hydra_main()
    else:
        main()
