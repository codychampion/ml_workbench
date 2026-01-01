"""
Hydra-AIM Integration Utilities
===============================
Provides seamless integration between Hydra configuration management
and AIM experiment tracking.

Usage:
    from utils.hydra_aim import init_aim_from_hydra, AimCallback

    @hydra.main(config_path="../conf", config_name="config")
    def train(cfg: DictConfig):
        run = init_aim_from_hydra(cfg)
        # Training code...
        run.close()
"""

import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any

from omegaconf import DictConfig, OmegaConf
from utils.git_utils import get_git_info

# AIM integration
try:
    from aim import Run, Image as AimImage
    from aim.sdk.objects import Distribution
    AIM_AVAILABLE = True
except ImportError:
    AIM_AVAILABLE = False
    Run = None


def flatten_dict(d: Dict, parent_key: str = '', sep: str = '.') -> Dict[str, Any]:
    """
    Flatten a nested dictionary for easier filtering in AIM UI.

    Args:
        d: Dictionary to flatten
        parent_key: Prefix for keys (used in recursion)
        sep: Separator between nested keys

    Returns:
        Flattened dictionary with dot-separated keys
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def init_aim_from_hydra(
    cfg: DictConfig,
    run_name: Optional[str] = None,
    experiment: Optional[str] = None,
    tags: Optional[list] = None,
) -> Optional[Run]:
    """
    Initialize AIM run from Hydra configuration.

    Args:
        cfg: Hydra DictConfig object
        run_name: Optional custom run name
        experiment: Optional experiment name override
        tags: Optional additional tags

    Returns:
        AIM Run object or None if AIM is not available
    """
    if not AIM_AVAILABLE:
        print("[Warning] AIM not installed, experiment tracking disabled")
        return None

    # Get AIM settings from config
    aim_cfg = cfg.get("aim", cfg.get("infrastructure", {}).get("aim", {}))

    # Determine repo path
    repo = aim_cfg.get("repo", "./outputs/aim")
    if "${" in str(repo):
        # Resolve any remaining interpolations
        repo = OmegaConf.to_container(OmegaConf.create({"repo": repo}), resolve=True)["repo"]

    # Create repo directory if needed
    Path(repo).mkdir(parents=True, exist_ok=True)

    # Get experiment name
    exp_name = experiment or aim_cfg.get("experiment", "default")
    if hasattr(cfg, "experiment") and hasattr(cfg.experiment, "name"):
        exp_name = cfg.experiment.name

    # Initialize AIM run
    run = Run(
        repo=str(repo),
        experiment=exp_name,
    )

    # Set run name
    if run_name:
        run.name = run_name
    elif hasattr(cfg, "experiment") and hasattr(cfg.experiment, "name"):
        run.name = f"{cfg.experiment.name}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    else:
        run.name = f"run-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

    # Log Hydra configuration
    log_hydra_config(run, cfg)

    # Add tags
    all_tags = tags or []
    if hasattr(cfg, "experiment") and hasattr(cfg.experiment, "tags"):
        all_tags.extend(cfg.experiment.tags)

    for tag in all_tags:
        run.add_tag(tag)

    # Log system info if enabled
    if aim_cfg.get("log_system_params", True):
        run["system"] = {
            "python_version": sys.version,
            "cwd": os.getcwd(),
            "device": cfg.get("compute", {}).get("device", "cpu"),
        }

    # Log git info for code traceability
    git_info = get_git_info()
    run["git"] = git_info
    run["git_hash"] = git_info.get("commit", "")

    # Warn if working directory is dirty
    if git_info.get("dirty"):
        print(f"[AIM] WARNING: Uncommitted changes detected! Results may not be reproducible.")
        print(f"[AIM] Uncommitted files: {git_info.get('uncommitted_files', [])}")

    print(f"[AIM] Initialized run: {run.name}")
    print(f"[AIM] Experiment: {exp_name}")
    print(f"[AIM] Repo: {repo}")
    print(f"[AIM] Git commit: {git_info.get('commit_short', 'unknown')} ({git_info.get('branch', 'unknown')})")

    return run


def log_hydra_config(run: Run, cfg: DictConfig) -> None:
    """
    Log Hydra configuration to AIM run for experiment tracking.

    Logs both nested and flattened versions:
    - Nested: Full config structure under run["hparams"]
    - Flattened: Dot-separated keys for easy filtering in AIM UI

    Args:
        run: AIM Run object
        cfg: Hydra DictConfig object with experiment configuration
    """
    if run is None:
        return

    # Convert to container and resolve interpolations
    config_dict = OmegaConf.to_container(cfg, resolve=True)

    # Log as hyperparameters
    run["hparams"] = config_dict

    # Also log flattened version for easier filtering in AIM UI
    flattened = flatten_dict(config_dict)
    for key, value in flattened.items():
        if isinstance(value, (int, float, str, bool)):
            run.set(key, value, strict=False)


class AimCallback:
    """
    Callback for integrating AIM with training loops.

    Usage:
        callback = AimCallback(run)

        for epoch in range(epochs):
            loss = train_epoch(...)
            callback.on_epoch_end(epoch, {"loss": loss})

        callback.on_training_end()
    """

    def __init__(self, run: Optional[Run], log_frequency: int = 1):
        """
        Initialize callback.

        Args:
            run: AIM Run object
            log_frequency: How often to log metrics (every N steps)
        """
        self.run = run
        self.log_frequency = log_frequency
        self.step = 0

    def on_step_end(
        self,
        step: int,
        metrics: Dict[str, Any],
        context: Optional[Dict[str, str]] = None
    ) -> None:
        """Log metrics at the end of a training step."""
        if self.run is None:
            return

        self.step = step

        if step % self.log_frequency == 0:
            for name, value in metrics.items():
                if isinstance(value, (int, float)):
                    self.run.track(value, name=name, step=step, context=context)

    def on_epoch_end(
        self,
        epoch: int,
        metrics: Dict[str, Any],
        context: Optional[Dict[str, str]] = None
    ) -> None:
        """Log metrics at the end of an epoch."""
        if self.run is None:
            return

        ctx = context or {}
        ctx["subset"] = ctx.get("subset", "train")

        for name, value in metrics.items():
            if isinstance(value, (int, float)):
                self.run.track(value, name=name, epoch=epoch, context=ctx)

    def log_image(
        self,
        image,
        name: str,
        step: Optional[int] = None,
        caption: Optional[str] = None,
        context: Optional[Dict[str, str]] = None
    ) -> None:
        """Log an image to AIM."""
        if self.run is None or not AIM_AVAILABLE:
            return

        aim_image = AimImage(image, caption=caption)
        self.run.track(aim_image, name=name, step=step or self.step, context=context)

    def log_distribution(
        self,
        values,
        name: str,
        step: Optional[int] = None,
        context: Optional[Dict[str, str]] = None
    ) -> None:
        """Log a distribution (e.g., gradient norms, weights)."""
        if self.run is None or not AIM_AVAILABLE:
            return

        dist = Distribution(values)
        self.run.track(dist, name=name, step=step or self.step, context=context)

    def on_training_end(self, summary: Optional[Dict[str, Any]] = None) -> None:
        """Called at the end of training."""
        if self.run is None:
            return

        if summary:
            self.run["summary"] = summary

    def close(self) -> None:
        """Close the AIM run."""
        if self.run is not None:
            self.run.close()


def get_hydra_config_from_run(run: Run) -> Optional[Dict]:
    """
    Retrieve Hydra configuration from an existing AIM run.

    Useful for reproducing experiments or comparing configurations.

    Args:
        run: AIM Run object to retrieve config from

    Returns:
        Configuration dictionary if available, None otherwise
    """
    if run is None:
        return None

    return run.get("hparams")
