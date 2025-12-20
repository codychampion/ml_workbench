# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo",
#     "torch",
#     "transformers",
#     "hydra-core",
#     "omegaconf",
#     "aim",
# ]
# ///
"""
Model Training Notebook
=======================
Interactive model training with Hydra config and AIM tracking.

Run with:
    marimo run notebooks/train_model.py
    marimo edit notebooks/train_model.py
"""

import marimo

__generated_with = "0.10.0"
app = marimo.App(width="full")


@app.cell
def __():
    import marimo as mo
    return (mo,)


@app.cell
def __(mo):
    mo.md(
        """
        # Interactive Model Training

        Configure and run training experiments with live monitoring.
        """
    )
    return


@app.cell
def __():
    import sys
    from pathlib import Path

    # Add project root to path
    PROJECT_ROOT = Path(".")
    sys.path.insert(0, str(PROJECT_ROOT))

    from omegaconf import OmegaConf

    return PROJECT_ROOT, Path, sys, OmegaConf


@app.cell
def __(mo):
    mo.md("## Training Configuration")
    return


@app.cell
def __(mo):
    # Training parameters
    epochs_input = mo.ui.slider(1, 20, value=3, step=1, label="Epochs")
    batch_size_input = mo.ui.dropdown(
        options=["1", "2", "4", "8", "16"],
        value="4",
        label="Batch Size"
    )
    lr_input = mo.ui.dropdown(
        options=["1e-5", "5e-5", "1e-4", "2e-4", "5e-4"],
        value="1e-4",
        label="Learning Rate"
    )

    mo.vstack([
        mo.hstack([epochs_input, batch_size_input, lr_input]),
    ])
    return epochs_input, batch_size_input, lr_input


@app.cell
def __(mo):
    # LoRA parameters
    lora_enabled = mo.ui.checkbox(label="Enable LoRA", value=True)
    lora_rank = mo.ui.slider(4, 64, value=8, step=4, label="LoRA Rank")
    lora_alpha = mo.ui.slider(8, 128, value=16, step=8, label="LoRA Alpha")

    mo.hstack([lora_enabled, lora_rank, lora_alpha])
    return lora_enabled, lora_rank, lora_alpha


@app.cell
def __(mo):
    # Model selection
    model_select = mo.ui.dropdown(
        options=["blip-base", "blip-large", "git-base", "git-large"],
        value="blip-base",
        label="Model"
    )
    model_select
    return (model_select,)


@app.cell
def __(mo, epochs_input, batch_size_input, lr_input, lora_enabled, lora_rank, lora_alpha, model_select, OmegaConf):
    # Build configuration
    config = OmegaConf.create({
        "train": {
            "epochs": epochs_input.value,
            "batch_size": int(batch_size_input.value),
            "learning_rate": float(lr_input.value),
        },
        "lora": {
            "enabled": lora_enabled.value,
            "rank": lora_rank.value,
            "alpha": lora_alpha.value,
        },
        "model": {
            "name": model_select.value,
        },
        "experiment": {
            "name": f"interactive-{model_select.value}",
            "tags": ["marimo", "interactive"],
        }
    })

    mo.md(f"```yaml\n{OmegaConf.to_yaml(config)}\n```")
    return (config,)


@app.cell
def __(mo):
    mo.md("## Training Execution")
    return


@app.cell
def __(mo):
    start_training = mo.ui.button(label="Start Training", kind="success")
    stop_training = mo.ui.button(label="Stop Training", kind="danger")

    mo.hstack([start_training, stop_training])
    return start_training, stop_training


@app.cell
def __(mo, start_training, config):
    # Training state
    training_status = mo.state("idle")
    training_logs = mo.state([])

    if start_training.value:
        training_status.set("running")
        training_logs.set(["Training started...", f"Config: {config}"])

    mo.md(f"**Status:** {training_status.value}")
    return training_status, training_logs


@app.cell
def __(mo, training_logs):
    # Display logs
    if training_logs.value:
        log_text = "\n".join(training_logs.value[-20:])  # Last 20 lines
        mo.md(f"```\n{log_text}\n```")
    return


@app.cell
def __(mo):
    mo.md(
        """
        ## Training Progress

        Live metrics will appear here once training starts.
        """
    )
    return


@app.cell
def __(mo):
    # Placeholder for training charts
    mo.md("*Training metrics will be displayed here*")
    return


@app.cell
def __(mo):
    mo.md(
        """
        ---

        ## Notes

        - Training runs are tracked in AIM (http://localhost:43800)
        - Checkpoints are saved to `outputs/training/`
        - Use the CLI for production training:

        ```bash
        python -m pipelines.train.finetune train.epochs=10 lora.rank=16
        ```
        """
    )
    return


if __name__ == "__main__":
    app.run()
