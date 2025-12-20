# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo",
#     "pandas",
#     "numpy",
#     "matplotlib",
#     "aim",
# ]
# ///
"""
Experiment Analysis Notebook
============================
Analyze AIM experiment runs using Marimo.

Run with:
    marimo run notebooks/analyze_experiments.py
    marimo edit notebooks/analyze_experiments.py
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
        # Experiment Analysis

        Analyze and compare AIM experiment runs.
        """
    )
    return


@app.cell
def __():
    from pathlib import Path
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt

    # AIM repo path
    AIM_REPO = Path("./outputs/aim")

    return AIM_REPO, Path, pd, np, plt


@app.cell
def __(mo, AIM_REPO):
    # Check if AIM repo exists
    if AIM_REPO.exists():
        mo.md(f"**AIM Repository:** `{AIM_REPO}`")
    else:
        mo.md(f"*AIM repository not found at `{AIM_REPO}`. Run some experiments first.*")
    return


@app.cell
def __(AIM_REPO):
    # Try to import AIM and load runs
    try:
        from aim import Repo

        repo = Repo(str(AIM_REPO)) if AIM_REPO.exists() else None
        runs = list(repo.iter_runs()) if repo else []
        aim_available = True
    except ImportError:
        repo = None
        runs = []
        aim_available = False
    except Exception as e:
        repo = None
        runs = []
        aim_available = True  # Available but repo issue

    return repo, runs, aim_available


@app.cell
def __(mo, runs, aim_available):
    if not aim_available:
        mo.md("*AIM not installed. Install with `pip install aim`*")
    elif not runs:
        mo.md("*No experiment runs found. Train some models first!*")
    else:
        mo.md(f"**Found {len(runs)} experiment runs**")
    return


@app.cell
def __(mo, runs):
    # Create run selector
    if runs:
        run_options = {
            f"{r.name} ({r.experiment})": r.hash
            for r in runs
        }
        run_selector = mo.ui.multiselect(
            options=run_options,
            label="Select runs to compare",
            value=list(run_options.keys())[:3]  # Default to first 3
        )
    else:
        run_selector = None

    run_selector
    return (run_selector,)


@app.cell
def __(mo, runs, run_selector, pd):
    # Build runs table
    if runs and run_selector and run_selector.value:
        selected_hashes = [run_selector.value[name] for name in run_selector.value]

        run_data = []
        for r in runs:
            if r.hash in selected_hashes:
                hparams = r.get("hparams") or {}
                summary = r.get("summary") or {}

                run_data.append({
                    "Name": r.name,
                    "Experiment": r.experiment,
                    "Hash": r.hash[:8],
                    "Created": str(r.created_at)[:19],
                    "Epochs": hparams.get("train", {}).get("epochs", "N/A"),
                    "LR": hparams.get("train", {}).get("learning_rate", "N/A"),
                    "Best Loss": summary.get("best_loss", "N/A"),
                })

        runs_df = pd.DataFrame(run_data)
        mo.ui.table(runs_df)
    else:
        mo.md("*Select runs to display*")
    return


@app.cell
def __(mo):
    mo.md(
        """
        ## Training Curves

        Compare loss curves across selected runs.
        """
    )
    return


@app.cell
def __(mo, runs, run_selector, plt, pd):
    if runs and run_selector and run_selector.value:
        selected_hashes = [run_selector.value[name] for name in run_selector.value]

        fig, ax = plt.subplots(figsize=(10, 6))

        for r in runs:
            if r.hash in selected_hashes:
                # Get loss metrics
                try:
                    for metric in r.metrics():
                        if "loss" in metric.name.lower():
                            values = list(metric.values.values())
                            steps = list(metric.values.keys())
                            if values:
                                ax.plot(steps, values, label=f"{r.name} - {metric.name}")
                except Exception:
                    pass

        ax.set_xlabel("Step/Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("Training Loss Comparison")
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        mo.mpl.interactive(fig)
    return


@app.cell
def __(mo):
    mo.md(
        """
        ## Hyperparameter Analysis

        Analyze hyperparameters across runs.
        """
    )
    return


@app.cell
def __(mo, runs, pd):
    if runs:
        # Extract hyperparameters from all runs
        hparam_data = []
        for r in runs:
            hparams = r.get("hparams") or {}
            train_cfg = hparams.get("train", {})
            lora_cfg = hparams.get("lora", {})
            summary = r.get("summary") or {}

            hparam_data.append({
                "run": r.name,
                "epochs": train_cfg.get("epochs"),
                "batch_size": train_cfg.get("batch_size"),
                "learning_rate": train_cfg.get("learning_rate"),
                "lora_rank": lora_cfg.get("rank"),
                "best_loss": summary.get("best_loss"),
            })

        hparam_df = pd.DataFrame(hparam_data)

        # Show correlation if we have numeric data
        numeric_cols = hparam_df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 1:
            mo.md("### Hyperparameter Correlation")
            mo.ui.table(hparam_df[numeric_cols].corr().round(3))
        else:
            mo.ui.table(hparam_df)
    return


@app.cell
def __(mo):
    mo.md(
        """
        ## Launch AIM UI

        Open the full AIM web interface for detailed analysis.
        """
    )
    return


@app.cell
def __(mo):
    aim_ui_button = mo.ui.button(label="Launch AIM UI")
    aim_ui_button
    return (aim_ui_button,)


@app.cell
def __(aim_ui_button, mo, AIM_REPO):
    if aim_ui_button.value:
        import subprocess
        try:
            # Start AIM UI in background
            subprocess.Popen(
                ["aim", "up", "--port", "43801", "--repo", str(AIM_REPO)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            mo.md("**AIM UI launched!** Open http://localhost:43801")
        except Exception as e:
            mo.md(f"*Failed to launch AIM UI: {e}*")
    return


if __name__ == "__main__":
    app.run()
