# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo",
#     "pandas",
#     "numpy",
#     "pillow",
#     "matplotlib",
#     "fiftyone",
# ]
# ///
"""
Data Exploration Notebook
=========================
Interactive data exploration using Marimo.

Run with:
    marimo run notebooks/explore_data.py
    marimo edit notebooks/explore_data.py  # For editing
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
        # Data Exploration

        Explore your collected and annotated datasets using this interactive notebook.
        """
    )
    return


@app.cell
def __():
    from pathlib import Path
    import pandas as pd
    import numpy as np
    from PIL import Image
    import matplotlib.pyplot as plt

    # Data directories
    DATA_DIR = Path("./data")
    COLLECTED_DIR = DATA_DIR / "collected"
    PROCESSED_DIR = DATA_DIR / "processed"

    return DATA_DIR, COLLECTED_DIR, PROCESSED_DIR, Path, pd, np, Image, plt


@app.cell
def __(mo, COLLECTED_DIR, Path):
    # Find all images
    image_extensions = {".jpg", ".jpeg", ".png", ".webp"}
    images = [
        f for f in COLLECTED_DIR.rglob("*")
        if f.suffix.lower() in image_extensions
    ]

    mo.md(f"**Found {len(images)} images in `{COLLECTED_DIR}`**")
    return images, image_extensions


@app.cell
def __(mo, images):
    # Slider to select image
    if images:
        image_slider = mo.ui.slider(
            start=0,
            stop=max(0, len(images) - 1),
            step=1,
            label="Select image",
            value=0
        )
    else:
        image_slider = None

    image_slider
    return (image_slider,)


@app.cell
def __(mo, images, image_slider, Image, Path):
    if images and image_slider is not None:
        selected_image = images[image_slider.value]
        img = Image.open(selected_image)

        # Check for caption
        caption_file = selected_image.with_suffix(".txt")
        caption = caption_file.read_text().strip() if caption_file.exists() else "No caption"

        mo.vstack([
            mo.md(f"### {selected_image.name}"),
            mo.md(f"**Size:** {img.size[0]} x {img.size[1]}"),
            mo.md(f"**Caption:** {caption}"),
            mo.image(img, width=512),
        ])
    else:
        mo.md("*No images found. Run the collect pipeline first.*")
    return


@app.cell
def __(mo):
    mo.md(
        """
        ## Dataset Statistics

        Summary statistics for the collected dataset.
        """
    )
    return


@app.cell
def __(images, pd, Image, Path):
    # Compute statistics
    if images:
        stats = []
        for img_path in images[:100]:  # Sample first 100
            try:
                img = Image.open(img_path)
                caption_file = img_path.with_suffix(".txt")
                caption_len = len(caption_file.read_text()) if caption_file.exists() else 0
                stats.append({
                    "filename": img_path.name,
                    "width": img.size[0],
                    "height": img.size[1],
                    "aspect_ratio": img.size[0] / img.size[1],
                    "caption_length": caption_len,
                })
            except Exception:
                pass

        stats_df = pd.DataFrame(stats)
        stats_df.describe()
    else:
        stats_df = pd.DataFrame()
    return (stats_df,)


@app.cell
def __(mo, stats_df, plt):
    if not stats_df.empty:
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))

        axes[0].hist(stats_df["width"], bins=20, edgecolor="black")
        axes[0].set_title("Image Widths")
        axes[0].set_xlabel("Width (px)")

        axes[1].hist(stats_df["height"], bins=20, edgecolor="black")
        axes[1].set_title("Image Heights")
        axes[1].set_xlabel("Height (px)")

        axes[2].hist(stats_df["caption_length"], bins=20, edgecolor="black")
        axes[2].set_title("Caption Lengths")
        axes[2].set_xlabel("Characters")

        plt.tight_layout()
        mo.mpl.interactive(fig)
    return


@app.cell
def __(mo):
    mo.md(
        """
        ## FiftyOne Integration

        Launch FiftyOne for advanced dataset visualization.
        """
    )
    return


@app.cell
def __(mo):
    launch_fiftyone = mo.ui.button(label="Launch FiftyOne")
    launch_fiftyone
    return (launch_fiftyone,)


@app.cell
def __(launch_fiftyone, mo, COLLECTED_DIR):
    if launch_fiftyone.value:
        try:
            import fiftyone as fo

            # Create or load dataset
            dataset_name = "marimo_explore"
            if dataset_name in fo.list_datasets():
                dataset = fo.load_dataset(dataset_name)
            else:
                dataset = fo.Dataset.from_dir(
                    str(COLLECTED_DIR),
                    dataset_type=fo.types.ImageDirectory,
                    name=dataset_name,
                )

            session = fo.launch_app(dataset, port=5152)
            mo.md(f"**FiftyOne launched!** Open http://localhost:5152")
        except ImportError:
            mo.md("*FiftyOne not installed. Install with `pip install fiftyone`*")
    return


if __name__ == "__main__":
    app.run()
