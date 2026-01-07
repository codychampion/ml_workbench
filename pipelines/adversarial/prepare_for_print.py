"""
Prepare Adversarial Patches for Physical Printing
==================================================
Convert digital patches to print-ready formats with proper resolution,
color profiles, and optional crop marks.

Usage:
    python -m pipelines.adversarial.prepare_for_print \
        --patch ./outputs/patches/punk_patch.png \
        --output ./outputs/print_ready/punk_patch_print.png \
        --size-inches 4 4 \
        --dpi 300
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import argparse
from typing import Tuple, Optional
import torch
from PIL import Image, ImageDraw, ImageFont
import numpy as np


def load_patch(patch_path: Path) -> np.ndarray:
    """Load patch image as numpy array."""
    img = Image.open(patch_path).convert('RGB')
    return np.array(img)


def upscale_patch(
    patch: np.ndarray,
    target_size_pixels: Tuple[int, int],
    method: str = "bicubic"
) -> np.ndarray:
    """
    Upscale patch to target print resolution.

    Args:
        patch: Input patch array [H, W, 3]
        target_size_pixels: (width, height) in pixels for print
        method: Resampling method (bicubic, lanczos, nearest)

    Returns:
        Upscaled patch array
    """
    img = Image.fromarray(patch)

    resample_methods = {
        "bicubic": Image.BICUBIC,
        "lanczos": Image.LANCZOS,
        "nearest": Image.NEAREST,
        "bilinear": Image.BILINEAR
    }

    resample = resample_methods.get(method, Image.BICUBIC)
    img_upscaled = img.resize(target_size_pixels, resample=resample)

    return np.array(img_upscaled)


def add_crop_marks(
    patch: np.ndarray,
    margin_pixels: int = 50
) -> np.ndarray:
    """
    Add crop marks around patch for cutting guidance.

    Args:
        patch: Input patch array [H, W, 3]
        margin_pixels: Margin size for crop marks

    Returns:
        Patch with crop marks
    """
    h, w = patch.shape[:2]

    # Create white canvas with margin
    canvas_h = h + 2 * margin_pixels
    canvas_w = w + 2 * margin_pixels
    canvas = np.ones((canvas_h, canvas_w, 3), dtype=np.uint8) * 255

    # Place patch in center
    canvas[margin_pixels:margin_pixels+h, margin_pixels:margin_pixels+w] = patch

    # Convert to PIL for drawing
    img = Image.fromarray(canvas)
    draw = ImageDraw.Draw(img)

    # Crop mark length
    mark_len = 20
    mark_offset = 5  # Distance from edge

    # Corner positions (where patch starts/ends)
    corners = [
        (margin_pixels, margin_pixels),  # Top-left
        (margin_pixels + w, margin_pixels),  # Top-right
        (margin_pixels, margin_pixels + h),  # Bottom-left
        (margin_pixels + w, margin_pixels + h)  # Bottom-right
    ]

    # Draw crop marks at each corner
    for x, y in corners:
        # Horizontal marks
        draw.line([(x - mark_offset - mark_len, y), (x - mark_offset, y)], fill='black', width=2)
        draw.line([(x + mark_offset, y), (x + mark_offset + mark_len, y)], fill='black', width=2)

        # Vertical marks
        draw.line([(x, y - mark_offset - mark_len), (x, y - mark_offset)], fill='black', width=2)
        draw.line([(x, y + mark_offset), (x, y + mark_offset + mark_len)], fill='black', width=2)

    return np.array(img)


def add_metadata_text(
    patch: np.ndarray,
    text: str,
    position: str = "bottom"
) -> np.ndarray:
    """Add metadata text to patch (e.g., size, date)."""
    img = Image.fromarray(patch)
    draw = ImageDraw.Draw(img)

    # Use default font
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
    except:
        font = ImageFont.load_default()

    # Get text size
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]

    # Position text
    h, w = patch.shape[:2]
    if position == "bottom":
        x = (w - text_width) // 2
        y = h - text_height - 10
    elif position == "top":
        x = (w - text_width) // 2
        y = 10
    else:
        x = y = 10

    # Draw text with background
    padding = 5
    draw.rectangle(
        [x - padding, y - padding, x + text_width + padding, y + text_height + padding],
        fill='white'
    )
    draw.text((x, y), text, fill='black', font=font)

    return np.array(img)


def save_print_ready(
    patch: np.ndarray,
    output_path: Path,
    dpi: int = 300,
    color_profile: str = "sRGB"
):
    """Save patch as print-ready image with embedded DPI metadata."""
    img = Image.fromarray(patch)

    # Set DPI metadata
    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(
        output_path,
        dpi=(dpi, dpi),
        quality=100,
        optimize=False
    )

    print(f"✓ Saved print-ready patch: {output_path}")
    print(f"  Resolution: {img.size[0]} x {img.size[1]} pixels")
    print(f"  DPI: {dpi}")
    print(f"  Color profile: {color_profile}")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare adversarial patches for physical printing"
    )

    # Input/Output
    parser.add_argument("--patch", type=Path, required=True,
                       help="Path to input patch image")
    parser.add_argument("--output", type=Path, required=True,
                       help="Path to output print-ready image")

    # Print specifications
    parser.add_argument("--size-inches", type=float, nargs=2, default=[4.0, 4.0],
                       metavar=("WIDTH", "HEIGHT"),
                       help="Physical size in inches (default: 4 4)")
    parser.add_argument("--dpi", type=int, default=300,
                       help="Print resolution in DPI (default: 300)")
    parser.add_argument("--upscale-method", type=str, default="lanczos",
                       choices=["bicubic", "lanczos", "nearest", "bilinear"],
                       help="Upscaling method (default: lanczos)")

    # Optional features
    parser.add_argument("--add-crop-marks", action="store_true",
                       help="Add crop marks for cutting")
    parser.add_argument("--crop-margin", type=int, default=50,
                       help="Margin for crop marks in pixels (default: 50)")
    parser.add_argument("--add-metadata", action="store_true",
                       help="Add size/date metadata text")
    parser.add_argument("--color-profile", type=str, default="sRGB",
                       choices=["sRGB", "AdobeRGB"],
                       help="Color profile (default: sRGB)")

    args = parser.parse_args()

    # Calculate target resolution
    width_inches, height_inches = args.size_inches
    target_width = int(width_inches * args.dpi)
    target_height = int(height_inches * args.dpi)

    print(f"Preparing patch for printing:")
    print(f"  Physical size: {width_inches}\" x {height_inches}\"")
    print(f"  Target resolution: {target_width} x {target_height} pixels @ {args.dpi} DPI")

    # Load patch
    print(f"\nLoading patch: {args.patch}")
    patch = load_patch(args.patch)
    print(f"  Input size: {patch.shape[1]} x {patch.shape[0]} pixels")

    # Upscale to print resolution
    print(f"\nUpscaling with {args.upscale_method} method...")
    patch_upscaled = upscale_patch(
        patch,
        target_size_pixels=(target_width, target_height),
        method=args.upscale_method
    )

    # Add crop marks if requested
    if args.add_crop_marks:
        print(f"Adding crop marks with {args.crop_margin}px margin...")
        patch_upscaled = add_crop_marks(patch_upscaled, margin_pixels=args.crop_margin)

    # Add metadata if requested
    if args.add_metadata:
        from datetime import datetime
        metadata = f"{width_inches}\"x{height_inches}\" @ {args.dpi}DPI - {datetime.now().strftime('%Y-%m-%d')}"
        print(f"Adding metadata: {metadata}")
        patch_upscaled = add_metadata_text(patch_upscaled, metadata, position="bottom")

    # Save print-ready image
    print(f"\nSaving print-ready image...")
    save_print_ready(
        patch_upscaled,
        args.output,
        dpi=args.dpi,
        color_profile=args.color_profile
    )

    # Print instructions
    print(f"\n{'='*60}")
    print("PRINTING INSTRUCTIONS")
    print(f"{'='*60}")
    print(f"1. Open {args.output} in image editor or print directly")
    print(f"2. Print at actual size ({width_inches}\" x {height_inches}\")")
    print(f"3. Ensure printer settings: {args.dpi} DPI, {args.color_profile} color")
    print(f"4. Use appropriate transfer paper or printing method")
    if args.add_crop_marks:
        print(f"5. Cut along crop marks for precise dimensions")
    print(f"\nRecommended: Fabric transfer paper for iron-on application")
    print(f"See README.md for detailed fabrication guide")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
