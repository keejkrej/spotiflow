"""
Spotiflow inference script for lnp.tif
Detects spots and visualizes results including flow and subpix fields
"""

import numpy as np
import matplotlib.pyplot as plt
import tifffile
from pathlib import Path

from spotiflow.model import Spotiflow


def main():
    # Load image
    img_path = Path(__file__).parent / "lnp.tif"
    img = tifffile.imread(img_path)
    print(f"Loaded image: {img_path}")
    print(f"Original shape: {img.shape}, dtype: {img.dtype}")

    # Squeeze extra dimensions (TZCYX -> YX)
    img = np.squeeze(img)
    print(f"Squeezed shape: {img.shape}")

    # Load pretrained model
    print("Loading pretrained 'general' model...")
    model = Spotiflow.from_pretrained("general")

    # Run inference with lower probability threshold to detect more spots
    prob_thresh = 0.1  # Lower threshold (default is 0.5)
    print(f"Running inference with prob_thresh={prob_thresh}...")
    points, details = model.predict(img, prob_thresh=prob_thresh)
    print(f"Detected {len(points)} spots")

    # Print available outputs
    print(f"\nAvailable outputs in details:")
    print(f"  - heatmap: {details.heatmap.shape if details.heatmap is not None else None}")
    print(f"  - flow: {details.flow.shape if details.flow is not None else None}")
    print(f"  - subpix: {details.subpix.shape if details.subpix is not None else None}")
    print(f"  - prob: {details.prob.shape if hasattr(details, 'prob') and details.prob is not None else None}")
    print(f"  - intens: {details.intens.shape if hasattr(details, 'intens') and details.intens is not None else None}")

    # Create visualization - 2x2 grid
    fig, axes = plt.subplots(2, 2, figsize=(8, 8))

    # Row 1: Original image and detected spots
    ax = axes[0, 0]
    ax.imshow(img, cmap="gray")
    ax.set_title("Original Image")
    ax.axis("off")

    ax = axes[0, 1]
    ax.imshow(img, cmap="gray")
    if len(points) > 0:
        ax.scatter(points[:, 1], points[:, 0], c="red", s=15, marker="o", alpha=0.7, edgecolors="white", linewidths=0.5)
    ax.set_title(f"Detected Spots (n={len(points)}, thresh={prob_thresh})")
    ax.axis("off")

    # Row 2: Heatmap and heatmap with spots
    ax = axes[1, 0]
    if details.heatmap is not None:
        im = ax.imshow(details.heatmap, cmap="hot")
        plt.colorbar(im, ax=ax, fraction=0.046)
        ax.set_title("Probability Heatmap")
    ax.axis("off")

    ax = axes[1, 1]
    if details.heatmap is not None:
        ax.imshow(details.heatmap, cmap="hot")
        if len(points) > 0:
            ax.scatter(points[:, 1], points[:, 0], c="cyan", s=15, marker="o", alpha=0.8, edgecolors="white", linewidths=0.5)
        ax.set_title("Heatmap with Spots Overlay")
    ax.axis("off")

    plt.tight_layout()

    # Save figure
    output_path = Path(__file__).parent / "inference_results.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved visualization to: {output_path}")

    # Save detected points to CSV
    csv_path = Path(__file__).parent / "detected_spots.csv"
    np.savetxt(csv_path, points, delimiter=",", header="y,x", comments="")
    print(f"Saved spot coordinates to: {csv_path}")

    plt.show()


if __name__ == "__main__":
    main()
