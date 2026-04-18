"""Static 3D visualization of Spotiflow inference results using Matplotlib.

Runs inference on the built-in synthetic 3D sample image with the `synth_3d`
pretrained model and saves a figure with three orthogonal max-intensity
projections (XY, XZ, YZ) with detected spots overlaid.

Usage:
    python scripts/viz_3d_matplotlib.py [--out OUTPUT.png]
"""

import argparse
from pathlib import Path


def main(out: Path) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    from spotiflow.model import Spotiflow
    from spotiflow.sample_data import test_image_synth_3d

    img = test_image_synth_3d()
    print(f"Image shape: {img.shape}, dtype: {img.dtype}")

    model = Spotiflow.from_pretrained("synth_3d")
    points, details = model.predict(img)
    print(f"Detected {len(points)} spots")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].set_title("XY (max Z projection)")
    axes[0].imshow(img.max(axis=0), cmap="gray")
    axes[0].scatter(points[:, 2], points[:, 1], s=20, c="red", linewidths=0.5, alpha=0.8)
    axes[0].set_xlabel("X")
    axes[0].set_ylabel("Y")

    axes[1].set_title("XZ (max Y projection)")
    axes[1].imshow(img.max(axis=1), cmap="gray")
    axes[1].scatter(points[:, 2], points[:, 0], s=20, c="red", linewidths=0.5, alpha=0.8)
    axes[1].set_xlabel("X")
    axes[1].set_ylabel("Z")

    axes[2].set_title("YZ (max X projection)")
    axes[2].imshow(img.max(axis=2), cmap="gray")
    axes[2].scatter(points[:, 1], points[:, 0], s=20, c="red", linewidths=0.5, alpha=0.8)
    axes[2].set_xlabel("Y")
    axes[2].set_ylabel("Z")

    plt.suptitle(
        f"3D Spotiflow inference (Matplotlib) — {len(points)} spots detected",
        fontsize=13,
    )
    plt.tight_layout()

    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out), dpi=150, bbox_inches="tight")
    print(f"Saved figure to {out}")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("spotiflow_3d_matplotlib.png"),
        help="Output PNG file path (default: spotiflow_3d_matplotlib.png)",
    )
    args = parser.parse_args()
    main(args.out)
