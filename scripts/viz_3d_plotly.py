"""Interactive 3D visualization of Spotiflow inference results using Plotly.

Runs inference on the built-in synthetic 3D sample image with the `synth_3d`
pretrained model and opens an interactive HTML visualization showing:
  - The predicted heatmap as a semi-transparent volume (isosurfaces)
  - Detected spots as a 3D scatter plot

Usage:
    python scripts/viz_3d_plotly.py [--out OUTPUT.html]
"""

import argparse
from pathlib import Path

import numpy as np
from scipy.ndimage import zoom


def main(out: Path) -> None:
    import plotly.graph_objects as go

    from spotiflow.model import Spotiflow
    from spotiflow.sample_data import test_image_synth_3d

    img = test_image_synth_3d()
    print(f"Image shape: {img.shape}, dtype: {img.dtype}")

    model = Spotiflow.from_pretrained("synth_3d")
    points, details = model.predict(img)
    print(f"Detected {len(points)} spots")

    # Upsample heatmap to image resolution
    scale = np.array(img.shape) / np.array(details.heatmap.shape)
    heatmap_up = zoom(details.heatmap, scale, order=1)
    Z, Y, X = np.mgrid[0 : img.shape[0], 0 : img.shape[1], 0 : img.shape[2]]

    fig = go.Figure()

    # Heatmap volume
    fig.add_trace(
        go.Volume(
            x=X.flatten(),
            y=Y.flatten(),
            z=Z.flatten(),
            value=heatmap_up.flatten(),
            isomin=0.15,
            isomax=float(heatmap_up.max()),
            opacity=0.08,
            surface_count=15,
            colorscale="Hot",
            showscale=False,
            name="Heatmap",
        )
    )

    # Spot scatter
    fig.add_trace(
        go.Scatter3d(
            x=points[:, 2],
            y=points[:, 1],
            z=points[:, 0],
            mode="markers",
            marker=dict(size=4, color="cyan", opacity=0.9, line=dict(width=0)),
            name="Spots",
        )
    )

    fig.update_layout(
        title=f"3D Spotiflow inference (Plotly) — {len(points)} spots detected",
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
            bgcolor="black",
            xaxis=dict(backgroundcolor="black", gridcolor="gray", showbackground=True),
            yaxis=dict(backgroundcolor="black", gridcolor="gray", showbackground=True),
            zaxis=dict(backgroundcolor="black", gridcolor="gray", showbackground=True),
        ),
        paper_bgcolor="black",
        font_color="white",
        margin=dict(l=0, r=0, t=40, b=0),
    )

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out))
    print(f"Saved interactive visualization to {out}")

    import webbrowser
    webbrowser.open(out.resolve().as_uri())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("spotiflow_3d_plotly.html"),
        help="Output HTML file path (default: spotiflow_3d_plotly.html)",
    )
    args = parser.parse_args()
    main(args.out)
