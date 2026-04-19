"""Interactive 3D visualization of Spotiflow inference results using Plotly.

Runs inference on the built-in synthetic 3D sample image with the `synth_3d`
pretrained model and opens interactive HTML visualizations showing:
  - The predicted heatmap as a semi-transparent volume (isosurfaces)
  - Detected spots as a 3D scatter plot
  - The original input volume in a separate companion HTML file

Usage:
    python scripts/viz_3d_plotly.py [--out OUTPUT.html] [--input-out INPUT.html]
"""

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.ndimage import zoom

INPUT_VOLUME_COLORSCALE = (
    (0.0, "rgb(0,0,0)"),
    (0.2, "rgb(24,24,24)"),
    (0.5, "rgb(96,96,96)"),
    (0.8, "rgb(192,192,192)"),
    (1.0, "rgb(255,255,255)"),
)


def _companion_output_path(out: Path, suffix: str) -> Path:
    return out.with_name(f"{out.stem}{suffix}{out.suffix}")


def _normalize_volume(img: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(img, (1.0, 99.8))
    if hi <= lo:
        return np.zeros_like(img, dtype=np.float32)
    return np.clip((img.astype(np.float32) - lo) / (hi - lo), 0, 1)


def _downsample_volume_for_render(
    volume: np.ndarray, max_voxels: int = 120_000
) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Reduce volume size for responsive Plotly rendering while preserving scale."""
    volume = volume.astype(np.float32, copy=False)
    if volume.size <= max_voxels:
        z = np.arange(volume.shape[0], dtype=np.float32)
        y = np.arange(volume.shape[1], dtype=np.float32)
        x = np.arange(volume.shape[2], dtype=np.float32)
        return volume, (z, y, x)

    scale = (max_voxels / float(volume.size)) ** (1.0 / volume.ndim)
    new_shape = tuple(max(8, min(s, int(np.floor(s * scale)))) for s in volume.shape)

    # Guard against rounding keeping the result above the target size.
    while np.prod(new_shape) > max_voxels:
        axis = int(np.argmax(new_shape))
        new_shape = tuple(
            max(8, dim - 1) if i == axis else dim for i, dim in enumerate(new_shape)
        )

    factors = tuple(n / s for n, s in zip(new_shape, volume.shape))
    volume_ds = zoom(volume, factors, order=1)
    z = np.linspace(0, volume.shape[0] - 1, volume_ds.shape[0], dtype=np.float32)
    y = np.linspace(0, volume.shape[1] - 1, volume_ds.shape[1], dtype=np.float32)
    x = np.linspace(0, volume.shape[2] - 1, volume_ds.shape[2], dtype=np.float32)
    return volume_ds, (z, y, x)


def _build_input_volume_figure(img: np.ndarray):
    return _build_volume_figure(
        img,
        "3D Spotiflow input volume (Plotly)",
        colorscale=INPUT_VOLUME_COLORSCALE,
        opacity=0.06,
        isomin=0.12,
        surface_count=18,
    )


def _build_volume_figure(
    volume: np.ndarray,
    title: str,
    colorscale: str,
    opacity: float,
    isomin: float,
    surface_count: int,
):
    import plotly.graph_objects as go

    volume, axes = _downsample_volume_for_render(volume)
    z, y, x = np.meshgrid(*axes, indexing="ij")
    isomax = max(float(volume.max()), isomin)
    fig = go.Figure()
    fig.add_trace(
        go.Volume(
            x=x.flatten(),
            y=y.flatten(),
            z=z.flatten(),
            value=volume.flatten(),
            isomin=isomin,
            isomax=isomax,
            opacity=opacity,
            surface_count=surface_count,
            colorscale=colorscale,
            showscale=False,
            name=title,
        )
    )
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
            aspectmode="data",
            bgcolor="black",
            xaxis=dict(backgroundcolor="black", gridcolor="gray", showbackground=True),
            yaxis=dict(backgroundcolor="black", gridcolor="gray", showbackground=True),
            zaxis=dict(backgroundcolor="black", gridcolor="gray", showbackground=True),
        ),
        paper_bgcolor="black",
        font_color="white",
        margin=dict(l=0, r=0, t=40, b=0),
    )
    return fig


def _write_figure_html(fig, path: Path) -> None:
    import plotly.io as pio

    figure_html = pio.to_html(
        fig,
        include_plotlyjs="directory",
        full_html=False,
        default_width="100vw",
        default_height="100vh",
        config={"displaylogo": False, "responsive": True},
    )
    html = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8" />
  <style>
    html, body {{
      margin: 0;
      width: 100%;
      height: 100%;
      background: black;
      color: white;
      overflow: hidden;
    }}
    body > div {{
      width: 100vw;
      height: 100vh;
    }}
  </style>
</head>
<body>
{figure_html}
</body>
</html>
"""
    path.write_text(html, encoding="utf-8")


def main(out: Path, input_out: Optional[Path] = None) -> None:
    import plotly.graph_objects as go

    from spotiflow.model import Spotiflow
    from spotiflow.sample_data import test_image_synth_3d

    img = test_image_synth_3d()
    print(f"Image shape: {img.shape}, dtype: {img.dtype}")
    if input_out is None:
        input_out = _companion_output_path(out, "_input")

    model = Spotiflow.from_pretrained("synth_3d")
    points, details = model.predict(img)
    print(f"Detected {len(points)} spots")

    # Upsample heatmap to image resolution
    scale = np.array(img.shape) / np.array(details.heatmap.shape)
    heatmap_up = zoom(details.heatmap, scale, order=1)
    fig = _build_volume_figure(
        heatmap_up,
        f"3D Spotiflow inference (Plotly) — {len(points)} spots detected",
        colorscale="Hot",
        opacity=0.08,
        isomin=0.15,
        surface_count=15,
    )

    # Spot scatter
    fig.add_trace(
        go.Scatter3d(
            x=points[:, 2],
            y=points[:, 1],
            z=points[:, 0],
            mode="markers",
            marker=dict(
                size=4,
                color="deepskyblue",
                opacity=0.9,
                line=dict(width=0),
            ),
            name="Spots",
        )
    )

    input_fig = _build_input_volume_figure(_normalize_volume(img))

    out.parent.mkdir(parents=True, exist_ok=True)
    input_out.parent.mkdir(parents=True, exist_ok=True)
    _write_figure_html(fig, out)
    _write_figure_html(input_fig, input_out)
    print(f"Saved interactive visualization to {out}")
    print(f"Saved original input visualization to {input_out}")

    import webbrowser

    webbrowser.open(out.resolve().as_uri())
    webbrowser.open(input_out.resolve().as_uri())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("spotiflow_3d_plotly.html"),
        help="Output HTML file path (default: spotiflow_3d_plotly.html)",
    )
    parser.add_argument(
        "--input-out",
        type=Path,
        default=None,
        help=(
            "Output HTML file path for the original input visualization "
            "(default: <out stem>_input.html)"
        ),
    )
    args = parser.parse_args()
    main(args.out, input_out=args.input_out)
