"""Look at a design in three dimensions, blocks and fasteners together.

Reads any CSV in the schema the repair uses and draws one row of it: each block as a
translucent box with its edges picked out, each screw as a rod driven into the block it is
stored against, and the table as a plane at the origin. Seeing the fasteners alongside the
geometry is the point -- a repaired joint can look sound while a screw sits over an edge.

The screw direction is not in the file, so it is inferred: a screw enters through the face of
its block that it lies closest to. Its length is not in the file either and is a setting in
the interface.

    python pipeline/visualize_design.py
    python pipeline/visualize_design.py --csv pipeline/foo.csv --port 8053

Serves a local page that lists the CSVs found next to this file and lets a row be chosen.
Nothing is written to disk.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import dash
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import dcc, html
from dash.dependencies import Input, Output

# Default screw length in metres, and how a screw is drawn. Blocks are given distinct colours
# by index and drawn translucent, so a block hidden behind another is still visible.
SCREW_LENGTH_M = 0.022
SCREW_LINE_WIDTH = 6
SCREW_COLOR = "#52525b"
BLOCK_COLORS = [
    "#3b82f6",
    "#10b981",
    "#f59e0b",
    "#ef4444",
    "#8b5cf6",
    "#ec4899",
    "#06b6d4",
    "#f97316",
    "#14b8a6",
]
BLOCK_OPACITY = 0.55
TABLE_OPACITY = 0.15
TABLE_COLOR = "#cbd5e1"


def _detect_n_blocks(df: pd.DataFrame) -> int:
    """Number of blocks the schema provides for, counted from the position columns."""
    n = 0
    while f"Block{n}_PosX" in df.columns:
        n += 1
    return n


def _cube_mesh(
    pos: np.ndarray,
    size: np.ndarray,
    color: str,
    name: str,
    opacity: float = BLOCK_OPACITY,
) -> go.Mesh3d:
    """A block as a solid box, given its centre and its full edge lengths."""
    cx, cy, cz = pos
    hx, hy, hz = size / 2.0
    verts = np.array(
        [
            [cx - hx, cy - hy, cz - hz],
            [cx + hx, cy - hy, cz - hz],
            [cx - hx, cy + hy, cz - hz],
            [cx + hx, cy + hy, cz - hz],
            [cx - hx, cy - hy, cz + hz],
            [cx + hx, cy - hy, cz + hz],
            [cx - hx, cy + hy, cz + hz],
            [cx + hx, cy + hy, cz + hz],
        ]
    )
    faces = np.array(
        [
            [0, 1, 2],
            [1, 3, 2],  # -Z
            [4, 6, 5],
            [5, 6, 7],  # +Z
            [0, 2, 4],
            [2, 6, 4],  # -X
            [1, 5, 3],
            [3, 5, 7],  # +X
            [0, 4, 1],
            [1, 4, 5],  # -Y
            [2, 3, 6],
            [3, 7, 6],  # +Y
        ]
    )
    return go.Mesh3d(
        x=verts[:, 0],
        y=verts[:, 1],
        z=verts[:, 2],
        i=faces[:, 0],
        j=faces[:, 1],
        k=faces[:, 2],
        color=color,
        opacity=opacity,
        flatshading=True,
        name=name,
        showlegend=True,
        hoverinfo="name",
    )


def _cube_edges(pos: np.ndarray, size: np.ndarray, color: str) -> go.Scatter3d:
    """The twelve edges of a block, drawn so the outline stays readable through the fill."""
    cx, cy, cz = pos
    hx, hy, hz = size / 2.0
    c = np.array(
        [
            [cx - hx, cy - hy, cz - hz],
            [cx + hx, cy - hy, cz - hz],
            [cx + hx, cy + hy, cz - hz],
            [cx - hx, cy + hy, cz - hz],
            [cx - hx, cy - hy, cz + hz],
            [cx + hx, cy - hy, cz + hz],
            [cx + hx, cy + hy, cz + hz],
            [cx - hx, cy + hy, cz + hz],
        ]
    )
    edges = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 4),
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
    ]
    xs, ys, zs = [], [], []
    for a, b in edges:
        xs.extend([c[a, 0], c[b, 0], None])
        ys.extend([c[a, 1], c[b, 1], None])
        zs.extend([c[a, 2], c[b, 2], None])
    return go.Scatter3d(
        x=xs,
        y=ys,
        z=zs,
        mode="lines",
        line=dict(color=color, width=2),
        showlegend=False,
        hoverinfo="skip",
    )


def _nearest_face_normal(
    point: np.ndarray, block_pos: np.ndarray, block_size: np.ndarray
) -> np.ndarray:
    """Outward normal of the block face a point lies closest to.

    The input gives a screw's entry point but not its direction, so the direction is taken to
    be into the block through the nearest face.
    """
    half = block_size / 2.0
    block_min = block_pos - half
    block_max = block_pos + half
    # Ordered as minus and plus on each axis in turn, matching `normals` below.
    dists = np.array(
        [
            abs(point[0] - block_min[0]),
            abs(point[0] - block_max[0]),
            abs(point[1] - block_min[1]),
            abs(point[1] - block_max[1]),
            abs(point[2] - block_min[2]),
            abs(point[2] - block_max[2]),
        ]
    )
    normals = np.array(
        [
            [-1, 0, 0],
            [1, 0, 0],
            [0, -1, 0],
            [0, 1, 0],
            [0, 0, -1],
            [0, 0, 1],
        ]
    )
    return normals[int(np.argmin(dists))]


def _screw_rod_trace(
    start_xyz: np.ndarray, inward_dir: np.ndarray, length: float, name: str
) -> go.Scatter3d:
    """The shaft of a screw, from its entry point inwards, with the length in metres."""
    end = start_xyz + inward_dir * length
    return go.Scatter3d(
        x=[start_xyz[0], end[0]],
        y=[start_xyz[1], end[1]],
        z=[start_xyz[2], end[2]],
        mode="lines",
        line=dict(color=SCREW_COLOR, width=SCREW_LINE_WIDTH),
        name=name,
        hoverinfo="name",
        showlegend=False,
    )


def _screw_head_marker(xyz: np.ndarray) -> go.Scatter3d:
    """The head of a screw, at its entry point."""
    return go.Scatter3d(
        x=[xyz[0]],
        y=[xyz[1]],
        z=[xyz[2]],
        mode="markers",
        marker=dict(color=SCREW_COLOR, size=4, symbol="circle"),
        showlegend=False,
        hoverinfo="skip",
    )


def _table_plane(half_extent_m: float) -> go.Mesh3d:
    """The table, as a plane at the origin. A block below it is a design that cannot be built."""
    h = half_extent_m
    verts = np.array(
        [
            [-h, -h, 0.0],
            [h, -h, 0.0],
            [h, h, 0.0],
            [-h, h, 0.0],
        ]
    )
    faces = np.array([[0, 1, 2], [0, 2, 3]])
    return go.Mesh3d(
        x=verts[:, 0],
        y=verts[:, 1],
        z=verts[:, 2],
        i=faces[:, 0],
        j=faces[:, 1],
        k=faces[:, 2],
        color=TABLE_COLOR,
        opacity=TABLE_OPACITY,
        name="table",
        showlegend=False,
        hoverinfo="skip",
    )


def build_figure(
    df: pd.DataFrame, row_idx: int, screw_len_m: float, title_prefix: str = ""
) -> tuple[go.Figure, dict]:
    """Draw one design row.

    Args:
        df: The table.
        row_idx: Which row to draw.
        screw_len_m: Length a screw is drawn with, in metres.
        title_prefix: Text placed before the generated title.

    Returns:
        The figure, and a summary of what went into it: how many blocks and screws were drawn,
        how many rows the table has, and the geometry of each block.
    """
    row = df.iloc[row_idx]
    n_blocks = _detect_n_blocks(df)
    fig = go.Figure()

    blocks_info: list[dict] = []
    for i in range(n_blocks):
        pos = np.array(
            [row[f"Block{i}_PosX"], row[f"Block{i}_PosY"], row[f"Block{i}_PosZ"]],
            dtype=float,
        )
        size = np.array(
            [row[f"Block{i}_SizeX"], row[f"Block{i}_SizeY"], row[f"Block{i}_SizeZ"]],
            dtype=float,
        )
        if np.isnan(pos).any() or np.isnan(size).any():
            continue
        color = BLOCK_COLORS[i % len(BLOCK_COLORS)]
        fig.add_trace(_cube_mesh(pos, size, color, name=f"Block{i}"))
        fig.add_trace(_cube_edges(pos, size, color))
        blocks_info.append({"idx": i, "pos": pos, "size": size})

    n_screws = 0
    for i in range(n_blocks):
        owner = next((b for b in blocks_info if b["idx"] == i), None)
        if owner is None:
            continue
        for s in range(5):
            cols = [
                f"Block{i}_Screw{s}_X",
                f"Block{i}_Screw{s}_Y",
                f"Block{i}_Screw{s}_Z",
            ]
            if not all(c in df.columns for c in cols):
                continue
            sx, sy, sz = (float(row[cols[0]]), float(row[cols[1]]), float(row[cols[2]]))
            if np.isnan(sx) or np.isnan(sy) or np.isnan(sz):
                continue
            screw_xyz = np.array([sx, sy, sz])
            outward = _nearest_face_normal(screw_xyz, owner["pos"], owner["size"])
            inward = -outward.astype(float)
            fig.add_trace(
                _screw_rod_trace(
                    screw_xyz,
                    inward,
                    screw_len_m,
                    name=f"Block{i}_Screw{s}",
                )
            )
            fig.add_trace(_screw_head_marker(screw_xyz))
            n_screws += 1

    # The table is sized to the design rather than fixed, so a small assembly is not lost on
    # a large plane.
    if blocks_info:
        all_max = np.max([b["pos"] + b["size"] / 2 for b in blocks_info], axis=0)
        all_min = np.min([b["pos"] - b["size"] / 2 for b in blocks_info], axis=0)
        span = max(np.max(np.abs(all_min[:2])), np.max(all_max[:2])) * 1.4
        span = max(float(span), 0.1)
    else:
        span = 0.2
    fig.add_trace(_table_plane(span))

    fig.update_layout(
        title=f"{title_prefix}({n_blocks} blocks, {n_screws} screws @ "
        f"{screw_len_m*1000:.0f}mm, row {row_idx})",
        scene=dict(
            xaxis_title="X (m)",
            yaxis_title="Y (m)",
            zaxis_title="Z (m)",
            aspectmode="data",
            bgcolor="#f8fafc",
        ),
        margin=dict(l=0, r=0, t=40, b=0),
        legend=dict(itemsizing="constant"),
    )

    return fig, {
        "n_blocks": n_blocks,
        "n_screws": n_screws,
        "n_rows": len(df),
        "blocks_info": blocks_info,
    }


def discover_csvs(pipeline_dir: Path) -> list[Path]:
    """Which files to offer.

    Every CSV in the given directory, unless the list below is filled in, in which case only
    those are shown. It is a hook for pinning a session to a few files of interest.
    """
    manually_added = [
    ]

    if len(manually_added) > 0:
        return [Path(p) for p in manually_added]
    else:
        return sorted(pipeline_dir.glob("*.csv"))


def build_app(csv_files: list[Path]) -> dash.Dash:
    """Assemble the page: a file and row selector, the screw length, and the figure."""
    app = dash.Dash(__name__)
    app.title = "Design Visualiser"

    # Read once at startup rather than per interaction; the files hold a handful of rows.
    loaded = {str(p): pd.read_csv(p) for p in csv_files}
    first_path = str(csv_files[0]) if csv_files else None
    first_df = loaded[first_path] if first_path else None

    csv_options = [{"label": Path(p).name, "value": p} for p in loaded.keys()]
    row_options = (
        [{"label": f"row {i}", "value": i} for i in range(len(first_df))]
        if first_df is not None
        else []
    )

    app.layout = html.Div(
        style={
            "fontFamily": "sans-serif",
            "padding": "16px",
            "maxWidth": "1600px",
            "margin": "0 auto",
        },
        children=[
            html.H2("Pipeline Design Visualiser", style={"marginBottom": "4px"}),
            html.Div(
                f"Auto-discovered {len(csv_files)} CSV(s) in pipeline/",
                style={"color": "#64748b", "marginBottom": "16px"},
            ),
            html.Div(
                style={
                    "display": "flex",
                    "gap": "16px",
                    "marginBottom": "12px",
                    "flexWrap": "wrap",
                },
                children=[
                    html.Div(
                        [
                            html.Label("CSV file:", style={"fontWeight": 600}),
                            dcc.Dropdown(
                                id="csv-dd",
                                options=csv_options,
                                value=first_path,
                                clearable=False,
                                style={"width": "360px"},
                            ),
                        ]
                    ),
                    html.Div(
                        [
                            html.Label("Row:", style={"fontWeight": 600}),
                            dcc.Dropdown(
                                id="row-dd",
                                options=row_options,
                                value=0,
                                clearable=False,
                                style={"width": "120px"},
                            ),
                        ]
                    ),
                    html.Div(
                        [
                            html.Label("Screw length (mm):", style={"fontWeight": 600}),
                            dcc.Input(
                                id="screw-len",
                                type="number",
                                value=22.0,
                                min=1,
                                max=200,
                                step=1,
                                style={"width": "100px"},
                            ),
                        ]
                    ),
                ],
            ),
            dcc.Graph(id="design-graph", style={"height": "780px", "width": "100%"}),
            html.Div(
                id="status",
                style={
                    "marginTop": "8px",
                    "color": "#64748b",
                    "fontFamily": "monospace",
                },
            ),
        ],
    )

    @app.callback(
        Output("row-dd", "options"),
        Output("row-dd", "value"),
        Input("csv-dd", "value"),
    )
    def update_row_options(csv_path):
        """Refill the row selector when another file is chosen."""
        df = loaded[csv_path]
        opts = [{"label": f"row {i}", "value": i} for i in range(len(df))]
        return opts, 0

    @app.callback(
        Output("design-graph", "figure"),
        Output("status", "children"),
        Input("csv-dd", "value"),
        Input("row-dd", "value"),
        Input("screw-len", "value"),
    )
    def update_graph(csv_path, row_idx, screw_len_mm):
        """Redraw the figure and the status line on any change."""
        df = loaded[csv_path]
        screw_len_m = (screw_len_mm or 22.0) / 1000.0
        fig, info = build_figure(
            df,
            int(row_idx or 0),
            screw_len_m,
            title_prefix=f"{Path(csv_path).name}  ",
        )
        status = (
            f"file: {Path(csv_path).name}  |  row {row_idx}/{info['n_rows']}  |  "
            f"{info['n_blocks']} blocks  |  {info['n_screws']} screws  |  "
            f"screw_len = {screw_len_m*1000:.0f} mm"
        )
        return fig, status

    return app


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        help="Specific CSV to show (default: auto-discover all " "*.csv in pipeline/)",
    )
    parser.add_argument(
        "--port", type=int, default=8053, help="Dash port (default 8053)"
    )
    args = parser.parse_args()

    pipeline_dir = Path(__file__).resolve().parent
    if args.csv:
        csvs = [Path(args.csv)]
    else:
        csvs = discover_csvs(pipeline_dir)
    csvs = [p for p in csvs if p.exists()]

    if not csvs:
        print(
            f"ERROR: no CSV files found in {pipeline_dir} "
            f"(or --csv path doesn't exist)"
        )
        return 1

    print(f"Loaded {len(csvs)} CSV file(s):")
    for p in csvs:
        print(f"  - {p}")

    app = build_app(csvs)
    print(f"\nStarting Dash dashboard at http://127.0.0.1:{args.port}")
    app.run(debug=False, port=args.port)
    return 0


if __name__ == "__main__":
    sys.exit(main())
