"""Browser viewer for a single design in the CSV schema of the generative stage.

Draws every block of one row as an axis-aligned box and every screw that is filled in as a rod
entering the block whose columns hold it, so that a design can be looked at before or after a
repair without going through the simulator. Positions and edge lengths are read in metres and
plotted in metres. This is a viewer only: it computes no metric and passes no judgement on
whether a design is buildable.

The screw length is a display choice, not data. The schema records only where a screw goes in,
so the rod is drawn at a fixed length, adjustable in the page, and pointing along the inward
normal of the face nearest the entry point.

Run as ``python pipeline/visualize_design.py``, which serves a page on port 8053 and offers
every CSV beside this file. ``--csv <path>`` restricts it to one file and ``--port`` moves the
server. Nothing is written to disk.
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

# The drawn length of a screw shaft, in metres. The schema carries an entry point but no
# length, so this is a drawing convention and carries no claim about the real fastener.
SCREW_LENGTH_M = 0.022
SCREW_LINE_WIDTH = 6
SCREW_COLOR = "#52525b"
# Indexed by block number, so a block keeps its colour across rows and across files and two
# views of the same design can be compared by eye.
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
# Blocks are drawn semi-transparent so that a joint hidden inside the assembly, and the screws
# that pass through it, stay visible from outside.
BLOCK_OPACITY = 0.55
TABLE_OPACITY = 0.15
TABLE_COLOR = "#cbd5e1"


def _detect_n_blocks(df: pd.DataFrame) -> int:
    """Count the block slots the schema provides, stopping at the first missing index."""
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
    """A block as a closed surface of twelve triangles, from its centre and full edge lengths."""
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
    """The outline of a block, drawn on top of the transparent surface so its extent stays
    readable where several blocks overlap."""
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
    # All twelve edges go into one trace, separated by a gap, so a whole block costs a single
    # trace rather than twelve.
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
    """The outward normal of the block face nearest a point, as a ``(3,)`` unit vector.

    The schema says where a screw enters but not which way it points, so the direction is
    recovered from the geometry: the shaft is drawn opposite this, into the block.
    """
    half = block_size / 2.0
    block_min = block_pos - half
    block_max = block_pos + half
    # The six faces in the order -X, +X, -Y, +Y, -Z, +Z, matching the normals below.
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
    """The shaft of one screw, from its entry point along ``inward_dir`` for ``length`` metres."""
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
    """The head of one screw, marking the entry point the schema records."""
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
    """The table surface as a square of half-width ``half_extent_m`` metres at height zero, the
    reference every vertical coordinate in the schema is measured from."""
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
        df: The whole file; the row is selected by position, not by index label.
        row_idx: Position of the design to draw.
        screw_len_m: Drawn shaft length in metres.
        title_prefix: Text placed before the generated title.

    Returns:
        A pair ``(fig, info)``. ``info`` reports how many block slots the schema has, how many
        screws were drawn, how many rows the file holds, and the geometry of the blocks that
        were actually present, each as ``{"idx", "pos", "size"}`` in metres.
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
        # An empty slot is skipped rather than drawn at the origin, so a design that uses fewer
        # blocks than the schema allows does not gain a phantom block.
        if np.isnan(pos).any() or np.isnan(size).any():
            continue
        color = BLOCK_COLORS[i % len(BLOCK_COLORS)]
        fig.add_trace(_cube_mesh(pos, size, color, name=f"Block{i}"))
        fig.add_trace(_cube_edges(pos, size, color))
        blocks_info.append({"idx": i, "pos": pos, "size": size})

    n_screws = 0
    for i in range(n_blocks):
        # The direction is taken from the block whose columns store the screw, which need not
        # be the block the screw is driven into.
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

    # The table is sized to reach past the design so it reads as a ground plane rather than a
    # tile under it, with a floor for the case where every block sits near the origin.
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
            # The three axes keep a common scale, so a thin block looks thin. Without this a
            # slab and a cube can be drawn identically.
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
    """List the CSVs to offer in the page, sorted by name.

    Filling in ``manually_added`` restricts the page to those files; leaving it empty offers
    every CSV in the folder.
    """
    manually_added = [
        # Paths relative to the repository root, for example "pipeline/my_design.csv".
    ]

    if len(manually_added) > 0:
        return [Path(p) for p in manually_added]
    else:
        return sorted(pipeline_dir.glob("*.csv"))


def build_app(csv_files: list[Path]) -> dash.Dash:
    """Assemble the page: a file and row selector, the screw length, and the figure."""
    app = dash.Dash(__name__)
    app.title = "Design Visualiser"

    # Every file is read once at start-up and held in memory, so switching between two designs
    # redraws without touching the disk.
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
        """Re-fill the row selector when the file changes, and go back to the first row."""
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
        """Redraw the figure and the status line for the current file, row and screw length."""
        df = loaded[csv_path]
        # The page states the screw length in millimetres; everything below it is in metres.
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
