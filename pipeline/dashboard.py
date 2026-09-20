"""Browser view of one design in three states, with the numbers behind each of them.

Takes one row of a design CSV and shows it as it arrived, after the two closed-form corrections
of :mod:`pipeline.repair_strategies`, and after the repair. Each state gets a three-dimensional
plot and a table of every connected pair, and each table carries the surrogate's prediction and
the closed-form measurement of the same geometry side by side, in millimetres.

Putting the two next to each other is the point of the page. Where the surrogate accepts a pair
the formulas reject, the row is flagged as a bluff; where it refuses one they accept, as
over-cautious. Reading a repair off the plot alone would show neither.

Loads the files named in ``CSV_FILES`` from ``pipeline/new_csv/`` and serves a page on port
8053; ``--port`` moves it. As a side effect, every row it computes is written back as
``<stem>_repaired.csv`` beside its input.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import dash
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import torch
from dash import dash_table, dcc, html
from dash.dependencies import Input, Output

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipeline.repair_strategies import (  # noqa: E402
    apply_snaps,
    detect_n_blocks,
    gnn_repair_layered,
    parse_blocks,
    parse_connections,
    parse_original_screws,
    resolve_penetrations,
    update_screws_for_repair,
)
from src.analytical_metrics import (  # noqa: E402
    analytical_overlap,
    analytical_thickness,
    infer_face,
)
from src.repair_process import (  # noqa: E402
    THRESH_OVERLAP,
    THRESH_THICKNESS,
    load_models,
    predict_feasibility,
)

# The designs offered in the page, resolved against pipeline/new_csv/. A name that is not
# present on disk is reported and skipped rather than raising, because that folder is not kept
# in the repository.
CSV_FILES: list[str] = [
    "castle_candidate_4_ml_input.csv",
    "cactus_candidate_4_ml_input.csv",
    "car_candidate_2_ml_input.csv",
    "dog_candidate_1_retry1_ml_input.csv",
    "giraffe_candidate_4_retry1_ml_input.csv",
    "stonehenge_candidate_4_retry1_buildable_ml_input.csv",
    "table_candidate_2_retry101_ml_input.csv",
    "table_candidate_1_retry1_ml_input.csv",
    "letter_t_candidate_6_buildable_ml_input.csv",
    # The two below have a block joined to more than one parent, which the pairwise
    # decomposition cannot express; they are kept in the list as the harder comparison.
    "arch_candidate_3_ml_input.csv",
    "letter_t_candidate_2_ml_input.csv",
]

SCREW_LENGTH_M = 0.022
SCREW_LINE_WIDTH = 6
SCREW_COLOR = "#52525b"
# Indexed by block number, so a block keeps its colour across the three columns and the same
# part can be followed from the input to the repaired state by eye.
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
    "#a855f7",
]
# Blocks are drawn semi-transparent so that a joint buried inside the assembly, which is
# exactly where the interesting failures are, stays visible from outside.
BLOCK_OPACITY = 0.55
TABLE_OPACITY = 0.15
TABLE_COLOR = "#cbd5e1"


def _cube_mesh(pos: np.ndarray, size: np.ndarray, color: str, name: str) -> go.Mesh3d:
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
            [1, 3, 2],
            [4, 6, 5],
            [5, 6, 7],
            [0, 2, 4],
            [2, 6, 4],
            [1, 5, 3],
            [3, 5, 7],
            [0, 4, 1],
            [1, 4, 5],
            [2, 3, 6],
            [3, 7, 6],
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
        opacity=BLOCK_OPACITY,
        flatshading=True,
        name=name,
        showlegend=True,
        hoverinfo="name",
    )


def _cube_edges(pos: np.ndarray, size: np.ndarray, color: str) -> go.Scatter3d:
    """The outline of a block, drawn over the transparent surface so its extent stays readable
    where several blocks overlap."""
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


def _table_plane(half_extent_m: float) -> go.Mesh3d:
    """The table surface as a square of half-width ``half_extent_m`` metres at height zero, the
    reference every vertical coordinate is measured from."""
    h = half_extent_m
    verts = np.array([[-h, -h, 0.0], [h, -h, 0.0], [h, h, 0.0], [-h, h, 0.0]])
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


def _build_3d_figure(
    blocks_dict: dict[int, dict],
    title: str,
    screws: dict[int, list[tuple[int, np.ndarray]]] | None = None,
) -> go.Figure:
    """Draw one of the three states of a design.

    Args:
        blocks_dict: Blocks in metres, as any stage of the pipeline returns them.
        title: Heading placed on the figure.
        screws: Optional ``{block: [(slot, (3,) position in metres), ...]}``, drawn as markers
            at the entry points. The shaft is not drawn here; only where a screw goes matters
            for judging a joint.

    Returns:
        The figure. The three states are drawn on the same axes convention, so the columns can
        be compared directly.
    """
    fig = go.Figure()
    blocks_info = []
    for i in sorted(blocks_dict.keys()):
        b = blocks_dict[i]
        pos = np.asarray(b["pos"], dtype=float)
        size = np.asarray(b["size"], dtype=float)
        if np.isnan(pos).any() or np.isnan(size).any():
            continue
        color = BLOCK_COLORS[i % len(BLOCK_COLORS)]
        fig.add_trace(_cube_mesh(pos, size, color, name=f"Block{i}"))
        fig.add_trace(_cube_edges(pos, size, color))
        blocks_info.append({"pos": pos, "size": size})

    if screws:
        xs, ys, zs, txts = [], [], [], []
        for bi, lst in screws.items():
            for slot, xyz in lst:
                if np.isnan(xyz).any():
                    continue
                xs.append(float(xyz[0]))
                ys.append(float(xyz[1]))
                zs.append(float(xyz[2]))
                txts.append(f"B{bi} screw{slot}")
        if xs:
            fig.add_trace(
                go.Scatter3d(
                    x=xs,
                    y=ys,
                    z=zs,
                    mode="markers",
                    marker=dict(
                        size=5,
                        color="#dc2626",
                        symbol="diamond",
                        line=dict(color="#7f1d1d", width=1),
                    ),
                    text=txts,
                    hoverinfo="text+x+y+z",
                    name="screws",
                    showlegend=True,
                )
            )

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
        title=title,
        scene=dict(
            xaxis_title="X (m)",
            yaxis_title="Y (m)",
            zaxis_title="Z (m)",
            # The three axes keep a common scale, so a block thinned by the repair looks
            # thinner rather than being stretched back to fill the frame.
            aspectmode="data",
            bgcolor="#f8fafc",
        ),
        margin=dict(l=0, r=0, t=40, b=0),
        legend=dict(itemsizing="constant", x=0, y=1),
    )
    return fig


def _compute_pair_metrics(
    blocks_dict: dict[int, dict],
    conns: list[tuple[int, int]],
    gnn_model,
    scale_factor: float,
    device: torch.device,
) -> list[dict]:
    """Measure every connected pair twice, once with the surrogate and once in closed form.

    Both verdicts are formed from the same thresholds and on the same geometry, which is what
    makes the disagreement between them meaningful rather than a difference of convention. The
    surrogate is only asked, never optimised, so this describes the state as it stands.

    Args:
        blocks_dict: One state of the design, in metres.
        conns: The declared connections.
        gnn_model: The surrogate.
        scale_factor: Factor between the frame the surrogate expects and metres.
        device: Device the tensors live on.

    Returns:
        One row per pair, with both overlaps and both thicknesses in millimetres, the
        feasibility probability, the two verdicts, and a flag that is ``BLUFF`` where the
        surrogate accepts what the formulas reject and ``OVERCAUTIOUS`` in the opposite case.
    """
    rows: list[dict] = []
    for i, j in conns:
        if i not in blocks_dict or j not in blocks_dict:
            continue
        b_i, b_j = blocks_dict[i], blocks_dict[j]
        # The lower block is the frozen parent and the upper one the part under test, matching
        # the order the layered repair works in. Getting this the wrong way round would measure
        # the thickness of the wrong block.
        if b_i["pos"][2] <= b_j["pos"][2]:
            anchor_idx, active_idx = i, j
            anchor, active = b_i, b_j
        else:
            anchor_idx, active_idx = j, i
            anchor, active = b_j, b_i

        pos_a_m = np.asarray(anchor["pos"], dtype=float)
        he_a_m = np.asarray(anchor["size"], dtype=float) / 2.0
        pos_c_m = np.asarray(active["pos"], dtype=float)
        he_c_m = np.asarray(active["size"], dtype=float) / 2.0

        face = infer_face(pos_a_m, he_a_m, pos_c_m, he_c_m)
        ana_ov = analytical_overlap(pos_a_m, he_a_m, pos_c_m, he_c_m, face)
        ana_th = analytical_thickness(he_c_m, face)
        ana_ok = (ana_ov >= THRESH_OVERLAP) and (ana_th <= THRESH_THICKNESS)

        size_a_scaled = torch.from_numpy(
            (np.asarray(anchor["size"], dtype=np.float32) * scale_factor).astype(
                np.float32
            )
        ).to(device)
        size_b_scaled = torch.from_numpy(
            (np.asarray(active["size"], dtype=np.float32) * scale_factor).astype(
                np.float32
            )
        ).to(device)
        pos_a_scaled = torch.from_numpy(
            (pos_a_m.astype(np.float32) * scale_factor).astype(np.float32)
        ).to(device)
        pos_b_scaled = torch.from_numpy(
            (pos_c_m.astype(np.float32) * scale_factor).astype(np.float32)
        ).to(device)

        p_bin, reg_m = predict_feasibility(
            size_a_scaled,
            size_b_scaled,
            pos_a_scaled,
            pos_b_scaled,
            gnn_model,
        )
        gnn_ov = float(reg_m[0])
        gnn_th = float(reg_m[1])
        gnn_ok = (
            (gnn_ov >= THRESH_OVERLAP)
            and (gnn_th <= THRESH_THICKNESS)
            and (p_bin >= 0.5)
        )
        is_bluff = gnn_ok and not ana_ok
        is_overcautious = (not gnn_ok) and ana_ok

        # Reported in millimetres because the criteria are stated in millimetres and the
        # differences that decide a pair are a millimetre or two wide.
        rows.append(
            {
                "pair": f"{anchor_idx}→{active_idx}",
                "face": int(face),
                "gnn_overlap": round(gnn_ov * 1000.0, 2),
                "ana_overlap": round(ana_ov * 1000.0, 2),
                "gnn_thickness": round(gnn_th * 1000.0, 2),
                "ana_thickness": round(ana_th * 1000.0, 2),
                "p_binary": round(p_bin, 3),
                "gnn_ok": "✓" if gnn_ok else "✗",
                "ana_ok": "✓" if ana_ok else "✗",
                "flag": (
                    "BLUFF"
                    if is_bluff
                    else ("OVERCAUTIOUS" if is_overcautious else "—")
                ),
            }
        )
    return rows


METRIC_COLUMNS = [
    {"name": "pair", "id": "pair"},
    {"name": "face", "id": "face"},
    {"name": "GNN ov (mm)", "id": "gnn_overlap"},
    {"name": "Ana ov (mm)", "id": "ana_overlap"},
    {"name": "GNN th (mm)", "id": "gnn_thickness"},
    {"name": "Ana th (mm)", "id": "ana_thickness"},
    {"name": "p_bin", "id": "p_binary"},
    {"name": "GNN ok", "id": "gnn_ok"},
    {"name": "Ana ok", "id": "ana_ok"},
    {"name": "flag", "id": "flag"},
]


def _metrics_table(rows: list[dict], table_id: str) -> dash_table.DataTable:
    """The pair table for one column, with bluffs shaded red and over-cautious rows amber."""
    return dash_table.DataTable(
        id=table_id,
        columns=METRIC_COLUMNS,
        data=rows,
        style_cell={
            "fontFamily": "monospace",
            "fontSize": "12px",
            "padding": "4px 6px",
            "textAlign": "center",
        },
        style_header={
            "backgroundColor": "#e2e8f0",
            "fontWeight": 600,
            "textAlign": "center",
        },
        style_data_conditional=[
            {
                "if": {"filter_query": "{flag} = 'BLUFF'"},
                "backgroundColor": "#fee2e2",
                "color": "#991b1b",
                "fontWeight": 600,
            },
            {
                "if": {"filter_query": "{flag} = 'OVERCAUTIOUS'"},
                "backgroundColor": "#fef3c7",
                "color": "#854d0e",
            },
        ],
    )


# Keyed by file, row and the position-freeze setting, because those three are what determine
# the result. A repair takes long enough that recomputing it on every redraw would make the page
# unusable.
_STAGE_CACHE: dict[tuple[str, int, bool], dict] = {}


def _compute_stages(
    csv_path: str,
    row_idx: int,
    df_cache: dict[str, pd.DataFrame],
    gnn_model,
    scale_factor: float,
    device: torch.device,
    freeze_pos_if_overlap_ok: bool = False,
) -> dict:
    """Run one design row through all three states and cache the result.

    Returns the three geometries under ``input``, ``snapped`` and ``repaired``, the connections
    under ``conns``, and the screws of each state. Writing the repaired CSV is a side effect of
    computing it.
    """
    key = (csv_path, row_idx, bool(freeze_pos_if_overlap_ok))
    if key in _STAGE_CACHE:
        return _STAGE_CACHE[key]

    df = df_cache[csv_path]
    row = df.iloc[row_idx]
    n = detect_n_blocks(df)
    blocks_input = parse_blocks(row, n)
    conns = parse_connections(row, n)
    conns_set = {(min(i, j), max(i, j)) for (i, j) in conns}

    resolved = resolve_penetrations(blocks_input, conns_set)
    snapped = apply_snaps(resolved, conns)

    repaired = gnn_repair_layered(
        snapped,
        conns,
        gnn_model,
        scale_factor,
        device,
        freeze_pos_if_overlap_ok=freeze_pos_if_overlap_ok,
    )

    orig_screws = parse_original_screws(row, n, n_screws=n)
    repaired_screws = update_screws_for_repair(
        orig_screws,
        blocks_input,
        repaired,
        conns,
    )

    stages = {
        "input": blocks_input,
        "snapped": snapped,
        "repaired": repaired,
        "conns": conns,
        "screws_input": orig_screws,
        # The closed-form corrections do not move screws, so the middle column shows them
        # where the design put them. Only the repair triggers a re-placement.
        "screws_snapped": orig_screws,
        "screws_repaired": repaired_screws,
    }
    _STAGE_CACHE[key] = stages

    _write_repaired_csv(csv_path, df, row_idx, n, repaired, repaired_screws)

    return stages


def _write_repaired_csv(
    csv_path: str,
    df: pd.DataFrame,
    row_idx: int,
    n: int,
    repaired: dict[int, dict],
    repaired_screws: dict[int, list],
) -> None:
    """Write the repaired row into ``<stem>_repaired.csv``, leaving the other rows as they are."""
    src = Path(csv_path)
    out_path = src.parent / f"{src.stem}_repaired.csv"

    # An existing output file is updated in place so that rows repaired one at a time
    # accumulate. It is discarded and rebuilt from the input if its shape no longer matches,
    # since writing one row into a file with different columns would corrupt it.
    if out_path.exists():
        try:
            out_df = pd.read_csv(out_path)
        except Exception:
            out_df = df.copy()
        if list(out_df.columns) != list(df.columns) or len(out_df) != len(df):
            out_df = df.copy()
    else:
        out_df = df.copy()

    out_row = df.iloc[row_idx].copy()
    for bi, b in repaired.items():
        for k, ax in enumerate("XYZ"):
            out_row[f"Block{bi}_Pos{ax}"] = float(b["pos"][k])
            out_row[f"Block{bi}_Size{ax}"] = float(b["size"][k])
    # Every screw column is cleared before the new values are written, so a screw the repair did
    # not reproduce leaves an empty slot rather than its stale original position.
    for bi in range(n):
        for slot in range(5):
            for ax in "XYZ":
                col = f"Block{bi}_Screw{slot}_{ax}"
                if col in out_row.index:
                    out_row[col] = np.nan
    for bi, lst in repaired_screws.items():
        for slot, xyz in lst:
            for k, ax in enumerate("XYZ"):
                col = f"Block{bi}_Screw{slot}_{ax}"
                if col in out_row.index:
                    out_row[col] = float(xyz[k])

    out_df.iloc[row_idx] = out_row
    # A failed write is reported and swallowed: the file is a convenience, and losing it must
    # not take the page down mid-session.
    try:
        out_df.to_csv(out_path, index=False, na_rep="NaN")
        print(f"[dashboard] wrote repaired CSV: {out_path}")
    except Exception as e:
        print(f"[dashboard] WARN: could not write {out_path}: {e}")


def build_app(
    csv_paths: list[Path],
    gnn_model,
    scale_factor: float,
    device: torch.device,
) -> dash.Dash:
    """Assemble the page: the file and row selectors, and the three columns of plot and table."""
    app = dash.Dash(__name__)
    app.title = "Repair Pipeline Dashboard"

    # Every file is read once at start-up and held in memory; only the repair itself is slow
    # enough to be worth caching separately.
    df_cache: dict[str, pd.DataFrame] = {str(p): pd.read_csv(p) for p in csv_paths}
    csv_options = [{"label": Path(p).name, "value": str(p)} for p in csv_paths]
    first_path = str(csv_paths[0])
    first_df = df_cache[first_path]
    row_options = [{"label": f"row {i}", "value": i} for i in range(len(first_df))]

    column_style = {
        "flex": "1",
        "minWidth": "420px",
        "padding": "8px",
    }

    app.layout = html.Div(
        style={
            "fontFamily": "sans-serif",
            "padding": "16px",
            "maxWidth": "2000px",
            "margin": "0 auto",
        },
        children=[
            html.H2(
                "Repair Pipeline — stage comparison", style={"marginBottom": "4px"}
            ),
            html.Div(
                f"{len(csv_paths)} CSV(s) available  ·  "
                "GNN-predicted vs analytical metrics shown side-by-side per pair",
                style={"color": "#64748b", "marginBottom": "16px"},
            ),
            html.Div(
                style={
                    "display": "flex",
                    "gap": "16px",
                    "marginBottom": "16px",
                    "flexWrap": "wrap",
                    "alignItems": "flex-end",
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
                            dcc.Checklist(
                                id="freeze-pos-cb",
                                options=[
                                    {
                                        "label": " Freeze pos if overlap already OK"
                                        " (size-only repair)",
                                        "value": "on",
                                    }
                                ],
                                value=["on"],
                                style={"fontSize": "13px"},
                            ),
                        ],
                        style={"alignSelf": "center"},
                    ),
                    html.Div(
                        id="status",
                        style={
                            "color": "#64748b",
                            "fontFamily": "monospace",
                            "fontSize": "13px",
                            "alignSelf": "center",
                        },
                    ),
                ],
            ),
            html.Div(
                style={"display": "flex", "flexWrap": "wrap", "gap": "8px"},
                children=[
                    html.Div(
                        style=column_style,
                        children=[
                            html.H3("1. Input", style={"margin": "4px 0"}),
                            dcc.Graph(id="fig-input", style={"height": "520px"}),
                            html.Div(id="tbl-input"),
                        ],
                    ),
                    html.Div(
                        style=column_style,
                        children=[
                            html.H3(
                                "2. After Snap + Pen-solve", style={"margin": "4px 0"}
                            ),
                            dcc.Graph(id="fig-snap", style={"height": "520px"}),
                            html.Div(id="tbl-snap"),
                        ],
                    ),
                    html.Div(
                        style=column_style,
                        children=[
                            html.H3("3. GNN Repaired", style={"margin": "4px 0"}),
                            dcc.Graph(id="fig-repaired", style={"height": "520px"}),
                            html.Div(id="tbl-repaired"),
                        ],
                    ),
                ],
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
        df = df_cache[csv_path]
        opts = [{"label": f"row {i}", "value": i} for i in range(len(df))]
        return opts, 0

    @app.callback(
        Output("fig-input", "figure"),
        Output("fig-snap", "figure"),
        Output("fig-repaired", "figure"),
        Output("tbl-input", "children"),
        Output("tbl-snap", "children"),
        Output("tbl-repaired", "children"),
        Output("status", "children"),
        Input("csv-dd", "value"),
        Input("row-dd", "value"),
        Input("freeze-pos-cb", "value"),
    )
    def update_all(csv_path, row_idx, freeze_pos_value):
        """Recompute the three states of the selected design and redraw all six panels."""
        row_idx = int(row_idx or 0)
        freeze_pos = "on" in (freeze_pos_value or [])
        stages = _compute_stages(
            csv_path,
            row_idx,
            df_cache,
            gnn_model,
            scale_factor,
            device,
            freeze_pos_if_overlap_ok=freeze_pos,
        )
        conns = stages["conns"]

        fig_in = _build_3d_figure(
            stages["input"],
            f"Input — row {row_idx}",
            screws=stages.get("screws_input"),
        )
        fig_sn = _build_3d_figure(
            stages["snapped"],
            "After Snap + Pen-solve",
            screws=stages.get("screws_snapped"),
        )
        fig_rp = _build_3d_figure(
            stages["repaired"],
            "GNN Repaired",
            screws=stages.get("screws_repaired"),
        )

        m_in = _compute_pair_metrics(
            stages["input"],
            conns,
            gnn_model,
            scale_factor,
            device,
        )
        m_sn = _compute_pair_metrics(
            stages["snapped"],
            conns,
            gnn_model,
            scale_factor,
            device,
        )
        m_rp = _compute_pair_metrics(
            stages["repaired"],
            conns,
            gnn_model,
            scale_factor,
            device,
        )

        # The summary counts the closed-form verdict, not the surrogate's, so a design is only
        # called repaired when its geometry says so.
        n_bluffs = sum(1 for r in m_rp if r["flag"] == "BLUFF")
        n_ok = sum(1 for r in m_rp if r["ana_ok"] == "✓")
        status = (
            f"file: {Path(csv_path).name}  ·  row {row_idx}  ·  "
            f"{len(conns)} pairs  ·  repaired: {n_ok}/{len(m_rp)} analytically ok  ·  "
            f"bluffs: {n_bluffs}"
        )

        return (
            fig_in,
            fig_sn,
            fig_rp,
            _metrics_table(m_in, "tbl-data-input"),
            _metrics_table(m_sn, "tbl-data-snap"),
            _metrics_table(m_rp, "tbl-data-repaired"),
            status,
        )

    return app


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--port",
        type=int,
        default=8053,
        help="Dash port (default 8053)",
    )
    args = parser.parse_args()

    pipeline_dir = Path(__file__).resolve().parent / "new_csv"
    csv_paths: list[Path] = []
    missing: list[str] = []
    for name in CSV_FILES:
        p = pipeline_dir / name
        if p.exists():
            csv_paths.append(p)
        else:
            missing.append(name)
    if missing:
        print("WARNING: the following CSVs from CSV_FILES were not found and skipped:")
        for m in missing:
            print(f"  - {m}")
    if not csv_paths:
        print(f"ERROR: no CSVs from CSV_FILES found in {pipeline_dir}/")
        return 1

    print(f"Loading {len(csv_paths)} CSV(s):")
    for p in csv_paths:
        print(f"  - {p.name}")

    print("\nLoading GNN model...")
    device = torch.device("cpu")
    gnn_model, scale_factor = load_models(device)
    print(f"scale_factor = {scale_factor:.4f}")

    app = build_app(csv_paths, gnn_model, scale_factor, device)
    print(f"\nDashboard ready at http://127.0.0.1:{args.port}")
    print("(first row of each CSV is pre-computed lazily on click)")
    app.run(debug=False, port=args.port)
    return 0


if __name__ == "__main__":
    sys.exit(main())
