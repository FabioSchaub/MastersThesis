"""Dash application that shows one design repaired in parameter space next to latent space.

Three columns for the selected design, on one shared camera and one shared cubic axis range:
the generated assembly after the analytical pre-processing, the same assembly repaired by the
parameter branch, and repaired by the latent branch. Below them a per-joint table gives overlap
and thickness in millimetres for all three states, the geometric verdict, and the probability
the surrogate assigned; a joint the surrogate calls feasible while the geometry does not is
marked as a bluff. This is the source of the before-and-after figure of Part II, which is why
the shared camera matters: without it the three columns cannot be compared by eye.

It reads nothing but ``results/dashboard_data/repair_comparison.json``, written by
``tools/export_repair_comparison.py``. No checkpoint, encoder or dataset is loaded, so the app
starts immediately and, more importantly, can show both branches at once even though they live
on separate branches and cannot be imported into the same interpreter. It writes nothing.

Run:
    python tools/repair_dashboard.py                 # http://127.0.0.1:8060
    python tools/repair_dashboard.py --port 8061
    python tools/repair_dashboard.py --data <path to json>

The cuboid and floor rendering helpers and the column layout follow
``pipeline/dashboard.py``, but everything is in millimetres here rather than metres.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, dash_table, dcc, html

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA = ROOT / "results" / "dashboard_data" / "repair_comparison.json"

# Column order, and also the key order inside the exported JSON.
STATES: tuple[str, str, str] = ("initial", "param", "latent")
AXIS_NAMES = ("X", "Y", "Z")

BLOCK_COLORS = [
    "#3b82f6", "#10b981", "#f59e0b", "#ef4444", "#8b5cf6",
    "#ec4899", "#06b6d4", "#f97316", "#14b8a6", "#a855f7",
    "#0ea5e9", "#84cc16",
]
BLOCK_OPACITY = 0.62
DIM_OPACITY = 0.07
FLOOR_COLOR = "#cbd5e1"
FLOOR_OPACITY = 0.18
PARENT_COLOR = "#1d4ed8"
CHILD_COLOR = "#ea580c"

C_PASS_BG, C_PASS_FG = "#dcfce7", "#166534"
C_FAIL_BG, C_FAIL_FG = "#fee2e2", "#991b1b"
C_BLUFF_BG, C_BLUFF_FG = "#fef3c7", "#92400e"

CAMERAS: dict[str, dict] = {
    "Isometric": dict(eye=dict(x=1.55, y=1.55, z=1.05)),
    "Front (−Y)": dict(eye=dict(x=0.0, y=-2.35, z=0.35)),
    "Side (+X)": dict(eye=dict(x=2.35, y=0.0, z=0.35)),
    "Top (+Z)": dict(eye=dict(x=0.0, y=-0.05, z=2.6)),
}
DEFAULT_CAMERA = "Isometric"


# Triangle indices into the eight corners produced by _corners, two per cuboid face.
_FACES = np.array(
    [
        [0, 1, 2], [1, 3, 2], [4, 6, 5], [5, 6, 7],
        [0, 2, 4], [2, 6, 4], [1, 5, 3], [3, 5, 7],
        [0, 4, 1], [1, 4, 5], [2, 3, 6], [3, 7, 6],
    ]
)


def _corners(center: np.ndarray, size: np.ndarray) -> np.ndarray:
    """The eight corners of an axis-aligned cuboid, in the order the _FACES table assumes."""
    cx, cy, cz = center
    hx, hy, hz = size / 2.0
    return np.array(
        [
            [cx - hx, cy - hy, cz - hz], [cx + hx, cy - hy, cz - hz],
            [cx - hx, cy + hy, cz - hz], [cx + hx, cy + hy, cz - hz],
            [cx - hx, cy - hy, cz + hz], [cx + hx, cy - hy, cz + hz],
            [cx - hx, cy + hy, cz + hz], [cx + hx, cy + hy, cz + hz],
        ]
    )


def _cube_mesh(
    center: np.ndarray, size: np.ndarray, color: str, name: str,
    opacity: float, hover: str,
) -> go.Mesh3d:
    """One block as a translucent triangle mesh."""
    v = _corners(center, size)
    return go.Mesh3d(
        x=v[:, 0], y=v[:, 1], z=v[:, 2],
        i=_FACES[:, 0], j=_FACES[:, 1], k=_FACES[:, 2],
        color=color, opacity=opacity, flatshading=True,
        name=name, showlegend=False, hoverinfo="text", text=hover,
    )


def _cube_edges(
    center: np.ndarray, size: np.ndarray, color: str, width: float
) -> go.Scatter3d:
    """The twelve edges of a block, drawn on top of the mesh so its extent stays readable."""
    v = _corners(center, size)
    order = [
        (0, 1), (1, 3), (3, 2), (2, 0), (4, 5), (5, 7), (7, 6), (6, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]
    xs: list = []
    ys: list = []
    zs: list = []
    for a, b in order:
        xs += [v[a, 0], v[b, 0], None]
        ys += [v[a, 1], v[b, 1], None]
        zs += [v[a, 2], v[b, 2], None]
    return go.Scatter3d(
        x=xs, y=ys, z=zs, mode="lines",
        line=dict(color=color, width=width),
        showlegend=False, hoverinfo="skip",
    )


def _floor_plane(x_rng: tuple[float, float], y_rng: tuple[float, float]) -> go.Mesh3d:
    """The table plane at z = 0, drawn so that a block sunk into it is visible."""
    x0, x1 = x_rng
    y0, y1 = y_rng
    return go.Mesh3d(
        x=[x0, x1, x1, x0], y=[y0, y0, y1, y1], z=[0.0, 0.0, 0.0, 0.0],
        i=[0, 0], j=[1, 2], k=[2, 3],
        color=FLOOR_COLOR, opacity=FLOOR_OPACITY,
        showlegend=False, hoverinfo="skip",
    )


def _shared_ranges(design: dict) -> tuple[tuple[float, float], ...]:
    """One cubic bounding box over all three states, so the columns share a scale."""
    lo = np.full(3, np.inf)
    hi = np.full(3, -np.inf)
    for state in STATES:
        for g in design["blocks"][state].values():
            c = np.asarray(g["center_mm"], dtype=float)
            h = np.asarray(g["edge_lengths_mm"], dtype=float) / 2.0
            lo = np.minimum(lo, c - h)
            hi = np.maximum(hi, c + h)
    lo[2] = min(lo[2], 0.0)  # the table plane is a reference and must stay in frame
    center = (lo + hi) / 2.0
    half = float(np.max(hi - lo)) / 2.0 * 1.12 + 5.0
    return tuple((float(center[k] - half), float(center[k] + half)) for k in range(3))


def _build_figure(
    design: dict, state: str, camera_key: str, highlight: str | None
) -> go.Figure:
    """One of the three 3D columns, with the joint counts in its subtitle."""
    ranges = _shared_ranges(design)
    blocks = design["blocks"][state]

    hi_parent = hi_child = None
    if highlight and highlight != "__all__":
        for p in design["pairs"]:
            if p["pair_id"] == highlight:
                hi_parent, hi_child = str(p["parent"]), str(p["child"])
                break

    fig = go.Figure()
    fig.add_trace(_floor_plane(ranges[0], ranges[1]))

    for bid in sorted(blocks, key=lambda s: int(s)):
        g = blocks[bid]
        c = np.asarray(g["center_mm"], dtype=float)
        s = np.asarray(g["edge_lengths_mm"], dtype=float)
        if not np.isfinite(c).all() or not np.isfinite(s).all():
            continue

        if hi_parent is None:
            color = BLOCK_COLORS[int(bid) % len(BLOCK_COLORS)]
            opacity, edge_w, edge_c = BLOCK_OPACITY, 2.0, color
        elif bid == hi_parent:
            color, opacity, edge_w, edge_c = PARENT_COLOR, 0.78, 4.0, "#1e3a8a"
        elif bid == hi_child:
            color, opacity, edge_w, edge_c = CHILD_COLOR, 0.78, 4.0, "#7c2d12"
        else:
            color = BLOCK_COLORS[int(bid) % len(BLOCK_COLORS)]
            opacity, edge_w, edge_c = DIM_OPACITY, 1.0, "#cbd5e1"

        hover = (
            f"<b>Block {bid}</b><br>"
            f"centre  {c[0]:.1f} / {c[1]:.1f} / {c[2]:.1f} mm<br>"
            f"edges   {s[0]:.1f} × {s[1]:.1f} × {s[2]:.1f} mm"
        )
        fig.add_trace(_cube_mesh(c, s, color, f"Block {bid}", opacity, hover))
        fig.add_trace(_cube_edges(c, s, edge_c, edge_w))

    n_ok = sum(1 for p in design["pairs"] if p[state]["analytical_ok"])
    n_tot = len(design["pairs"])
    subtitle = f"{n_ok}/{n_tot} joints geometrically feasible"
    if state != "initial":
        n_bluff = sum(1 for p in design["pairs"] if p[state]["bluff"])
        subtitle += f"   ·   {n_bluff} surrogate bluff{'' if n_bluff == 1 else 's'}"

    axis_common = dict(
        showbackground=True, backgroundcolor="#f8fafc",
        gridcolor="#e2e8f0", zerolinecolor="#94a3b8",
        tickfont=dict(size=11),
    )
    def _axis(label: str, rng: tuple[float, float]) -> dict:
        """One axis with an explicit range, which is what ties the three columns together."""
        return dict(
            title=dict(text=label, font=dict(size=13)),
            range=list(rng), **axis_common,
        )

    fig.update_layout(
        title=dict(
            text=f"<b>{STATE_TITLES[state]}</b><br>"
                 f"<span style='font-size:12px;color:#64748b'>{subtitle}</span>",
            x=0.5, xanchor="center", font=dict(size=15),
        ),
        scene=dict(
            xaxis=_axis("x (mm)", ranges[0]),
            yaxis=_axis("y (mm)", ranges[1]),
            zaxis=_axis("z (mm)", ranges[2]),
            aspectmode="cube",
            camera=CAMERAS[camera_key],
        ),
        margin=dict(l=0, r=0, t=58, b=0),
        paper_bgcolor="white",
        font=dict(family="Inter, Segoe UI, sans-serif", size=13),
        # Keyed on design and preset so that a manually dragged camera survives a redraw but is
        # discarded when the user switches to a different design or preset.
        uirevision=f"{design['design']}|{camera_key}",
    )
    return fig


def _mark(ok: bool) -> str:
    """The geometric verdict of one joint as a table cell."""
    return "✓ OK" if ok else "✗ fail"


def _fmt_p(p: float) -> str:
    """Format a probability, falling back to exponent notation once it rounds away to zero."""
    if p >= 0.001:
        return f"{p:.3f}"
    return f"{p:.1e}"


def _table_rows(design: dict) -> list[dict]:
    """One table row per joint, with overlap and thickness in millimetres for all three states."""
    rows = []
    for p in design["pairs"]:
        rows.append(
            {
                "joint": f"{p['parent']} → {p['child']}",
                "axis": AXIS_NAMES[p["contact_axis"]],
                "ov_ini": round(p["initial"]["overlap_mm"], 2),
                "ov_par": round(p["param"]["overlap_mm"], 2),
                "ov_lat": round(p["latent"]["overlap_mm"], 2),
                "th_ini": round(p["initial"]["thickness_mm"], 2),
                "th_par": round(p["param"]["thickness_mm"], 2),
                "th_lat": round(p["latent"]["thickness_mm"], 2),
                "ok_ini": _mark(p["initial"]["analytical_ok"]),
                "ok_par": _mark(p["param"]["analytical_ok"]),
                "ok_lat": _mark(p["latent"]["analytical_ok"]),
                "p_par": _fmt_p(p["param"]["p_feasible"]),
                "p_lat": _fmt_p(p["latent"]["p_feasible"]),
                # Not shown as columns; the conditional styling can only branch on cell
                # values, so the flags have to travel in the row.
                "_id": p["pair_id"],
                "_bluff_par": "1" if p["param"]["bluff"] else "0",
                "_bluff_lat": "1" if p["latent"]["bluff"] else "0",
            }
        )
    return rows


def _table_columns(th_ov: float, th_th: float) -> list[dict]:
    """Two-level column headers, with the two thresholds spelled out in the lower level."""
    ov = f"overlap ≥ {th_ov:g} mm"
    th = f"thickness ≤ {th_th:g} mm"
    return [
        {"name": ["", "joint"], "id": "joint"},
        {"name": ["", "axis"], "id": "axis"},
        {"name": ["1 · initial", ov], "id": "ov_ini"},
        {"name": ["1 · initial", th], "id": "th_ini"},
        {"name": ["1 · initial", "geometry"], "id": "ok_ini"},
        {"name": ["2 · parameter repair", ov], "id": "ov_par"},
        {"name": ["2 · parameter repair", th], "id": "th_par"},
        {"name": ["2 · parameter repair", "geometry"], "id": "ok_par"},
        {"name": ["2 · parameter repair", "p(feasible)"], "id": "p_par"},
        {"name": ["3 · latent repair", ov], "id": "ov_lat"},
        {"name": ["3 · latent repair", th], "id": "th_lat"},
        {"name": ["3 · latent repair", "geometry"], "id": "ok_lat"},
        {"name": ["3 · latent repair", "p(feasible)"], "id": "p_lat"},
    ]


def _table_styles(th_ov: float, th_th: float, highlight: str | None) -> list[dict]:
    """Conditional cell colouring: verdict, threshold violations, bluffs, selected joint."""
    styles: list[dict] = []
    for col in ("ok_ini", "ok_par", "ok_lat"):
        styles += [
            {
                "if": {"filter_query": f'{{{col}}} contains "OK"', "column_id": col},
                "backgroundColor": C_PASS_BG, "color": C_PASS_FG, "fontWeight": 700,
            },
            {
                "if": {"filter_query": f'{{{col}}} contains "fail"', "column_id": col},
                "backgroundColor": C_FAIL_BG, "color": C_FAIL_FG, "fontWeight": 700,
            },
        ]
    for col in ("ov_ini", "ov_par", "ov_lat"):
        styles.append(
            {
                "if": {"filter_query": f"{{{col}}} < {th_ov}", "column_id": col},
                "color": C_FAIL_FG, "fontWeight": 700,
            }
        )
    for col in ("th_ini", "th_par", "th_lat"):
        styles.append(
            {
                "if": {"filter_query": f"{{{col}}} > {th_th}", "column_id": col},
                "color": C_FAIL_FG, "fontWeight": 700,
            }
        )
    for col, flag in (("p_par", "_bluff_par"), ("p_lat", "_bluff_lat")):
        styles.append(
            {
                "if": {"filter_query": f'{{{flag}}} = "1"', "column_id": col},
                "backgroundColor": C_BLUFF_BG, "color": C_BLUFF_FG, "fontWeight": 700,
            }
        )
    # Each of these is the first column of one of the three states, so a left border here
    # separates the three groups.
    for col in ("ov_ini", "ov_par", "ov_lat"):
        styles.append({"if": {"column_id": col}, "borderLeft": "2px solid #94a3b8"})
    if highlight and highlight != "__all__":
        styles.append(
            {
                "if": {"filter_query": f'{{_id}} = "{highlight}"'},
                "backgroundColor": "#eef2ff",
                "border": "2px solid #4338ca",
            }
        )
    return styles


# Column titles as the exporter wrote them, so the app does not name the two branches itself.
STATE_TITLES: dict[str, str] = {}


def build_app(data: dict) -> Dash:
    """Assemble the layout and the callbacks around one already-loaded comparison file."""
    global STATE_TITLES
    meta = data["meta"]
    STATE_TITLES = meta["state_titles"]
    th_ov = float(meta["thresh_overlap_mm"])
    th_th = float(meta["thresh_thickness_mm"])
    designs = {d["design"]: d for d in data["designs"]}
    design_options = [
        {"label": d["label"], "value": d["design"]} for d in data["designs"]
    ]
    first = data["designs"][0]["design"]

    app = Dash(__name__)
    app.title = "Part II — repair comparison"

    graph_config = {
        "displaylogo": False,
        "toImageButtonOptions": {
            "format": "png", "scale": 3, "filename": "repair_comparison",
        },
    }
    col_style = {"flex": "1 1 0", "minWidth": "330px"}
    label_style = {
        "fontWeight": 600, "fontSize": "13px", "color": "#334155",
        "display": "block", "marginBottom": "3px",
    }

    app.layout = html.Div(
        style={
            "fontFamily": "Inter, Segoe UI, sans-serif",
            "padding": "18px 22px",
            "background": "white",
            "maxWidth": "2100px",
            "margin": "0 auto",
        },
        children=[
            html.H2(
                "Part II — gradient-based repair: parameter space vs. latent space",
                style={"margin": "0 0 4px 0", "fontSize": "23px", "color": "#0f172a"},
            ),
            html.Div(
                [
                    f"Thresholds: overlap ≥ {th_ov:g} mm, thickness ≤ {th_th:g} mm.  ",
                    html.Span(
                        f"Parameter branch: {meta['variants']['param']['checkpoint']}  ·  "
                        f"latent branch: {meta['variants']['latent']['checkpoint']}",
                        style={"color": "#94a3b8"},
                    ),
                ],
                style={"fontSize": "12.5px", "color": "#64748b",
                       "marginBottom": "14px"},
            ),
            html.Div(
                style={"display": "flex", "gap": "22px", "flexWrap": "wrap",
                       "alignItems": "flex-end", "marginBottom": "10px"},
                children=[
                    html.Div(
                        [
                            html.Label("Design", style=label_style),
                            dcc.Dropdown(
                                id="design-dd", options=design_options, value=first,
                                clearable=False, style={"width": "420px"},
                            ),
                        ]
                    ),
                    html.Div(
                        [
                            html.Label("Highlight joint", style=label_style),
                            dcc.Dropdown(
                                id="joint-dd", value="__all__", clearable=False,
                                style={"width": "230px"},
                            ),
                        ]
                    ),
                    html.Div(
                        [
                            html.Label("Camera", style=label_style),
                            dcc.Dropdown(
                                id="cam-dd",
                                options=[{"label": k, "value": k} for k in CAMERAS],
                                value=DEFAULT_CAMERA, clearable=False,
                                style={"width": "170px"},
                            ),
                        ]
                    ),
                    html.Div(
                        id="summary",
                        style={"fontSize": "13.5px", "color": "#334155",
                               "paddingBottom": "6px"},
                    ),
                ],
            ),
            html.Div(
                style={"display": "flex", "gap": "10px", "flexWrap": "wrap"},
                children=[
                    html.Div(
                        dcc.Graph(id=f"view-{s}", style={"height": "540px"},
                                  config=graph_config),
                        style=col_style,
                    )
                    for s in STATES
                ],
            ),
            html.Div(
                style={"marginTop": "14px"},
                children=[
                    html.H3(
                        "Per-joint metrics",
                        style={"fontSize": "17px", "margin": "0 0 6px 0",
                               "color": "#0f172a"},
                    ),
                    html.Div(
                        "Red = threshold violated · green/red = verdict of the "
                        "geometric check · amber p(feasible) = surrogate says "
                        "feasible while the geometry does not (bluff).",
                        style={"fontSize": "12.5px", "color": "#64748b",
                               "marginBottom": "8px"},
                    ),
                    dash_table.DataTable(
                        id="joint-table",
                        columns=_table_columns(th_ov, th_th),
                        merge_duplicate_headers=True,
                        style_table={"overflowX": "auto"},
                        style_cell={
                            "fontFamily": "SFMono-Regular, Consolas, monospace",
                            "fontSize": "13px", "padding": "6px 10px",
                            "textAlign": "center", "border": "1px solid #e2e8f0",
                        },
                        style_header={
                            "backgroundColor": "#f1f5f9", "fontWeight": 700,
                            "fontSize": "12.5px", "textAlign": "center",
                            "border": "1px solid #cbd5e1",
                            "fontFamily": "Inter, Segoe UI, sans-serif",
                        },
                        style_cell_conditional=[
                            {"if": {"column_id": "joint"}, "fontWeight": 700,
                             "textAlign": "left"},
                        ],
                    ),
                ],
            ),
        ],
    )

    @app.callback(
        Output("joint-dd", "options"),
        Output("joint-dd", "value"),
        Input("design-dd", "value"),
    )
    def _joint_options(design_key):
        """Refill the joint selector for the chosen design and reset it to show all joints."""
        d = designs[design_key]
        opts = [{"label": "all joints", "value": "__all__"}] + [
            {
                "label": f"{p['parent']} → {p['child']}  "
                         f"({AXIS_NAMES[p['contact_axis']]}-contact)",
                "value": p["pair_id"],
            }
            for p in d["pairs"]
        ]
        return opts, "__all__"

    @app.callback(
        Output("view-initial", "figure"),
        Output("view-param", "figure"),
        Output("view-latent", "figure"),
        Output("joint-table", "data"),
        Output("joint-table", "style_data_conditional"),
        Output("summary", "children"),
        Input("design-dd", "value"),
        Input("joint-dd", "value"),
        Input("cam-dd", "value"),
    )
    def _update(design_key, joint, camera_key):
        """Redraw the three views, the table and the summary line from the selection."""
        d = designs[design_key]
        figs = [_build_figure(d, s, camera_key, joint) for s in STATES]
        rows = _table_rows(d)
        styles = _table_styles(th_ov, th_th, joint)
        n = len(d["pairs"])
        n_par = sum(1 for p in d["pairs"] if p["param"]["analytical_ok"])
        n_lat = sum(1 for p in d["pairs"] if p["latent"]["analytical_ok"])
        n_ini = sum(1 for p in d["pairs"] if p["initial"]["analytical_ok"])
        summary = html.Span(
            [
                f"{d['n_blocks']} blocks · {n} joints — feasible: ",
                html.B(f"{n_ini}/{n}"), " initial, ",
                html.B(f"{n_par}/{n}"), " parameter, ",
                html.B(f"{n_lat}/{n}"), " latent",
            ]
        )
        return figs[0], figs[1], figs[2], rows, styles, summary

    # Dragging one view must move the other two, otherwise the comparison stops being a
    # comparison. This runs in the browser rather than as a server callback because a round
    # trip per frame would make the drag unusable; it fails silently, since losing the sync is
    # a cosmetic problem and the preset dropdown still sets all three at once.
    app.clientside_callback(
        """
        function(r0, r1, r2) {
            const nu = window.dash_clientside.no_update;
            try {
                const ctx = window.dash_clientside.callback_context;
                if (!ctx || !ctx.triggered || !ctx.triggered.length) return nu;
                const src = ctx.triggered[0].prop_id.split('.')[0];
                const rel = ctx.triggered[0].value;
                if (!rel || !rel['scene.camera']) return nu;
                const cam = rel['scene.camera'];
                ['view-initial', 'view-param', 'view-latent'].forEach(function (id) {
                    if (id === src) return;
                    const host = document.getElementById(id);
                    if (!host) return;
                    const gd = host.getElementsByClassName('js-plotly-plot')[0];
                    if (gd && window.Plotly) {
                        window.Plotly.relayout(gd, {'scene.camera': cam});
                    }
                });
            } catch (e) { /* no-op */ }
            return nu;
        }
        """,
        Output("summary", "title"),
        Input("view-initial", "relayoutData"),
        Input("view-param", "relayoutData"),
        Input("view-latent", "relayoutData"),
        prevent_initial_call=True,
    )

    return app


def main() -> int:
    ap = argparse.ArgumentParser(description="Part II repair dashboard")
    ap.add_argument("--port", type=int, default=8060)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--data", default=str(DEFAULT_DATA))
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        print(
            f"Data file not found: {data_path}\n"
            "Generate it first with tools/export_repair_comparison.py "
            "(see the module docstring)."
        )
        return 1
    data = json.loads(data_path.read_text(encoding="utf-8"))

    app = build_app(data)
    print(
        f"Part II repair dashboard - {len(data['designs'])} designs  ->  "
        f"http://{args.host}:{args.port}"
    )
    app.run(host=args.host, port=args.port, debug=args.debug)
    return 0


if __name__ == "__main__":
    sys.exit(main())
