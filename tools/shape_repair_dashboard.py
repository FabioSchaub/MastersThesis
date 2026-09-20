"""Part III repair dashboard — one configuration, three states side by side.

Screenshot-oriented Dash app for the thesis figure of Part III. For one selected
configuration it shows

    1  Start                            the sampled infeasible pair
    2  Size and placement               the honest branch, shape frozen
    3  Size, placement and shape code   the shape branch, which bluffs

on a shared camera and a shared cubic axis range, with the two screwdriving
criteria measured on the geometry underneath each view.

The app reads ONLY the bundle written by `tools/export_shape_repair_views.py`,
so it starts immediately and needs neither a checkpoint nor the dataset.

Run:
    python tools/shape_repair_dashboard.py                  # http://127.0.0.1:8061
    python tools/shape_repair_dashboard.py --port 8062
    python tools/shape_repair_dashboard.py --data <bundle without extension>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, dcc, html

ROOT = Path(__file__).resolve().parent.parent
# The bundle is a local artefact, not part of the repository. It is kept next to
# the other local data outside the checkout, so that cleaning the working tree
# cannot delete it; the in-repo location is still accepted.
DATA_CANDIDATES = (ROOT.parent / "data_and_latents" / "shape_repair_views",
                   ROOT / "results" / "dashboard_data" / "shape_repair_views")

C_ANCHOR = "#9aa0a6"
C_OK = "#2e9e5b"
C_FAIL = "#c0392b"
C_START = "#4a7fb5"
GRID = "#d8d8d8"

# A fixed set of viewpoints. The three views share whichever is selected, so that the columns of
# a screenshot differ only in the configuration they show and not in how it is seen.
CAMERAS = {
    "isometric": dict(eye=dict(x=1.45, y=1.45, z=1.05)),
    "front":     dict(eye=dict(x=0.0, y=-2.2, z=0.35)),
    "side":      dict(eye=dict(x=2.2, y=0.0, z=0.35)),
    "top":       dict(eye=dict(x=0.0, y=-0.01, z=2.4)),
}


def load_bundle(stem: Path):
    """Read the two files of a bundle, given their shared path without the extension.

    Returns:
        The configurations and their measurements, and the meshes by name, each holding its
        vertices and its faces.
    """
    meta_path = stem.with_suffix(".json")
    mesh_path = stem.with_suffix(".npz")
    bundle = json.loads(meta_path.read_text())
    npz = np.load(mesh_path)
    meshes = {}
    for key in npz.files:
        kind, name = key.split("::", 1)
        meshes.setdefault(name, {})[kind] = npz[key]
    return bundle, meshes


def placed(mesh, bbox, pos):
    """Scale a canonical mesh uniformly onto a bounding box and translate it to a position.

    The same placement the export tool describes and the simulator applies.
    """
    v = mesh["v"]
    extent = v.max(0) - v.min(0)
    s = float(np.max(bbox)) / float(np.max(extent))
    return v * s + np.asarray(pos, dtype=np.float32), mesh["f"]


def verdict_label(view) -> tuple[str, str]:
    """The caption and colour of one view.

    A view without a surrogate confidence is the starting state, which was never repaired and so
    can only be feasible or not. For the other two, a configuration the surrogate accepted and
    the geometry rejected is called out as such rather than simply shown as infeasible.
    """
    if view["p_good"] is None:
        return ("infeasible at the start" if not view["geo_ok"] else "feasible at the start",
                C_FAIL if not view["geo_ok"] else C_OK)
    if view["bluff"]:
        return "bluff: surrogate accepts, geometry does not", C_FAIL
    if view["geo_ok"]:
        return "feasible", C_OK
    return "infeasible", C_FAIL


def make_figure(cfg, meshes, idx, camera, span_mm):
    """Draw one view of one configuration: the base in grey, the part under test coloured.

    Args:
        idx: Which of the three views to draw.
        span_mm: Edge of the cubic range, shared by all three views so that a change of size
            between them is visible as a change of size and not absorbed by the axes.
    """
    view = cfg["views"][idx]
    anchor = cfg["anchor"]
    av, af = placed(meshes[anchor["mesh"].split("::", 1)[1]], anchor["bbox"], anchor["pos"])
    cv, cf = placed(meshes[view["mesh"].split("::", 1)[1]], view["bbox"], view["pos"])
    # Drawn in millimetres, the unit the two criteria are reported in below each view.
    av, cv = av * 1e3, cv * 1e3

    colour = C_START if idx == 0 else (C_OK if view["geo_ok"] else C_FAIL)
    fig = go.Figure()
    fig.add_trace(go.Mesh3d(x=av[:, 0], y=av[:, 1], z=av[:, 2],
                            i=af[:, 0], j=af[:, 1], k=af[:, 2],
                            color=C_ANCHOR, opacity=0.55, flatshading=True,
                            name="base", hoverinfo="skip"))
    fig.add_trace(go.Mesh3d(x=cv[:, 0], y=cv[:, 1], z=cv[:, 2],
                            i=cf[:, 0], j=cf[:, 1], k=cf[:, 2],
                            color=colour, opacity=1.0, flatshading=True,
                            name="child", hoverinfo="skip"))

    centre = np.array([anchor["pos"][0], anchor["pos"][1], anchor["pos"][2]]) * 1e3
    half = span_mm / 2
    rng = [[centre[a] - half, centre[a] + half] for a in range(3)]
    axis = dict(showbackground=True, backgroundcolor="white", gridcolor=GRID,
                zerolinecolor=GRID, showticklabels=False, title="")
    fig.update_layout(
        scene=dict(xaxis={**axis, "range": rng[0]},
                   yaxis={**axis, "range": rng[1]},
                   zaxis={**axis, "range": rng[2]},
                   aspectmode="cube", camera=camera),
        margin=dict(l=0, r=0, t=0, b=0), height=380,
        showlegend=False, paper_bgcolor="white",
        uirevision="keep",
    )
    return fig


def metric_block(cfg, idx, thresholds):
    """The measurements printed under one view, each coloured by whether it meets its threshold."""
    view = cfg["views"][idx]
    label, colour = verdict_label(view)
    ov, th = view["overlap_mm"], view["thickness_mm"]
    ov_ok = ov >= thresholds[0]
    th_ok = th <= thresholds[1]

    def row(name, value, ok, unit=" mm"):
        """One measurement, green if it meets its threshold and red if it does not."""
        return html.Div([
            html.Span(name, style={"color": "#555"}),
            html.Span(f"{value:.1f}{unit}", style={
                "float": "right", "fontWeight": 600,
                "color": C_OK if ok else C_FAIL}),
        ], style={"padding": "2px 0"})

    rows = [
        row(f"contact  (≥ {thresholds[0]:.0f} mm)", ov, ov_ok),
        row(f"thickness  (≤ {thresholds[1]:.0f} mm)", th, th_ok),
    ]
    if view["p_good"] is not None:
        rows.append(row("surrogate confidence", view["p_good"], True, unit=""))
    rows.append(html.Div(label, style={
        "marginTop": "6px", "padding": "3px 6px", "borderRadius": "3px",
        "background": colour, "color": "white", "fontSize": "13px",
        "textAlign": "center"}))
    return html.Div(rows, style={"fontSize": "14px", "padding": "0 8px"})


def build_app(bundle, meshes):
    """Assemble the page: a filter, a configuration, a viewpoint, and the three columns."""
    cfgs = {c["id"]: c for c in bundle["configs"]}
    thr = (bundle["meta"]["thresh_overlap_mm"], bundle["meta"]["thresh_thickness_mm"])

    def kind(c):
        """Classify a configuration by how the two repairs came out.

        This is what makes the figure findable: the case the argument rests on, where the honest
        repair works and the shape repair only appears to, is one group among a hundred.
        """
        honest, shape = c["views"][1], c["views"][2]
        if honest["geo_ok"] and shape["bluff"]:
            return "contrast"
        if honest["geo_ok"] and shape["geo_ok"]:
            return "both_ok"
        if not honest["geo_ok"] and shape["bluff"]:
            return "shape_bluff"
        return "other"

    groups = {
        "contrast": "honest repair works, shape repair bluffs",
        "both_ok": "both branches feasible",
        "shape_bluff": "honest fails, shape repair bluffs",
        "other": "everything else",
        "all": "all configurations",
    }
    kinds = {i: kind(c) for i, c in cfgs.items()}

    def options(group):
        """The configurations of one group, each labelled with how both repairs came out."""
        ids = [i for i in sorted(cfgs) if group == "all" or kinds[i] == group]
        out = []
        for i in ids:
            c = cfgs[i]
            h = "feasible" if c["views"][1]["geo_ok"] else "infeasible"
            s = "bluff" if c["views"][2]["bluff"] else (
                "feasible" if c["views"][2]["geo_ok"] else "infeasible")
            out.append({"label": f"#{i:03d}   {c['child_name'].replace('.usd','')}"
                                 f"   •  honest: {h}   •  shape: {s}",
                        "value": i})
        return out

    app = Dash(__name__)
    # Opens on the contrasting case if the bundle contains one, since that is what the page was
    # built to show, and falls back to everything otherwise.
    first_group = "contrast" if any(k == "contrast" for k in kinds.values()) else "all"
    first_opts = options(first_group)

    app.layout = html.Div([
        html.Div([
            html.H2("Part III  —  repair over size, placement and shape",
                    style={"margin": "0 0 2px 0", "fontSize": "22px"}),
            html.Div(f"{len(cfgs)} configurations from {Path(bundle['meta']['source']).name}"
                     f"   •   base grey, child coloured by the Isaac Sim check",
                     style={"color": "#666", "fontSize": "13px"}),
        ], style={"padding": "10px 14px 6px 14px"}),

        html.Div([
            html.Div([html.Label("show", style={"fontSize": "12px", "color": "#555"}),
                      dcc.Dropdown(id="group", value=first_group, clearable=False,
                                   options=[{"label": v, "value": k}
                                            for k, v in groups.items()])],
                     style={"width": "310px", "marginRight": "14px"}),
            html.Div([html.Label("configuration", style={"fontSize": "12px", "color": "#555"}),
                      dcc.Dropdown(id="cfg", clearable=False, options=first_opts,
                                   value=first_opts[0]["value"] if first_opts else None)],
                     style={"flex": "1", "marginRight": "14px"}),
            html.Div([html.Label("camera", style={"fontSize": "12px", "color": "#555"}),
                      dcc.Dropdown(id="cam", value="isometric", clearable=False,
                                   options=[{"label": k, "value": k} for k in CAMERAS])],
                     style={"width": "160px"}),
        ], style={"display": "flex", "alignItems": "flex-end", "padding": "0 14px 10px 14px"}),

        html.Div(id="columns", style={"display": "flex", "gap": "10px",
                                      "padding": "0 14px 14px 14px"}),
        html.Div(id="footer", style={"padding": "0 14px 18px 14px",
                                     "color": "#666", "fontSize": "13px"}),
    ], style={"fontFamily": "Segoe UI, Helvetica, Arial, sans-serif",
              "background": "white", "maxWidth": "1500px", "margin": "0 auto"})

    @app.callback(Output("cfg", "options"), Output("cfg", "value"),
                  Input("group", "value"), State("cfg", "value"))
    def _refill(group, current):
        """Refill the configuration list when the group changes, keeping the selection if it fits."""
        opts = options(group)
        if not opts:
            return [], None
        keep = current if any(o["value"] == current for o in opts) else opts[0]["value"]
        return opts, keep

    @app.callback(Output("columns", "children"), Output("footer", "children"),
                  Input("cfg", "value"), Input("cam", "value"))
    def _render(cfg_id, cam_name):
        """Redraw the three columns and the caption for the selected configuration and viewpoint."""
        if cfg_id is None:
            return html.Div("no configuration in this group"), ""
        c = cfgs[cfg_id]
        camera = CAMERAS[cam_name]
        # Taken over the base and all three views, so the range is the same in every column and
        # a part that grew under repair is seen to have grown.
        span = max(max(c["anchor"]["bbox"]),
                   *[max(v["bbox"]) for v in c["views"]]) * 1e3 * 2.6

        cols = []
        for idx, view in enumerate(c["views"]):
            cols.append(html.Div([
                html.Div(view["title"], style={
                    "textAlign": "center", "fontWeight": 600, "fontSize": "15px",
                    "padding": "4px 0 2px 0"}),
                dcc.Graph(figure=make_figure(c, meshes, idx, camera, span),
                          config={"displaylogo": False,
                                  "toImageButtonOptions": {"format": "png", "scale": 3}}),
                metric_block(c, idx, thr),
            ], style={"flex": "1", "border": "1px solid #e4e4e4", "borderRadius": "4px",
                      "paddingBottom": "10px", "background": "white"}))

        drift = c.get("z_drift")
        foot = (f"configuration #{cfg_id}   •   child {c['child_name'].replace('.usd','')}"
                f"   •   base {c['anchor']['name'].replace('.usd','')}")
        if drift is not None:
            foot += f"   •   shape-code drift {drift:.3f}"
        return cols, foot

    return app


def main() -> None:
    """Locate the bundle, load it and serve the page."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None, help="bundle path without extension")
    ap.add_argument("--port", type=int, default=8061)
    args = ap.parse_args()
    if args.data:
        stem = Path(args.data)
        if not stem.is_absolute():
            stem = ROOT / stem
    else:
        found = [c for c in DATA_CANDIDATES if c.with_suffix(".json").exists()]
        if not found:
            raise SystemExit("no bundle found; run tools/export_shape_repair_views.py, "
                             f"or pass --data. Looked in: "
                             f"{', '.join(str(c) for c in DATA_CANDIDATES)}")
        stem = found[0]
    bundle, meshes = load_bundle(stem)
    print(f"{len(bundle['configs'])} configurations, {len(meshes)} meshes "
          f"-> http://127.0.0.1:{args.port}")
    build_app(bundle, meshes).run(debug=False, port=args.port)


if __name__ == "__main__":
    main()
