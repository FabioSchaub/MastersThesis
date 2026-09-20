"""Pre-compute the geometry behind the Part III repair figure.

Reads the export of `tools/sim_repair_export.py` (the measurement run) and turns
every configuration into three ready-to-plot states:

    0  start                     the sampled infeasible pair, before any repair
    1  size and placement        the honest branch (shape frozen)
    2  size, placement and code  the shape branch (the one that bluffs)

The start state is not stored in the export, but it is reproducible: the export
samples the same rows with `random.Random(seed).shuffle` over the infeasible
flat-active rows of the dataset, so re-running that scan with the same seed
recovers the pair each id came from.

Shapes are decoded from their latent codes with the frozen shape autoencoder and
turned into triangle meshes by marching cubes, so no mesh files are needed. The
canonical mesh is scaled uniformly onto the bounding box the repair produced,
which is how the simulator spawns it as well.

Output is one bundle that the dashboard reads:

    <out>.json   per-configuration verdicts, measurements and mesh keys
    <out>.npz    the vertex and face arrays, one entry per distinct shape

Run:
    python tools/export_shape_repair_views.py \
        --export C:/Users/Fabio/repair_configs.json \
        --txt    data/sim_shape_continuum_2027.txt \
        --latents encoder_decoder_model/shape_latents_lv_v14_lam005.pt \
                  data_and_latents/latents_continuum.pth \
        --ae-ckpt encoder_decoder_model/best_encoder_decoder_general_canon_lv_v14_lam005_latentdim32_active9.pth \
        --out results/dashboard_data/shape_repair_views
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from skimage import measure

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.dec_sdf import sdf_decoder_from_ckpt  # noqa: E402
from src.sim_gnn_dataset import load_shape_latents  # noqa: E402
from tools.sim_repair_demo import scan_rows, to_pair  # noqa: E402

# The two criteria, in metres, stated here so the bundle is self-describing: the viewer reads
# them from it and needs neither the configuration nor a checkpoint.
THRESH_OVERLAP_M = 0.010
THRESH_THICKNESS_M = 0.020


def decode_mesh(dec, z, res: int = 64, lim: float = 0.75, chunk: int = 200_000):
    """Turn a latent code into a triangle mesh in the canonical frame, centred on its own box.

    Args:
        lim: Half span of the grid the field is sampled on, in the canonical frame.
        chunk: Query points per forward pass; the whole grid at once would not fit.
    """
    lin = np.linspace(-lim, lim, res, dtype=np.float32)
    gx, gy, gz = np.meshgrid(lin, lin, lin, indexing="ij")
    pts = torch.from_numpy(np.stack([gx, gy, gz], -1).reshape(-1, 3))
    zt = torch.as_tensor(np.asarray(z, dtype=np.float32)).reshape(-1)
    with torch.no_grad():
        vals = []
        for i in range(0, pts.shape[0], chunk):
            c = pts[i:i + chunk]
            vals.append(dec(zt.unsqueeze(0).expand(c.shape[0], -1), c).reshape(-1))
        field = torch.cat(vals).numpy().reshape(res, res, res)
    verts, faces, _, _ = measure.marching_cubes(field, level=0.0,
                                                spacing=(lin[1] - lin[0],) * 3)
    verts = verts.astype(np.float32) + lin[0]
    centre = 0.5 * (verts.min(0) + verts.max(0))
    return verts - centre, faces.astype(np.int32)


def place(verts: np.ndarray, bbox, pos) -> np.ndarray:
    """Scale a canonical mesh uniformly onto a bounding box and move it to a position.

    Uniform and matched on the largest extent, which is how the simulator scales what it spawns,
    so what is drawn is what would be replayed.
    """
    extent = verts.max(0) - verts.min(0)
    s = float(np.max(bbox)) / float(np.max(extent))
    return verts * s + np.asarray(pos, dtype=np.float32)


def overlap_thickness(bbox_a, pos_a, bbox_b, pos_b) -> tuple[float, float]:
    """The two criteria on the axis-aligned bounding boxes, in metres.

    The same computation the repair performs, repeated here for the starting state, which the
    export does not record.
    """
    bbox_a, pos_a, bbox_b, pos_b = (np.asarray(v, float)
                                    for v in (bbox_a, pos_a, bbox_b, pos_b))
    ov = []
    for k in (0, 1):
        lo = max(pos_a[k] - bbox_a[k] / 2, pos_b[k] - bbox_b[k] / 2)
        hi = min(pos_a[k] + bbox_a[k] / 2, pos_b[k] + bbox_b[k] / 2)
        ov.append(max(0.0, hi - lo))
    return float(min(ov)), float(bbox_b[2])


def main() -> None:
    """Decode every configuration once and write the bundle the viewer reads."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", required=True, help="repair_configs.json of the run")
    ap.add_argument("--txt", default="data/sim_shape_continuum_2027.txt")
    ap.add_argument("--latents", nargs="+", required=True)
    ap.add_argument("--ae-ckpt", required=True)
    ap.add_argument("--seed", type=int, default=0, help="must match the export run")
    ap.add_argument("--res", type=int, default=64, help="marching-cubes grid")
    ap.add_argument("--out", default="results/dashboard_data/shape_repair_views")
    args = ap.parse_args()

    def _p(p):
        """Resolve a path against the repository root unless it is already absolute."""
        p = Path(p)
        return p if p.is_absolute() else ROOT / p

    payload = json.loads(_p(args.export).read_text())
    assemblies = payload["assemblies"]
    by_id: dict[int, dict] = {}
    for a in assemblies:
        by_id.setdefault(a["id"], {})[a["mode"]] = a
    n = len(by_id)
    print(f"export: {len(assemblies)} assemblies, {n} configurations, "
          f"ckpt {payload['meta'].get('ckpt')}")

    latents = load_shape_latents([_p(p) for p in args.latents])
    rows, ci, _ = scan_rows(_p(args.txt), latents)
    random.Random(args.seed).shuffle(rows)
    rows = rows[:n]
    print(f"dataset: {len(rows)} start pairs recovered with seed {args.seed}")

    dec = sdf_decoder_from_ckpt(
        torch.load(_p(args.ae_ckpt), map_location="cpu", weights_only=False), device="cpu")
    dec.eval()

    meshes: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    def mesh_key(key: str, z) -> str:
        """Decode a shape once and refer to it by key thereafter.

        The base shapes recur across configurations, and decoding is the expensive step here.
        """
        if key not in meshes:
            meshes[key] = decode_mesh(dec, z, res=args.res)
        return key

    configs = []
    mismatched = 0
    for i, r in enumerate(rows):
        pair, child_name = to_pair(r, ci, latents)
        rec = by_id[i]
        scale, scale_z = rec["scale"], rec["scale_z"]

        # The recovered pair must be the one the export describes. A different seed or a
        # different dataset would silently pair a repair with the wrong starting state, and the
        # figure would then compare two unrelated configurations.
        if scale_z["obj1"].get("from_mesh") != child_name or not np.allclose(
                scale["obj0"]["bbox"], pair["bbox_anchor"], atol=1e-6):
            mismatched += 1
            continue

        anchor_name = r[ci["Obj0_UsdName"]]
        k_anchor = mesh_key(f"shape::{anchor_name}", latents[anchor_name].numpy())
        k_child = mesh_key(f"shape::{child_name}", latents[child_name].numpy())
        k_repair = mesh_key(f"code::{i}", scale_z["obj1"]["z"])

        ov0, th0 = overlap_thickness(pair["bbox_anchor"], pair["pos_anchor"],
                                     pair["bbox_active0"], pair["pos_active0"])
        views = [
            {"title": "Start", "mesh": k_child, "bbox": list(map(float, pair["bbox_active0"])),
             "pos": list(map(float, pair["pos_active0"])),
             "overlap_mm": ov0 * 1e3, "thickness_mm": th0 * 1e3,
             "p_good": None, "geo_ok": ov0 >= THRESH_OVERLAP_M and th0 <= THRESH_THICKNESS_M,
             "bluff": False},
        ]
        for title, a, key in (("Size and placement", scale, k_child),
                              ("Size, placement and shape code", scale_z, k_repair)):
            off = a["offline"]
            views.append({
                "title": title, "mesh": key,
                "bbox": [float(x) for x in a["obj1"]["bbox"]],
                "pos": [float(x) for x in a["obj1"]["pos"]],
                "overlap_mm": off["overlap_m"] * 1e3,
                "thickness_mm": off["thickness_m"] * 1e3,
                "p_good": off["p_good"], "geo_ok": off["geo_ok"], "bluff": off["bluff"],
            })

        configs.append({
            "id": i,
            "anchor": {"mesh": k_anchor, "name": anchor_name,
                       "bbox": list(map(float, pair["bbox_anchor"])),
                       "pos": list(map(float, pair["pos_anchor"]))},
            "child_name": child_name,
            "z_drift": scale_z["obj1"].get("z_drift"),
            "views": views,
        })

    if mismatched:
        print(f"WARNING: {mismatched} configurations did not match the export and were skipped")

    # The meshes are stored in the canonical frame, not placed: the viewer scales and moves them
    # itself, so a configuration costs one copy of each distinct shape and not one per view.
    verts_out, faces_out = {}, {}
    for k, (v, f) in meshes.items():
        verts_out[k] = v
        faces_out[k] = f

    out = _p(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out.with_suffix(".npz"),
                        **{f"v::{k}": v for k, v in verts_out.items()},
                        **{f"f::{k}": f for k, f in faces_out.items()})
    out.with_suffix(".json").write_text(json.dumps({
        "meta": {"source": str(_p(args.export)), "n": len(configs), "seed": args.seed,
                 "res": args.res, "thresh_overlap_mm": THRESH_OVERLAP_M * 1e3,
                 "thresh_thickness_mm": THRESH_THICKNESS_M * 1e3,
                 "export_meta": payload["meta"]},
        "configs": configs,
    }, indent=1))
    print(f"wrote {len(configs)} configurations and {len(meshes)} meshes -> "
          f"{out.with_suffix('.json')} / {out.with_suffix('.npz')}")


if __name__ == "__main__":
    main()
