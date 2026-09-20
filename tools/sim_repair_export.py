"""Write the repaired configurations out so that physics can decide whether they really work.

This is the second and last point at which this repository meets the simulation repository, and
it is what turns the comparison of the branches into a claim about reality rather than about the
surrogate's opinion of itself. Each sampled configuration is repaired twice, once with the shape
held fixed and once with the shape code freed, and both results are written to one file for the
replay to spawn and drop.

The two branches are described differently on purpose. The honest one names the mesh it started
from, since its shape has not changed. The other can name no mesh: its shape exists only as a
latent code, and that code has to be decoded into a mesh before the simulator can be given
anything at all. The simulator never sees a code.

Positions and sizes are absolute and in the same frame as the dataset, so a configuration can be
placed without further interpretation. Each entry also carries this run's own verdict, so that
what the surrogate believed can be set against what physics returns, configuration by
configuration.

Run:
    python -m tools.sim_repair_export --n 100 --out repair_configs.json \
        --txt data/sim_shape_continuum_2027.txt \
        --latents encoder_decoder_model/shape_latents_lv_v14_lam005.pt \
                  encoder_decoder_model/latents_continuum.pth \
        --ckpt gnn_models/sim_gnn_shape_node37_32.pth

Writes:
    one file holding two entries per configuration, one per branch.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.sim_gnn_dataset import load_shape_latents  # noqa: E402
from src.sim_repair_optimizer import load_sim_gnn, repair_pair  # noqa: E402
from tools.sim_repair_demo import scan_rows, to_pair  # noqa: E402

# The largest extent, in metres, that each shape's mesh is authored at. The replay cannot infer
# a scale from a name, so it computes one from the size asked for and the size the mesh has, and
# these are the values it divides by. They have to agree with how the meshes were written; a
# wrong entry here yields a part of the wrong size in an otherwise valid configuration.
BASE_MAX_M = {
    "box": 0.10, "rounded_box": 0.10, "cylinder": 0.10, "half_cylinder": 0.10,
    "sphere": 0.06, "ellipsoid": 0.10, "capsule": 0.10, "cone": 0.10,
    "hex_prism": 0.10, "wedge": 0.10, "i_profile": 0.10, "u_profile": 0.10,
    "l_profile": 0.10, "t_profile": 0.10,
}


def _base_of(mesh_name: str) -> str:
    """The vocabulary shape a mesh derives from, stripping any variant marker and the suffix."""
    return mesh_name.split("__")[0].replace(".usd", "")


def _scale(mesh_name, bbox) -> float:
    """Uniform factor bringing the authored mesh to the largest extent of the target box."""
    return float(max(bbox)) / BASE_MAX_M[_base_of(mesh_name)]


def _obj(mesh, bbox, pos) -> dict:
    """One part as the replay needs it: which mesh, at what scale, and where."""
    return {"mesh": mesh, "scale": _scale(mesh, bbox),
            "bbox": [float(x) for x in bbox], "pos": [float(x) for x in pos]}


def _offline(out) -> dict:
    """This run's own verdict on a repair, to be set against what physics returns for it."""
    return {"geo_ok": bool(out["geo_ok_after"]), "gnn_good": bool(out["gnn_good_after"]),
            "bluff": bool(out["bluff"]), "p_good": float(out["p_good_after"]),
            "overlap_m": float(out["overlap_after_m"]), "thickness_m": float(out["thickness_after_m"])}


def main() -> None:
    """Repair the sampled configurations under both branches and write them out for the replay."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="gnn_models/sim_gnn_shape_node37_32.pth")
    ap.add_argument("--txt", default="data/sim_shape_continuum_2027.txt")
    ap.add_argument("--latents", nargs="+",
                    default=["encoder_decoder_model/shape_latents_lv_v14_lam005.pt"])
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="repair_configs.json")
    args = ap.parse_args()

    latents = load_shape_latents([ROOT / p for p in args.latents])
    model, fail_names, thr = load_sim_gnn(args.ckpt)
    rows, ci, _ = scan_rows(args.txt, latents)
    # The same scan and the same seed as the comparison tool, so this exports the configurations
    # that tool measured, and so the figure tool can recover the starting states afterwards.
    random.Random(args.seed).shuffle(rows)
    rows = rows[:args.n]
    print(f"exporting {len(rows)} pairs x 2 modes | ckpt targets {fail_names}")

    assemblies = []
    for i, r in enumerate(rows):
        pair, child_name = to_pair(r, ci, latents)
        anchor = _obj(r[ci["Obj0_UsdName"]], pair["bbox_anchor"], pair["pos_anchor"])

        out_s = repair_pair(**pair, model=model, thresholds=thr, opt_z=False, opt_scale=True)
        assemblies.append({
            "mode": "scale", "id": i, "obj0": anchor,
            "obj1": _obj(child_name, out_s["bbox_out"], out_s["pos_out"]),
            "offline": _offline(out_s),
        })

        # The same configuration, repaired again with the shape code freed. Both branches start
        # from the identical pair and differ only in that.
        out_z = repair_pair(**pair, model=model, thresholds=thr, opt_z=True, opt_scale=True)
        assemblies.append({
            "mode": "scale_z", "id": i, "obj0": anchor,
            # No mesh, only the code: the shape this repair produced does not exist as a file
            # yet and has to be decoded before the replay can spawn it. The mesh it started
            # from is recorded alongside, for the scale and for comparison.
            "obj1": {"z": [float(v) for v in out_z["z_out"]],
                     "bbox": [float(x) for x in out_z["bbox_out"]],
                     "pos": [float(x) for x in out_z["pos_out"]],
                     "from_mesh": child_name, "z_drift": float(out_z["z_drift"])},
            "offline": _offline(out_z),
        })

    payload = {
        "meta": {"ckpt": args.ckpt, "n": len(rows), "modes": ["scale", "scale_z"],
                 "fail_names": fail_names, "good_thr": float(thr["good"])},
        "assemblies": assemblies,
    }
    out_path = ROOT / args.out if not Path(args.out).is_absolute() else Path(args.out)
    out_path.write_text(json.dumps(payload, indent=1))

    n_scale = sum(a["mode"] == "scale" for a in assemblies)
    n_bluff_s = sum(a["mode"] == "scale" and a["offline"]["bluff"] for a in assemblies)
    n_bluff_z = sum(a["mode"] == "scale_z" and a["offline"]["bluff"] for a in assemblies)
    print(f"wrote {len(assemblies)} assemblies ({n_scale} pairs x 2) -> {out_path}")
    print(f"offline bluff: scale {n_bluff_s}/{n_scale}  scale_z {n_bluff_z}/{n_scale}")


if __name__ == "__main__":
    main()
