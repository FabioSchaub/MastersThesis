"""Compare the ways of repairing a configuration, on the same infeasible configurations.

This is the measurement the central claim of Part III rests on. A sample of configurations the
simulator rejected is repaired four times over, and the four differ in nothing but which
variables the optimiser is allowed to move:

    size                the size and the placement, with the form held fixed.
    size and code       the same, with the shape code freed as well.
    size and code,      the same again, but with the code restricted to the entries the
    restricted          compression left varying.
    shape search        size and placement only, with the form changed, if at all, by choosing
                        another shape of the vocabulary outright.

Three figures are reported per branch. The confidence of the surrogate says how convinced it is;
the feasible rate says how often the geometry agrees; and the difference between them, the rate
at which the surrogate accepts what the geometry rejects, is the quantity the argument turns on.

Only round shapes are excluded from the sample, since the two measured criteria are read off
axis-aligned boxes and would not mean the same thing for them.

What is measured here is still partial: two of the four conditions exist only inside the
simulator. The claim about real feasibility comes from the replay, not from this tool.

Run:
    python -m tools.sim_repair_demo --n 40
    python -m tools.sim_repair_demo --n 100 --shape-search

Writes nothing; the table goes to the terminal.
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.sim_gnn_dataset import load_shape_latents  # noqa: E402
from src.sim_repair_optimizer import (  # noqa: E402
    active_latent_dims, load_sim_gnn, repair_pair, repair_with_shape_search,
)

# Shapes without a flat face to rest or be fastened on. The overlap and the thickness are
# measured on axis-aligned boxes, which for these describes a contact that does not exist.
ROUND = {"sphere", "ellipsoid", "cylinder", "half_cylinder", "capsule", "cone"}
DEFAULT_CKPT = "gnn_models/sim_gnn_shape_node37_32.pth"
DEFAULT_TXT = "data/sim_shape_dataset_1613.txt"
DEFAULT_LAT = "encoder_decoder_model/shape_latents_lv_v14_lam005.pt"


def scan_rows(txt, latents):
    """Read the dataset once and collect what the repair needs from it.

    Also used by the export and the figure tool, which is what lets all three reproduce the same
    sample of configurations from the same seed.

    Returns:
        The configurations the simulator rejected whose part under test has a flat face, the
        column positions of the file, and a representative bounding box per shape taken as the
        median over every occurrence of that shape in the file. The last is what a candidate
        shape is started from in the shape search, since it has no size of its own there.
    """
    infeasible, sizes = [], {}
    with open(ROOT / txt) as f:
        header = f.readline().split()
        ci = {c: i for i, c in enumerate(header)}
        for line in f:
            r = line.split()
            # Sizes are collected from every row and from both roles, feasible ones included:
            # the aim is a size that is typical for the shape, not one tied to this sample.
            for pfx in ("Obj0", "Obj1"):
                nm = r[ci[f"{pfx}_UsdName"]]
                if nm in latents:
                    sizes.setdefault(nm, []).append(
                        [float(r[ci[f"{pfx}_Size{a}"]]) for a in "XYZ"])
            if r[ci["Assembly_Good?"]] not in ("0", "0.0"):
                continue
            n0, n1 = r[ci["Obj0_UsdName"]], r[ci["Obj1_UsdName"]]
            if n0 not in latents or n1 not in latents:
                continue
            if n1.replace(".usd", "") in ROUND:
                continue
            infeasible.append(r)
    med = {k: np.median(np.array(v), axis=0) for k, v in sizes.items()}
    return infeasible, ci, med


def to_pair(r, ci, latents):
    """Turn one row into the arguments the repair takes, and the name of the part under test."""
    def vec(pfx, comp):
        """One three-component quantity of one object, in metres."""
        return np.array([float(r[ci[f"{pfx}_{comp}{a}"]]) for a in "XYZ"])
    pair = dict(
        z_anchor=latents[r[ci["Obj0_UsdName"]]].numpy(),
        bbox_anchor=vec("Obj0", "Size"), pos_anchor=vec("Obj0", "Pos"),
        z_active0=latents[r[ci["Obj1_UsdName"]]].numpy(),
        bbox_active0=vec("Obj1", "Size"), pos_active0=vec("Obj1", "Pos"),
    )
    return pair, r[ci["Obj1_UsdName"]]


def run_grad_mode(pairs, model, thr, opt_z, active_dims=None):
    """Repair every configuration by gradient and average the outcomes.

    Only ``opt_z`` and ``active_dims`` differ between the branches; everything else is left at
    the defaults of the repair, so the comparison is between the free variables alone.
    """
    pb, pa, geo, bluff, zd = [], [], [], [], []
    for pr, _ in pairs:
        out = repair_pair(**pr, model=model, thresholds=thr, opt_z=opt_z,
                          active_dims=active_dims, opt_scale=True)
        pb.append(out["p_good_before"]); pa.append(out["p_good_after"])
        geo.append(out["geo_ok_after"]); bluff.append(out["bluff"]); zd.append(out["z_drift"])
    return dict(p_before=np.mean(pb), p_after=np.mean(pa),
                geo_rate=np.mean(geo), bluff_rate=np.mean(bluff), z_drift=np.mean(zd))


def run_shape_search(pairs, model, thr, latents, med, candidates):
    """Repair every configuration by size, placement and a discrete choice of shape.

    Reports how often the result was accepted and how often that required substituting a
    different shape, which is the honest counterpart of the drift in the shape code.
    """
    feas, changed = [], []
    for pr, shp in pairs:
        out = repair_with_shape_search(pr, shp, model, thr, latents, med, candidates)
        feas.append(out["feasible"]); changed.append(out["changed"])
    return dict(feas_rate=np.mean(feas), change_rate=np.mean(changed))


def main():
    """Sample infeasible configurations and repair each of them under every branch."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=DEFAULT_CKPT)
    ap.add_argument("--txt", default=DEFAULT_TXT)
    ap.add_argument("--latents", nargs="+", default=[DEFAULT_LAT],
                    help="one or more {name:z} tables, merged (base-14 + z-continuum)")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--shape-search", action="store_true",
                    help="also run Path-1 discrete shape-search (base-shape candidates only)")
    args = ap.parse_args()

    latents = load_shape_latents([ROOT / p for p in args.latents])
    model, fail_names, thr = load_sim_gnn(args.ckpt)
    print(f"ckpt targets: {fail_names} | good thr {thr['good']:.3f}")

    rows, ci, med = scan_rows(args.txt, latents)
    # Seeded, so that every branch sees the same configurations and the export and the figure
    # tool can recover exactly this sample later.
    random.Random(args.seed).shuffle(rows)
    pairs = [to_pair(r, ci, latents) for r in rows[:args.n]]
    # Only the base shapes are offered as substitutes, not the perturbed variants of them: the
    # discrete branch is meant to choose between shapes a catalogue would contain.
    candidates = [k for k in latents if "__j" not in k and k.replace(".usd", "") not in ROUND]
    active = active_latent_dims(latents)
    print(f"repairing {len(pairs)} infeasible (flat-active) assemblies | "
          f"{len(candidates)} flat candidate shapes | "
          f"{len(active)}/{len(active) and latents[next(iter(latents))].numel()} LV-active dims\n")

    print(f"{'mode':<16} {'p_before':>9} {'p_after':>9} {'feasible':>9} {'bluff':>7} {'z_drift':>8} {'changed':>8}")
    modes = [("scale", False, None),
             ("scale+z", True, None),
             ("scale+z(active)", True, active)]
    for name, oz, ad in modes:
        m = run_grad_mode(pairs, model, thr, opt_z=oz, active_dims=ad)
        print(f"{name:<16} {m['p_before']:>9.3f} {m['p_after']:>9.3f} "
              f"{m['geo_rate']*100:>8.1f}% {m['bluff_rate']*100:>6.1f}% {m['z_drift']:>8.3f} {'':>8}")
    if args.shape_search:
        s = run_shape_search(pairs, model, thr, latents, med, candidates)
        print(f"{'shape-search':<16} {'':>9} {'':>9} {s['feas_rate']*100:>8.1f}% "
              f"{'':>7} {'':>8} {s['change_rate']*100:>7.1f}%")

    print("\nfeasible = analytic overlap+thickness AND GNN-good surrogate; "
          "tipping & Assembly_Good need Isaac for real verification.")


if __name__ == "__main__":
    main()
