"""Produce the head-to-head comparison of parameter-space against latent-space repair.

This is the script the central result of Part II is derived from. For every connected block
pair of every design CSV in ``pipeline/new_csv/`` it records four geometry states, as block
centres and full edge lengths in millimetres, and for each state the analytical overlap and
thickness, the analytical verdict, and what the surrogate predicts:

    raw        the untouched geometry of the CSV, as the generative stage produced it
    initial    after the analytical pre-processing only, penetrations resolved and faces
               snapped, which is the state both branches are handed and must agree on
    param      after gradient repair over the three edge lengths
    latent     after gradient repair over the latent code

A pair whose surrogate verdict is positive while the analytical one is not is flagged as a
bluff, which is what makes the two branches comparable on honesty and not only on success rate.

The two branches cannot be imported into one interpreter, so the work is split. ``--stage
dump`` runs the pipeline inside one branch working tree and writes a partial file; it is run
twice, once per branch, over the same CSV directory. ``--stage merge`` then joins the two by
design and pair, checks that the geometry they share really is identical, and writes the
combined file with any disagreement recorded under ``meta.warnings``:

    # latent branch (this working tree)
    python tools/export_repair_comparison.py --stage dump --variant latent \
        --root <repo> --csvdir <repo>/pipeline/new_csv --out /tmp/latent.json

    # parameter branch (separate git worktree)
    python tools/export_repair_comparison.py --stage dump --variant param \
        --root C:/tmpwt2 --csvdir <repo>/pipeline/new_csv --out /tmp/param.json

    python tools/export_repair_comparison.py --stage merge \
        --latent /tmp/latent.json --param /tmp/param.json \
        --out results/dashboard_data/repair_comparison.json

The merged file is self-contained: ``tools/repair_dashboard.py`` renders the three states from
it without loading a checkpoint, and it is kept in the repository so the comparison can be
re-derived without rerunning either branch.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def run_dump(root: Path, csvdir: Path, out: Path, variant: str) -> int:
    """Repair every pair inside one branch working tree and write the partial dump."""
    root = root.resolve()
    csvdir = csvdir.resolve()
    # The branch root is made the working directory and put ahead of the path before anything
    # is imported, so this file can drive a checkout other than the one it lives in.
    os.chdir(root)
    sys.path.insert(0, str(root))

    import numpy as np
    import pandas as pd
    import torch

    from pipeline.repair_strategies import (  # noqa: E402
        apply_snaps,
        detect_n_blocks,
        gnn_repair_layered,
        parse_blocks,
        parse_connections,
        resolve_penetrations,
    )
    from src.analytical_metrics import (  # noqa: E402
        analytical_overlap,
        analytical_thickness,
        infer_face,
        is_repaired_analytical,
    )
    from src.repair_process import (  # noqa: E402
        MODEL_NAME,
        THRESH_OVERLAP,
        THRESH_THICKNESS,
        load_models,
        predict_feasibility,
    )

    device = torch.device("cpu")
    gnn_model, scale_factor = load_models(device)
    gnn_model = gnn_model.to(device)
    print(f"[{variant}] checkpoint  : {MODEL_NAME}")
    print(f"[{variant}] scale_factor: {scale_factor}")
    print(
        f"[{variant}] thresholds  : overlap >= {THRESH_OVERLAP * 1000:.1f} mm, "
        f"thickness <= {THRESH_THICKNESS * 1000:.1f} mm"
    )

    def _t(x):
        """Wrap a sequence as a float tensor on the chosen device."""
        return torch.tensor(
            np.asarray(x, dtype=np.float32), dtype=torch.float, device=device
        )

    def _bottom_z(b) -> float:
        """Height of a block's lower face, used to decide which of a pair is the parent."""
        return float(b["pos"][2] - b["size"][2] / 2.0)

    def geom(b) -> dict:
        """Convert one block's centre and size from metres to millimetres."""
        return {
            "center_mm": [float(v) * 1000.0 for v in b["pos"]],
            "edge_lengths_mm": [float(v) * 1000.0 for v in b["size"]],
        }

    def state_metrics(parent, child, face_ref) -> dict:
        """Analytical and surrogate metrics for one pair in one state."""
        # Recorded for reference only. The overlap and thickness below are measured on
        # ``face_ref``, the face of the raw geometry, not on this one.
        face = infer_face(
            parent["pos"], parent["size"] / 2.0, child["pos"], child["size"] / 2.0
        )
        ov = analytical_overlap(
            parent["pos"],
            parent["size"] / 2.0,
            child["pos"],
            child["size"] / 2.0,
            face_ref,
        )
        th = analytical_thickness(child["size"] / 2.0, face_ref)
        ok, _ = is_repaired_analytical(ov, th)
        p, reg = predict_feasibility(
            _t(parent["size"]), _t(child["size"]), _t(parent["pos"]), _t(child["pos"]),
            gnn_model,
        )
        gnn_ov, gnn_th = float(reg[0]), float(reg[1])
        # The surrogate has to clear both regressed criteria and the feasibility head before the
        # pair counts as passed, which is the same rule the repair stops on.
        gnn_ok = (
            gnn_ov >= THRESH_OVERLAP and gnn_th <= THRESH_THICKNESS and float(p) >= 0.5
        )
        return {
            "contact_face_code": int(face),
            "contact_axis": int(face // 2),
            "overlap_mm": ov * 1000.0,
            "thickness_mm": th * 1000.0,
            "analytical_ok": bool(ok),
            "p_feasible": float(p),
            "gnn_overlap_mm": gnn_ov * 1000.0,
            "gnn_thickness_mm": gnn_th * 1000.0,
            "gnn_ok": bool(gnn_ok),
            "bluff": bool(gnn_ok and not ok),
        }

    csvs = sorted(csvdir.glob("*_ml_input.csv"))
    # Repaired CSVs are outputs of an earlier run and would be counted as designs of their own.
    csvs = [p for p in csvs if not p.name.endswith("_repaired.csv")]
    print(f"[{variant}] CSVs: {[p.name for p in csvs]}")

    designs = []
    for csv_path in csvs:
        df = pd.read_csv(csv_path)
        n = detect_n_blocks(df)
        design = csv_path.name.replace("_ml_input.csv", "")
        for row_idx in range(len(df)):
            row = df.iloc[row_idx]
            blocks_raw = parse_blocks(row, n)
            conns = parse_connections(row, n)
            if len(blocks_raw) < 2 or not conns:
                continue
            conns_set = sorted({(min(i, j), max(i, j)) for (i, j) in conns})

            resolved = resolve_penetrations(blocks_raw, set(conns_set))
            snapped = apply_snaps(resolved, conns)
            # Positions stay free even where the overlap already passes, so both branches get
            # the same set of degrees of freedom and differ only in how shape is parameterised.
            repaired = gnn_repair_layered(
                snapped,
                conns,
                gnn_model,
                scale_factor,
                device,
                freeze_pos_if_overlap_ok=False,
            )

            pairs = []
            for (i, j) in conns_set:
                if i not in blocks_raw or j not in blocks_raw:
                    continue
                # The lower block is the parent, so the child is the part under test and the
                # roles stay the same across all four states even after the geometry moves.
                if _bottom_z(blocks_raw[i]) <= _bottom_z(blocks_raw[j]):
                    par_id, ch_id = i, j
                else:
                    par_id, ch_id = j, i

                p_raw, c_raw = blocks_raw[par_id], blocks_raw[ch_id]
                p_ini, c_ini = snapped.get(par_id, p_raw), snapped.get(ch_id, c_raw)
                p_rep, c_rep = repaired.get(par_id, p_ini), repaired.get(ch_id, c_ini)

                # The contact face is fixed once from the raw geometry and reused for every
                # state. Inferring it per state would let it flip mid-comparison and silently
                # measure the before and the after on different axes.
                face_ref = infer_face(
                    p_raw["pos"], p_raw["size"] / 2.0, c_raw["pos"], c_raw["size"] / 2.0
                )

                pairs.append(
                    {
                        "pair_id": f"{min(i, j)}-{max(i, j)}",
                        "parent": int(par_id),
                        "child": int(ch_id),
                        "contact_face_code": int(face_ref),
                        "contact_axis": int(face_ref // 2),
                        "raw": state_metrics(p_raw, c_raw, face_ref),
                        "initial": state_metrics(p_ini, c_ini, face_ref),
                        # Keyed by the variant, so the merge can put the two dumps side by side
                        # without either overwriting the other.
                        variant: state_metrics(p_rep, c_rep, face_ref),
                    }
                )

            designs.append(
                {
                    "design": design,
                    "csv_file": csv_path.name,
                    "row": int(row_idx),
                    "n_blocks": int(n),
                    "connections": [[int(a), int(b)] for a, b in conns_set],
                    "blocks": {
                        "raw": {str(k): geom(v) for k, v in blocks_raw.items()},
                        "initial": {str(k): geom(v) for k, v in snapped.items()},
                        variant: {str(k): geom(v) for k, v in repaired.items()},
                    },
                    "pairs": pairs,
                }
            )
            print(f"[{variant}]   done {design} row {row_idx}", flush=True)

    payload = {
        "variant": variant,
        "checkpoint": MODEL_NAME,
        "scale_factor": float(scale_factor),
        "thresh_overlap_mm": THRESH_OVERLAP * 1000.0,
        "thresh_thickness_mm": THRESH_THICKNESS * 1000.0,
        "root": str(root),
        "designs": designs,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    n_pairs = sum(len(d["pairs"]) for d in designs)
    print(f"[{variant}] wrote {len(designs)} designs / {n_pairs} pairs -> {out}")
    return 0


# The per-state fields carried into the merged file. Listed explicitly so a dump that gains a
# field does not silently change the shape of the published artefact.
_STATE_KEYS = (
    "contact_face_code",
    "contact_axis",
    "overlap_mm",
    "thickness_mm",
    "analytical_ok",
    "p_feasible",
    "gnn_overlap_mm",
    "gnn_thickness_mm",
    "gnn_ok",
    "bluff",
)


def run_merge(latent_path: Path, param_path: Path, out: Path) -> int:
    """Join the two dumps pair by pair and write the comparison, with consistency warnings."""
    lat = json.loads(latent_path.read_text(encoding="utf-8"))
    par = json.loads(param_path.read_text(encoding="utf-8"))

    par_by_key = {(d["design"], d["row"]): d for d in par["designs"]}

    warnings: list[str] = []
    designs = []
    for dl in lat["designs"]:
        key = (dl["design"], dl["row"])
        dp = par_by_key.get(key)
        if dp is None:
            warnings.append(f"design {key} missing in param dump — skipped")
            continue

        # The pre-processing runs before either optimiser and must therefore land on identical
        # geometry in both branches. This is the check that the comparison rests on a common
        # starting point, so any disagreement is recorded rather than averaged away.
        for bid, g in dl["blocks"]["initial"].items():
            gp = dp["blocks"]["initial"].get(bid)
            if gp is None:
                warnings.append(f"{key} block {bid}: no initial geometry in param dump")
                continue
            for field in ("center_mm", "edge_lengths_mm"):
                d = max(abs(a - b) for a, b in zip(g[field], gp[field]))
                if d > 1e-6:
                    warnings.append(
                        f"{key} block {bid} initial.{field}: "
                        f"param/latent differ by {d:.3e} mm"
                    )

        pairs_p = {p["pair_id"]: p for p in dp["pairs"]}
        pairs = []
        for pl in dl["pairs"]:
            pp = pairs_p.get(pl["pair_id"])
            if pp is None:
                warnings.append(f"{key} pair {pl['pair_id']} missing in param dump")
                continue
            merged = {
                k: pl[k]
                for k in ("pair_id", "parent", "child", "contact_face_code",
                          "contact_axis")
            }
            merged["raw"] = pl["raw"]
            # The two branches read the starting state with different checkpoints, so their
            # surrogate columns legitimately differ there; the measured geometry must not.
            for field in ("overlap_mm", "thickness_mm"):
                d = abs(pl["initial"][field] - pp["initial"][field])
                if d > 1e-6:
                    warnings.append(
                        f"{key} pair {pl['pair_id']} initial.{field}: "
                        f"param/latent differ by {d:.3e} mm"
                    )
            # One shared starting state, taken from the latent dump, plus both branches'
            # feasibility estimates of it kept separately for the same reason.
            merged["initial"] = {k: pl["initial"][k] for k in _STATE_KEYS}
            merged["initial_p_feasible_param"] = pp["initial"]["p_feasible"]
            merged["initial_p_feasible_latent"] = pl["initial"]["p_feasible"]
            merged["param"] = {k: pp["param"][k] for k in _STATE_KEYS}
            merged["latent"] = {k: pl["latent"][k] for k in _STATE_KEYS}
            pairs.append(merged)

        designs.append(
            {
                "design": dl["design"],
                "label": f"{dl['design']}  (row {dl['row']}, "
                         f"{dl['n_blocks']} blocks, {len(pairs)} joints)",
                "csv_file": dl["csv_file"],
                "row": dl["row"],
                "n_blocks": dl["n_blocks"],
                "connections": dl["connections"],
                "blocks": {
                    "raw": dl["blocks"]["raw"],
                    "initial": dl["blocks"]["initial"],
                    "param": dp["blocks"]["param"],
                    "latent": dl["blocks"]["latent"],
                },
                "pairs": pairs,
            }
        )

    payload = {
        "meta": {
            "thresh_overlap_mm": lat["thresh_overlap_mm"],
            "thresh_thickness_mm": lat["thresh_thickness_mm"],
            "states": ["initial", "param", "latent"],
            "state_titles": {
                "initial": "1 · Generated + analytical pre-processing",
                "param": "2 · Repaired in parameter space",
                "latent": "3 · Repaired in latent space",
            },
            "variants": {
                "param": {
                    "branch": "feature/parameter-optimization",
                    "checkpoint": par["checkpoint"],
                    "scale_factor": par["scale_factor"],
                },
                "latent": {
                    "branch": "feature/latentspace-optimization",
                    "checkpoint": lat["checkpoint"],
                    "scale_factor": lat["scale_factor"],
                },
            },
            "warnings": warnings,
        },
        "designs": designs,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    n_pairs = sum(len(d["pairs"]) for d in designs)
    print(f"merged {len(designs)} designs / {n_pairs} pairs -> {out}")
    if warnings:
        print(f"WARNINGS ({len(warnings)}):")
        for w in warnings:
            print("  -", w)
    else:
        print("no consistency warnings")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", choices=("dump", "merge"), required=True)
    ap.add_argument("--variant", choices=("param", "latent"))
    ap.add_argument("--root", help="branch working tree to import from")
    ap.add_argument("--csvdir", help="folder with *_ml_input.csv")
    ap.add_argument("--latent", help="merge: latent dump json")
    ap.add_argument("--param", help="merge: param dump json")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    if args.stage == "dump":
        if not (args.variant and args.root and args.csvdir):
            ap.error("--stage dump needs --variant, --root and --csvdir")
        return run_dump(
            Path(args.root), Path(args.csvdir), Path(args.out).resolve(), args.variant
        )
    if not (args.latent and args.param):
        ap.error("--stage merge needs --latent and --param")
    return run_merge(
        Path(args.latent), Path(args.param), Path(args.out).resolve()
    )


if __name__ == "__main__":
    sys.exit(main())
