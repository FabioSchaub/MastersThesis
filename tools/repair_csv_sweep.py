"""Run the full latent repair over every design CSV and report each joint before and after.

Walks all ``*_ml_input.csv`` files under ``pipeline/`` and puts each through the same chain the
dashboard uses — parse, resolve penetrations, snap faces, then the layered repair — so the
numbers reported here are the ones the pipeline would actually commit. For every connected pair
it prints the contact-axis thickness before and after, the overlap after, what the surrogate
predicted for both, and the feasibility probability.

Two verdicts are printed side by side and they are the point of the script. ``ACTUAL`` is the
analytical check on the repaired geometry; ``GNN`` is what the surrogate believes and what the
pipeline commits on. A pair the surrogate passes while the geometry rejects it is counted as a
bluff, per design and in total. Where ``pipeline/repair_strategies.py`` repairs a single
design, this runs the whole set and gives the aggregate.

Prints only; nothing is written. A design that raises is reported as an error and the sweep
continues, so one broken CSV does not cost the whole run.

Run:
    python tools/repair_csv_sweep.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

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
    THRESH_OVERLAP,
    THRESH_THICKNESS,
    load_models,
    predict_feasibility,
)

THR_TH_MM = THRESH_THICKNESS * 1000.0
THR_OV_MM = THRESH_OVERLAP * 1000.0


def _t(x, device):
    """Turn a block field into a float tensor on the target device."""
    return torch.tensor(np.asarray(x, dtype=np.float32), dtype=torch.float, device=device)


def _bottom_z(b: dict) -> float:
    """Height of the underside of a block, in metres."""
    return float(b["pos"][2] - b["size"][2] / 2.0)


def feasibility(parent: dict, child: dict, gnn_model, device):
    """Query the surrogate with the parent as base and the child as part under test.

    Returns:
        ``(p_feasible, overlap_m, thickness_m)``, the two regressed quantities in metres.
    """
    p, reg = predict_feasibility(
        _t(parent["size"], device), _t(child["size"], device),
        _t(parent["pos"], device), _t(child["pos"], device), gnn_model,
    )
    return float(p), float(reg[0]), float(reg[1])


def process_csv(csv_path: Path, gnn_model, scale_factor, device) -> dict:
    """Repair every design in one CSV and collect one record per connected pair."""
    df = pd.read_csv(csv_path)
    n = detect_n_blocks(df)
    csv_pairs_ok = 0
    csv_pairs_total = 0
    rows_out = []

    for row_idx in range(len(df)):
        row = df.iloc[row_idx]
        blocks_in = parse_blocks(row, n)
        conns = parse_connections(row, n)
        if len(blocks_in) < 2 or not conns:
            continue
        conns_set = {(min(i, j), max(i, j)) for (i, j) in conns}
        resolved = resolve_penetrations(blocks_in, conns_set)
        snapped = apply_snaps(resolved, conns)
        repaired = gnn_repair_layered(
            snapped, conns, gnn_model, scale_factor, device,
            freeze_pos_if_overlap_ok=False,
        )

        for (i, j) in sorted(conns_set):
            if i not in blocks_in or j not in blocks_in:
                continue
            # The connection list is unordered, so the roles are fixed by height: the lower
            # block is the base, the upper one is the part under test and the one thinned.
            par_id, ch_id = (i, j) if _bottom_z(blocks_in[i]) <= _bottom_z(blocks_in[j]) else (j, i)
            p_in, c_in = blocks_in[par_id], blocks_in[ch_id]
            p_rep, c_rep = repaired.get(par_id, p_in), repaired.get(ch_id, c_in)

            # The contact axis is taken from the input geometry and kept for the before and
            # after comparison, so that thickness means the same quantity in both columns.
            face = infer_face(
                p_in["pos"], p_in["size"] / 2.0, c_in["pos"], c_in["size"] / 2.0
            )
            # Recomputed on the repaired geometry only to detect the case where the repair
            # moved the blocks enough for the inferred axis to change; a flipped pair is
            # flagged rather than silently re-measured on a different axis.
            face_rep = infer_face(
                p_rep["pos"], p_rep["size"] / 2.0, c_rep["pos"], c_rep["size"] / 2.0
            )
            axis_flip = (face // 2) != (face_rep // 2)

            th_before = analytical_thickness(c_in["size"] / 2.0, face) * 1000.0
            th_after = analytical_thickness(c_rep["size"] / 2.0, face) * 1000.0

            # The verdict from the geometry itself, which is the ground truth the surrogate is
            # judged against below.
            ov_after_m = analytical_overlap(
                p_rep["pos"], p_rep["size"] / 2.0, c_rep["pos"], c_rep["size"] / 2.0, face
            )
            th_after_m = analytical_thickness(c_rep["size"] / 2.0, face)
            actual_ok, _ = is_repaired_analytical(ov_after_m, th_after_m)

            # The surrogate has to clear both regressed thresholds and the feasibility head;
            # this is the same gate the repair itself commits on.
            p_a, ov_a, tg_a = feasibility(p_rep, c_rep, gnn_model, device)
            gnn_ok = (ov_a >= THRESH_OVERLAP) and (tg_a <= THRESH_THICKNESS) and (p_a >= 0.5)
            bluff = gnn_ok and not actual_ok

            csv_pairs_total += 1
            csv_pairs_ok += int(actual_ok)
            rows_out.append({
                "row": row_idx, "pair": f"{par_id}->{ch_id}", "axis": face // 2,
                "th_before": th_before, "th_after": th_after,
                "ov_after": ov_after_m * 1000.0,
                "gnn_th_after": tg_a * 1000.0, "gnn_ov_after": ov_a * 1000.0,
                "p_after": p_a, "actual_ok": actual_ok, "gnn_ok": gnn_ok,
                "bluff": bluff, "axis_flip": axis_flip,
            })

    return {"name": csv_path.name, "rows": rows_out,
            "ok": csv_pairs_ok, "total": csv_pairs_total}


def main():
    device = torch.device("cpu")
    gnn_model, scale_factor = load_models(device)
    gnn_model = gnn_model.to(device)
    print(f"\nthresholds: overlap >= {THR_OV_MM:.0f} mm, thickness <= {THR_TH_MM:.0f} mm  "
          f"(scale_factor={scale_factor})\n")

    csvs = sorted(
        p for p in ROOT.glob("pipeline/**/*_ml_input.csv")
        if not p.name.endswith("_repaired.csv")
    )
    print(f"Found {len(csvs)} candidate CSVs.\n")

    grand_ok = grand_total = 0
    summary = []
    for csv_path in csvs:
        try:
            res = process_csv(csv_path, gnn_model, scale_factor, device)
        except Exception as e:
            print(f"==== {csv_path.name} ====  ERROR: {type(e).__name__}: {e}\n")
            summary.append((csv_path.name, "ERR", 0, 0))
            continue
        n_bluff = sum(r["bluff"] for r in res["rows"])
        print(f"==== {res['name']} ====  ACTUAL {res['ok']}/{res['total']} feasible"
              f"   (bluffs: {n_bluff})")
        print(f"  {'pair':>8} {'ax':>3} {'th_bef':>7} {'th_aft':>7} {'ov_aft':>7} "
              f"| {'gnnTh':>6} {'gnnOv':>6} {'p':>5} | {'ACTUAL':>7} {'GNN':>5} {'flag':>6}")
        for r in res["rows"]:
            act = "OK" if r["actual_ok"] else "FAIL"
            gnn = "OK" if r["gnn_ok"] else "no"
            flag = "BLUFF" if r["bluff"] else ("flip" if r["axis_flip"] else "")
            print(f"  {r['pair']:>8} {r['axis']:>3} {r['th_before']:>6.1f} {r['th_after']:>6.1f} "
                  f"{r['ov_after']:>6.1f} | {r['gnn_th_after']:>6.1f} {r['gnn_ov_after']:>6.1f} "
                  f"{r['p_after']:>5.2f} | {act:>7} {gnn:>5} {flag:>6}")
        grand_ok += res["ok"]
        grand_total += res["total"]
        summary.append((res["name"], "ok", res["ok"], res["total"], n_bluff))
        print()

    print("=" * 70)
    print("SUMMARY — ACTUAL (analytical) feasibility after repair")
    tot_bluff = 0
    for name, status, ok, total, *rest in summary:
        nb = rest[0] if rest else 0
        tot_bluff += nb
        if status == "ERR":
            print(f"  {name:<48} ERROR")
        else:
            pct = (100.0 * ok / total) if total else 0.0
            bl = f"  [{nb} bluff]" if nb else ""
            print(f"  {name:<48} {ok:>3}/{total:<3}  ({pct:.0f}%){bl}")
    print("-" * 70)
    print(f"  TOTAL: {grand_ok}/{grand_total} pairs ACTUALLY feasible "
          f"({100.0*grand_ok/grand_total if grand_total else 0:.0f}%)   "
          f"GNN-bluffs (GNN says ok, geometry says no): {tot_bluff}")


if __name__ == "__main__":
    main()
