"""Ask how far below the nominal limit the repair has to aim for the joint to really be under it.

The repair drives the surrogate's thickness under a target, but what has to hold is the
measured thickness. Where the two differ, aiming at the nominal limit leaves joints just above
it. This runs the whole repair several times on the same designs with nothing changed but that
target, and reports both thicknesses per joint afterwards, so the offset can be read off
rather than guessed.

    python tools/sweep_thickness_thresholds.py

Prints one table per design and target. Nothing is written to disk; what comes out of it is a
value for ``REPAIR_TARGET_THICKNESS_M`` in ``pipeline/repair_strategies.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import src.repair_optimizer as _repair_optimizer  # noqa: E402

# Targets in metres, from the nominal limit downwards. Only the target the repair aims at
# changes between runs; the limit a joint is judged against does not.
THRESHOLDS_M = [0.014, 0.017, 0.020]

# Designs the sweep is run on, chosen because their joints sit near the thickness limit, which
# is where the offset between predicted and measured thickness actually matters.
CSV_DIR = ROOT / "pipeline" / "new_csv"
DESIGNS = [
    # "cactus_candidate_4_ml_input.csv",
    "castle_candidate_4_ml_input.csv",
    "letter_t_candidate_2_ml_input.csv",
]


def _import_pipeline():
    """Import the repair, and return the pieces of it this script drives.

    Imported inside a function on purpose: ``pipeline.repair_strategies`` writes a fixed
    thickness target onto the optimiser module when it loads, and that has to happen before
    the sweep starts overwriting it, not in the middle.
    """
    from pipeline.repair_strategies import (
        apply_snaps,
        detect_n_blocks,
        gnn_repair_layered,
        parse_blocks,
        parse_connections,
        resolve_penetrations,
    )
    from src.analytical_metrics import infer_face
    from src.repair_process import load_models, predict_feasibility

    return {
        "apply_snaps": apply_snaps,
        "detect_n_blocks": detect_n_blocks,
        "gnn_repair_layered": gnn_repair_layered,
        "parse_blocks": parse_blocks,
        "parse_connections": parse_connections,
        "resolve_penetrations": resolve_penetrations,
        "infer_face": infer_face,
        "load_models": load_models,
        "predict_feasibility": predict_feasibility,
    }


def _pair_diagnostics(
    blocks: dict[int, dict],
    conns: list[tuple[int, int]],
    gnn_model,
    scale_factor: float,
    infer_face,
    predict_feasibility,
) -> list[dict]:
    """Measure and predict the thickness of every joint of a repaired design, in millimetres.

    The overlap and the feasibility probability come along because a thickness target that is
    aimed too low buys thickness at the cost of the other criterion, and that has to be
    visible in the same table.
    """
    rows: list[dict] = []
    for i, j in conns:
        if i not in blocks or j not in blocks:
            continue
        b_i, b_j = blocks[i], blocks[j]

        # The lower block is the parent, as in the repair itself; thickness is measured on
        # the child, so the order decides which block is measured.
        if b_i["pos"][2] <= b_j["pos"][2]:
            parent_id, child_id = i, j
            p, c = b_i, b_j
        else:
            parent_id, child_id = j, i
            p, c = b_j, b_i

        pos_p_m = np.asarray(p["pos"], dtype=float)
        size_p_m = np.asarray(p["size"], dtype=float)
        pos_c_m = np.asarray(c["pos"], dtype=float)
        size_c_m = np.asarray(c["size"], dtype=float)
        he_p_m = size_p_m / 2.0
        he_c_m = size_c_m / 2.0

        face = infer_face(pos_p_m, he_p_m, pos_c_m, he_c_m)
        contact_ax = face // 2
        ana_th_mm = 1000.0 * float(size_c_m[contact_ax])

        sf = float(scale_factor)
        size_anchor = torch.from_numpy((size_p_m * sf).astype(np.float32))
        size_active = torch.from_numpy((size_c_m * sf).astype(np.float32))
        pos_anchor = torch.from_numpy((pos_p_m * sf).astype(np.float32))
        pos_active = torch.from_numpy((pos_c_m * sf).astype(np.float32))
        p_bin, reg_m = predict_feasibility(
            size_anchor,
            size_active,
            pos_anchor,
            pos_active,
            gnn_model,
        )
        gnn_th_mm = 1000.0 * float(reg_m[1])
        gnn_ov_mm = 1000.0 * float(reg_m[0])

        rows.append(
            {
                "parent": parent_id,
                "child": child_id,
                "axis": "xyz"[contact_ax],
                "ana_th_mm": ana_th_mm,
                "gnn_th_mm": gnn_th_mm,
                "gnn_ov_mm": gnn_ov_mm,
                "p_bin": p_bin,
            }
        )
    return rows


def _run_design_at_threshold(
    csv_path: Path,
    threshold_m: float,
    api: dict,
    gnn_model,
    scale_factor: float,
    device: torch.device,
) -> list[dict]:
    """Run the whole repair on one design with one thickness target, and judge the result.

    The target is written onto the optimiser module, which is where the hinge, the early stop
    and the commit gate read it from. Everything else is left exactly as it is, so the runs
    differ in one thing only.
    """
    _repair_optimizer.THRESH_THICKNESS = float(threshold_m)

    df = pd.read_csv(csv_path)
    row = df.iloc[0]
    n = api["detect_n_blocks"](df)
    blocks = api["parse_blocks"](row, n)
    conns = api["parse_connections"](row, n)
    conns_set = {(min(a, b), max(a, b)) for (a, b) in conns}

    resolved = api["resolve_penetrations"](blocks, conns_set)
    snapped = api["apply_snaps"](resolved, conns)
    repaired = api["gnn_repair_layered"](
        snapped,
        conns,
        gnn_model,
        scale_factor,
        device,
        freeze_pos_if_overlap_ok=True,
    )

    return _pair_diagnostics(
        repaired,
        conns,
        gnn_model,
        scale_factor,
        api["infer_face"],
        api["predict_feasibility"],
    )


def main() -> int:
    api = _import_pipeline()

    device = torch.device("cpu")
    gnn_model, scale_factor = api["load_models"](device)
    gnn_model.eval()

    print(f"\nScale factor: {scale_factor}")
    print(f"Thresholds (mm): {[int(t*1000) for t in THRESHOLDS_M]}\n")

    for csv_name in DESIGNS:
        csv_path = CSV_DIR / csv_name
        if not csv_path.exists():
            print(f"SKIP: {csv_path} not found")
            continue

        # The design as it arrives, so every repaired result can be read against a common
        # starting point rather than only against the other targets.
        df0 = pd.read_csv(csv_path)
        row0 = df0.iloc[0]
        n0 = api["detect_n_blocks"](df0)
        blocks0 = api["parse_blocks"](row0, n0)
        conns0 = api["parse_connections"](row0, n0)
        pre_rows = _pair_diagnostics(
            blocks0,
            conns0,
            gnn_model,
            scale_factor,
            api["infer_face"],
            api["predict_feasibility"],
        )

        print("=" * 90)
        print(f"DESIGN: {csv_name}")
        print("=" * 90)
        print("\nPre-repair (input geometry):")
        print(
            f"  {'pair':<10} {'ax':<3} {'ana_th_mm':>11} {'gnn_th_mm':>11} "
            f"{'gnn_ov_mm':>11} {'p_bin':>7}"
        )
        for r in pre_rows:
            print(
                f"  {r['parent']}-{r['child']:<8} {r['axis']:<3} "
                f"{r['ana_th_mm']:>11.3f} {r['gnn_th_mm']:>11.3f} "
                f"{r['gnn_ov_mm']:>11.3f} {r['p_bin']:>7.3f}"
            )

        for t in THRESHOLDS_M:
            print(f"\n--- THRESH_THICKNESS = {int(t*1000)} mm ---")
            rows = _run_design_at_threshold(
                csv_path,
                t,
                api,
                gnn_model,
                scale_factor,
                device,
            )
            print(
                f"  {'pair':<10} {'ax':<3} {'ana_th_mm':>11} {'gnn_th_mm':>11} "
                f"{'gnn_ov_mm':>11} {'p_bin':>7} {'over20?':>9}"
            )
            for r in rows:
                over = "YES" if r["ana_th_mm"] > 20.0 else "."
                print(
                    f"  {r['parent']}-{r['child']:<8} {r['axis']:<3} "
                    f"{r['ana_th_mm']:>11.3f} {r['gnn_th_mm']:>11.3f} "
                    f"{r['gnn_ov_mm']:>11.3f} {r['p_bin']:>7.3f} "
                    f"{over:>9}"
                )

        print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
