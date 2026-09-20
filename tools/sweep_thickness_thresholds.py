"""Ask what happens to the repair when the thickness target is tightened below the real one.

Runs the full repair on a fixed pair of designs several times, each with a different value of
``THRESH_THICKNESS`` in the repair optimiser, and reports per joint the thickness the geometry
actually has next to the thickness the surrogate predicts, with the overlap and the feasibility
probability. A pre-repair table for the same joints is printed first as the reference.

The question behind it is the gap between the two: the repair stops when the surrogate is
satisfied, so if the surrogate reads a joint as thinner than it is, aiming the repair below the
real threshold is a way to buy back the difference. The last column flags joints whose true
thickness is still over the threshold.

Prints only; nothing is written.

Run:
    python tools/sweep_thickness_thresholds.py
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

# In metres. The last entry is config.gnn.thresh_thickness_max, the threshold that actually
# has to be met; the two below it are the deliberately over-tightened targets.
THRESHOLDS_M = [0.014, 0.017, 0.020]

CSV_DIR = ROOT / "pipeline" / "new_csv"
DESIGNS = [
    # "cactus_candidate_4_ml_input.csv",
    "castle_candidate_4_ml_input.csv",
    "letter_t_candidate_2_ml_input.csv",
]


def _import_pipeline():
    """Collect the repair entry points, imported here rather than at module scope.

    The threshold is overridden on the already-imported ``src.repair_optimizer`` before each
    sweep, so the import order matters: importing the pipeline lazily keeps that override the
    last word on the value the repair reads.
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
    """Per connected pair, the geometric thickness in millimetres next to the surrogate's."""
    rows: list[dict] = []
    for i, j in conns:
        if i not in blocks or j not in blocks:
            continue
        b_i, b_j = blocks[i], blocks[j]

        # The connection list is unordered; fixing the roles by height keeps the same joint on
        # the same table row across all thresholds, which is what makes the columns comparable.
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
    """Repair the first design of one CSV at a given thickness target and measure the result."""
    # Rebinding the module attribute is what makes the sweep possible at all: the threshold is
    # a module-level constant of the repair, not an argument it can be called with.
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

        # The input geometry as parsed, with no repair and not even the penetration and snap
        # pre-processing, so the sweep rows below can be read as changes against it.
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
                # Compared against the real threshold in millimetres, not against the swept
                # target: the point is whether the tighter aim moved the true thickness under
                # the constraint that has to hold.
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
