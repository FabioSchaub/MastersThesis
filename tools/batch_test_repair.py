"""Repair every design in ``pipeline/`` and count how many joints actually became feasible.

The count that matters is the closed-form one, before and after: how many joints satisfy both
criteria as measured, not as predicted. The surrogate's own verdict is reported beside it, and
the gap between the two lines is what a repair has to be judged on -- a design where the
surrogate's count rises and the measured count does not has not been repaired.

Failures are caught per design so that one design which cannot be repaired does not end the
run, and the timing is reported because the repair has to be usable inside a pipeline.

    python tools/batch_test_repair.py
    python tools/batch_test_repair.py table letter

Prints a line per design and a total. Nothing is written to disk.
"""

from __future__ import annotations

import io
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipeline.repair_strategies import (  # noqa: E402
    detect_n_blocks,
    parse_blocks,
    parse_connections,
    repair_csv_string,
)
from src.analytical_metrics import (  # noqa: E402
    analytical_overlap,
    analytical_thickness,
    infer_face,
)
from src.repair_optimizer import THRESH_OVERLAP, THRESH_THICKNESS  # noqa: E402
from src.repair_process import load_models, predict_feasibility  # noqa: E402

PIPELINE_DIR = ROOT / "pipeline"


def per_pair_metrics(blocks: dict, conns: list[tuple[int, int]],
                     gnn_model=None, scale_factor: float = 1.0,
                     device: torch.device | None = None) -> list[dict]:
    """Judge every joint of a design, in closed form and optionally with the surrogate.

    Both verdicts are formed on the same geometry, so any difference between them belongs to
    the model. Lengths are reported in millimetres.

    Args:
        blocks: The design, keyed by block index, in metres.
        conns: Joined pairs.
        gnn_model: The surrogate. Omitted to report the closed form alone.
        scale_factor: Factor between the scaled domain and metres.
        device: Device the surrogate runs on.
    """
    rows = []
    for i, j in conns:
        if i not in blocks or j not in blocks:
            continue
        b_i, b_j = blocks[i], blocks[j]
        # The lower block is the parent, as in the repair itself; the two quantities are not
        # symmetric in the pair.
        if b_i["pos"][2] <= b_j["pos"][2]:
            anchor, active = b_i, b_j
        else:
            anchor, active = b_j, b_i

        pos_a = np.asarray(anchor["pos"], dtype=float)
        he_a = np.asarray(anchor["size"], dtype=float) / 2.0
        pos_c = np.asarray(active["pos"], dtype=float)
        he_c = np.asarray(active["size"], dtype=float) / 2.0

        face = infer_face(pos_a, he_a, pos_c, he_c)
        ana_ov = float(analytical_overlap(pos_a, he_a, pos_c, he_c, face))
        ana_th = float(analytical_thickness(he_c, face))
        ana_ok = (ana_ov >= THRESH_OVERLAP) and (ana_th <= THRESH_THICKNESS)

        row = {
            "pair": f"{i}->{j}",
            "ana_ov_mm": ana_ov * 1000,
            "ana_th_mm": ana_th * 1000,
            "ana_ok": ana_ok,
        }

        if gnn_model is not None:
            size_a_s = torch.from_numpy(
                (np.asarray(anchor["size"], dtype=np.float32) * scale_factor)
                .astype(np.float32)
            ).to(device)
            size_b_s = torch.from_numpy(
                (np.asarray(active["size"], dtype=np.float32) * scale_factor)
                .astype(np.float32)
            ).to(device)
            pos_a_s = torch.from_numpy(
                (pos_a.astype(np.float32) * scale_factor).astype(np.float32)
            ).to(device)
            pos_b_s = torch.from_numpy(
                (pos_c.astype(np.float32) * scale_factor).astype(np.float32)
            ).to(device)
            p_bin, reg_m = predict_feasibility(
                size_a_s, size_b_s, pos_a_s, pos_b_s, gnn_model,
            )
            row["p_bin"] = float(p_bin)
            row["gnn_ov_mm"] = float(reg_m[0]) * 1000
            row["gnn_th_mm"] = float(reg_m[1]) * 1000
            row["gnn_ok"] = (
                row["gnn_ov_mm"] >= THRESH_OVERLAP * 1000
                and row["gnn_th_mm"] <= THRESH_THICKNESS * 1000
                and row["p_bin"] >= 0.5
            )
        rows.append(row)
    return rows


def analyze_csv(csv_text: str, gnn_model, scale_factor: float,
                device: torch.device) -> dict:
    """Judge one design and summarise it: how many joints pass by each verdict."""
    df = pd.read_csv(io.StringIO(csv_text))
    row = df.iloc[0]
    n = detect_n_blocks(df)
    blocks = parse_blocks(row, n)
    conns = parse_connections(row, n)
    pair_rows = per_pair_metrics(blocks, conns, gnn_model, scale_factor, device)
    return {
        "n_blocks": n,
        "n_pairs": len(pair_rows),
        "n_ana_ok": sum(1 for r in pair_rows if r["ana_ok"]),
        "n_gnn_ok": sum(1 for r in pair_rows if r.get("gnn_ok", False)),
        "avg_p_bin": float(np.mean([r["p_bin"] for r in pair_rows]))
        if pair_rows else 0.0,
        "pairs": pair_rows,
    }


def run_one(csv_path: Path, gnn_model, scale_factor: float,
            device: torch.device) -> dict:
    """Judge one design, repair it, and judge it again.

    A repair that raises is recorded and the run continues: one design that cannot be handled
    should not cost the results of all the others.
    """
    csv_in = csv_path.read_text()
    pre = analyze_csv(csv_in, gnn_model, scale_factor, device)

    t0 = time.time()
    try:
        csv_out = repair_csv_string(csv_in)
        post = analyze_csv(csv_out, gnn_model, scale_factor, device)
        elapsed = time.time() - t0
        ok = True
        err = None
    except Exception as e:  # noqa: BLE001
        elapsed = time.time() - t0
        post = None
        ok = False
        err = str(e)
        csv_out = None

    return {
        "name": csv_path.stem,
        "pre": pre,
        "post": post,
        "elapsed_s": elapsed,
        "ok": ok,
        "err": err,
        "csv_out": csv_out,
    }


def main() -> int:
    name_filter = [a.lower() for a in sys.argv[1:]]
    csvs = sorted(
        p for p in PIPELINE_DIR.glob("*.csv")
        if "repaired" not in p.stem.lower()
        and "output" not in p.stem.lower()
    )
    if name_filter:
        csvs = [p for p in csvs if any(f in p.stem.lower() for f in name_filter)]
    if not csvs:
        print(f"No matching CSVs in {PIPELINE_DIR}/")
        return 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Loading GNN model...")
    gnn_model, scale_factor = load_models(device)
    print()

    print("=" * 90)
    print(f"  Testing {len(csvs)} design(s)")
    print("=" * 90)

    results: list[dict] = []
    for p in csvs:
        print(f"\n--- {p.stem} ---")
        r = run_one(p, gnn_model, scale_factor, device)
        results.append(r)
        if not r["ok"]:
            print(f"  ERROR: {r['err']}")
            continue
        pre, post = r["pre"], r["post"]
        print(
            f"  blocks={pre['n_blocks']}, pairs={pre['n_pairs']}, "
            f"time={r['elapsed_s']:.1f}s"
        )
        print(
            f"  ANA  ok: {pre['n_ana_ok']}/{pre['n_pairs']} -> "
            f"{post['n_ana_ok']}/{post['n_pairs']}"
        )
        print(
            f"  GNN  ok: {pre['n_gnn_ok']}/{pre['n_pairs']} -> "
            f"{post['n_gnn_ok']}/{post['n_pairs']}   "
            f"(avg p_binary: {pre['avg_p_bin']:.3f} -> {post['avg_p_bin']:.3f})"
        )

    print("\n" + "=" * 90)
    print(f"  SUMMARY")
    print("=" * 90)
    print(
        f"  {'design':<48} {'pairs':>6} "
        f"{'ana':>10} {'gnn':>10} {'p_bin':>14} {'time':>6}"
    )
    for r in results:
        if not r["ok"]:
            print(f"  {r['name']:<48} ERROR: {r['err']}")
            continue
        pre, post = r["pre"], r["post"]
        ana_str = f"{pre['n_ana_ok']}->{post['n_ana_ok']}"
        gnn_str = f"{pre['n_gnn_ok']}->{post['n_gnn_ok']}"
        p_str = f"{pre['avg_p_bin']:.2f}->{post['avg_p_bin']:.2f}"
        print(
            f"  {r['name']:<48} {pre['n_pairs']:>6} "
            f"{ana_str:>10} {gnn_str:>10} {p_str:>14} "
            f"{r['elapsed_s']:>5.1f}s"
        )

    # Totalled over joints rather than averaged over designs, so a design with many joints
    # weighs accordingly and a two-block design does not count as much as a whole assembly.
    ok_results = [r for r in results if r["ok"]]
    if ok_results:
        total_pairs = sum(r["pre"]["n_pairs"] for r in ok_results)
        total_ana_pre = sum(r["pre"]["n_ana_ok"] for r in ok_results)
        total_ana_post = sum(r["post"]["n_ana_ok"] for r in ok_results)
        total_gnn_pre = sum(r["pre"]["n_gnn_ok"] for r in ok_results)
        total_gnn_post = sum(r["post"]["n_gnn_ok"] for r in ok_results)
        print()
        print(
            f"  TOTALS: {total_pairs} pairs across {len(ok_results)} designs"
        )
        print(
            f"  ANA ok: {total_ana_pre}/{total_pairs} "
            f"({total_ana_pre/total_pairs*100:.0f}%) -> "
            f"{total_ana_post}/{total_pairs} "
            f"({total_ana_post/total_pairs*100:.0f}%)"
        )
        print(
            f"  GNN ok: {total_gnn_pre}/{total_pairs} "
            f"({total_gnn_pre/total_pairs*100:.0f}%) -> "
            f"{total_gnn_post}/{total_pairs} "
            f"({total_gnn_post/total_pairs*100:.0f}%)"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
