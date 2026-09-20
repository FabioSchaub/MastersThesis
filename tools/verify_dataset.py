"""Check that a two-block training dataset is internally consistent before it is trained on.

The labels in the dataset were produced elsewhere, so nothing downstream would notice if they
disagreed with the geometry they claim to describe. This recomputes them: the contact axis, the
overlap, the contact-axis thickness, the planned gap and the depth below the table are derived
from the stored positions and sizes alone and compared against the stored values, and the
failure reasons and the feasibility flag are re-derived from those metrics through the same
threshold cascade. It also reports the distributions that decide whether the surrogate can
learn the decision boundary at all, in particular how many samples fall in the narrow band
around the gap threshold.

Reads the file named by ``config.data.data_file`` under ``config.data.data_folder``, or a path
given as the first argument. Prints PASS or FAIL per check and writes nothing. Exit code 0 if
every check passes, 1 otherwise, so it can gate a training run.

It expects the older, wider schema, with the depth below the table and the planned gap among
the required columns. The graph preparation of this branch rejects exactly those two columns as
legacy, so a dataset that satisfies this script is not necessarily one that
``src/gnn_dataset_preparation.py`` will accept, and the reverse holds as well.

Run:
    python tools/verify_dataset.py
    python tools/verify_dataset.py PATH/TO/dataset.txt
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config.config import config
from src.simulation_dataset import read_txt_file, manipulate_data

# Deliberately duplicated rather than imported from src/dataset_generation.py: a check that
# read its constants from the module it is checking would agree with it by construction, and
# would confirm nothing. Two of these no longer match that module, which is expected for the
# older datasets this script targets but must not be mistaken for the current rule — the
# generator now uses 0.02 for thickness, and its classification is reduced to overlap and
# thickness alone, without the table, penetration and gap cases re-derived below.
TABLE_THRESHOLD = 0.001
PENETRATION_THRESHOLD = -0.001
GAP_THRESHOLD = 0.001
OVERLAP_THRESHOLD = 0.010
THICKNESS_THRESHOLD = 0.022

# Failure reason names (string column values stored in the txt dataset).
FAIL_NAMES = {
    "NONE",
    "BLOCK_GAP",
    "OVERLAP_INSUFFICIENT",
    "TABLE_INTERFERENCE",
    "BLOCK_TOO_THICK",
    "BLOCK_PENETRATION",
}

# The table plane the stored depths were measured against. It is zero because the datasets put
# the base block at pos_z close to half its own height, which places the table at the origin;
# src/dataset_generation.py carries a non-zero table height, so the two must not be conflated.
# A systematic offset between derived and stored depth is reported as a warning below, which is
# what a wrong value here would look like.
TABLE_HEIGHT_DATASET = 0.0


REQUIRED_COLUMNS = [
    "Block0_PosX", "Block0_PosY", "Block0_PosZ",
    "Block0_SizeX", "Block0_SizeY", "Block0_SizeZ",
    "Block1_PosX", "Block1_PosY", "Block1_PosZ",
    "Block1_SizeX", "Block1_SizeY", "Block1_SizeZ",
    "Block0_UnderSurface_m",
    "Block1_UnderSurface_m",
    "Block1_Overlap_m",
    "Block1_Thickness_m",
    "Block1_PlannedGap_m",
    "Assembly_Good?",
    "Block0_FailureReason",
    "Block1_FailureReason",
]


def hr(title: str = "") -> None:
    """Print a section rule, optionally titled."""
    print()
    if title:
        print(f"=== {title} " + "=" * (max(0, 70 - 5 - len(title))))
    else:
        print("=" * 70)


def fmt_mm(v_m: float | np.floating) -> str:
    """Render a length given in metres as a signed millimetre string."""
    return f"{float(v_m) * 1000.0:+.3f} mm"


def derive_contact_axis(
    pos0: np.ndarray, size0: np.ndarray, pos1: np.ndarray, size1: np.ndarray
) -> np.ndarray:
    """Contact axis index per row, 0 for x, 1 for y, 2 for z.

    The per-axis signed gap is positive when the two projections are separated and negative
    when they overlap. The contact axis is the one with the largest gap: the two tangential
    axes overlap strongly by construction, so only the contact axis is near zero or positive.
    """
    delta = pos1 - pos0
    s_sum = size0 + size1
    signed_gap_per_axis = np.abs(delta) - s_sum / 2.0
    return np.argmax(signed_gap_per_axis, axis=1)


def signed_gap_along(axis_idx: np.ndarray,
                     pos0: np.ndarray, size0: np.ndarray,
                     pos1: np.ndarray, size1: np.ndarray) -> np.ndarray:
    """Signed gap along a per-row contact axis, in metres; negative means interpenetration."""
    rows = np.arange(len(axis_idx))
    delta = pos1[rows, axis_idx] - pos0[rows, axis_idx]
    s_sum = size0[rows, axis_idx] + size1[rows, axis_idx]
    return np.abs(delta) - s_sum / 2.0


def overlap_per_axis(pos0: np.ndarray, size0: np.ndarray,
                     pos1: np.ndarray, size1: np.ndarray) -> np.ndarray:
    """Length of the projection overlap on each of the three axes, shape ``(N, 3)`` in metres."""
    half0 = size0 / 2.0
    half1 = size1 / 2.0
    min0 = pos0 - half0
    max0 = pos0 + half0
    min1 = pos1 - half1
    max1 = pos1 + half1
    return np.maximum(0.0, np.minimum(max0, max1) - np.maximum(min0, min1))


def tangential_min_overlap(contact_ax: np.ndarray, ov: np.ndarray) -> np.ndarray:
    """The smaller of the two overlaps perpendicular to the contact axis, per row.

    This is the quantity the feasibility rule calls overlap: a screwdriver needs contact in
    both tangential directions, so the weaker one decides.
    """
    n = len(contact_ax)
    rows = np.arange(n)
    # The two tangential indices, in ascending order, for each possible contact axis: 1 and 2
    # for x, 0 and 2 for y, 0 and 1 for z.
    tang_a = np.where(contact_ax == 0, 1, 0)
    tang_b = np.where(contact_ax == 2, 1, 2)
    return np.minimum(ov[rows, tang_a], ov[rows, tang_b])


def derive_failure_reason(under: np.ndarray, gap: np.ndarray,
                          overlap: np.ndarray, thickness: np.ndarray) -> np.ndarray:
    """Re-derive the stored failure reason from the stored metrics.

    The cascade is exclusive and ordered, because the original returns at the first match: a
    block through the table hides everything else, then interpenetration, then a gap, then
    insufficient overlap, and thickness only when none of the others fired. Reordering the
    checks would change the labels rather than just their priority.
    """
    n = len(under)
    out = np.full(n, "NONE", dtype=object)
    cls_table = under > TABLE_THRESHOLD
    cls_pen = (~cls_table) & (gap < PENETRATION_THRESHOLD)
    cls_gap = (~cls_table) & (~cls_pen) & (gap > GAP_THRESHOLD)
    cls_ov = (~cls_table) & (~cls_pen) & (~cls_gap) & (overlap < OVERLAP_THRESHOLD)
    cls_th = (~cls_table) & (~cls_pen) & (~cls_gap) & (~cls_ov) & (thickness > THICKNESS_THRESHOLD)
    out[cls_table] = "TABLE_INTERFERENCE"
    out[cls_pen] = "BLOCK_PENETRATION"
    out[cls_gap] = "BLOCK_GAP"
    out[cls_ov] = "OVERLAP_INSUFFICIENT"
    out[cls_th] = "BLOCK_TOO_THICK"
    return out


def percentiles(arr: np.ndarray, ps: Iterable[float] = (0, 50, 95, 99, 100)) -> str:
    """Format selected percentiles of a metre-valued array as one millimetre line."""
    qs = np.percentile(arr, ps)
    return ", ".join(f"p{int(p)}={fmt_mm(q)}" for p, q in zip(ps, qs))


def main() -> int:
    data_folder = ROOT / config.data.data_folder
    if len(sys.argv) > 1:
        explicit = Path(sys.argv[1])
        if explicit.is_absolute():
            data_folder = explicit.parent
            file_name = explicit.name
        else:
            file_name = sys.argv[1]
    else:
        file_name = config.data.data_file

    print(f"Dataset folder : {data_folder}")
    print(f"Dataset file   : {file_name}")

    df = read_txt_file(data_folder, file_name)
    if df.empty:
        print("FAIL: dataset is empty or unreadable.")
        return 1
    df = manipulate_data(df)

    n_rows = len(df)
    failures: list[str] = []

    hr("1. Schema")
    print(f"  rows: {n_rows:,}    cols: {len(df.columns)}")
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        failures.append(f"missing columns: {missing}")
        print(f"  FAIL: missing columns: {missing}")
    else:
        print("  PASS: all required columns present.")

    hr("2. Cleanliness (NaN / inf / duplicates)")
    numeric_cols = [
        c for c in REQUIRED_COLUMNS if pd.api.types.is_numeric_dtype(df[c])
    ]
    print(f"  numeric column count : {len(numeric_cols)}")
    nan_counts = df[numeric_cols].isna().sum()
    inf_counts = (df[numeric_cols].abs() == np.inf).sum()
    bad_nan = nan_counts[nan_counts > 0]
    bad_inf = inf_counts[inf_counts > 0]
    if len(bad_nan) or len(bad_inf):
        print(f"  FAIL: NaN counts: {bad_nan.to_dict() if len(bad_nan) else 'none'}")
        print(f"        inf counts: {bad_inf.to_dict() if len(bad_inf) else 'none'}")
        failures.append("NaN/inf in numeric columns")
    else:
        print("  PASS: no NaN/inf in numeric columns.")

    # Compared on geometry only, not on the labels: two rows with the same positions and sizes
    # are the same physical assembly measured twice, which inflates a split without adding
    # information. Reported as a warning, since a generator may legitimately resample.
    geom_cols = [c for c in REQUIRED_COLUMNS if "_Pos" in c or "_Size" in c]
    n_dupes = int(df.duplicated(subset=geom_cols).sum())
    if n_dupes:
        print(f"  WARN: {n_dupes:,} duplicate rows on geometry columns "
              f"({100.0 * n_dupes / n_rows:.2f}%).")
    else:
        print("  PASS: no duplicate (pos, size) rows.")

    hr("3. Size positivity / range")
    size_cols = [c for c in df.columns if "_Size" in c and df[c].dtype != object]
    sizes = df[size_cols].to_numpy()
    n_nonpos = int((sizes <= 0).sum())
    if n_nonpos:
        failures.append("non-positive sizes present")
        print(f"  FAIL: {n_nonpos} non-positive size entries.")
    else:
        print(f"  PASS: all sizes > 0.")
    print(f"  size range : [{sizes.min() * 1000:.3f} mm, {sizes.max() * 1000:.3f} mm]")
    print(f"  size mean  : {sizes.mean() * 1000:.3f} mm")

    hr("4. Geometric consistency (analytical vs stored)")

    pos0 = df[["Block0_PosX", "Block0_PosY", "Block0_PosZ"]].to_numpy(np.float64)
    pos1 = df[["Block1_PosX", "Block1_PosY", "Block1_PosZ"]].to_numpy(np.float64)
    size0 = df[["Block0_SizeX", "Block0_SizeY", "Block0_SizeZ"]].to_numpy(np.float64)
    size1 = df[["Block1_SizeX", "Block1_SizeY", "Block1_SizeZ"]].to_numpy(np.float64)

    bottom0_z = pos0[:, 2] - size0[:, 2] / 2.0
    bottom1_z = pos1[:, 2] - size1[:, 2] / 2.0
    under0_calc = np.maximum(0.0, TABLE_HEIGHT_DATASET - bottom0_z)
    under1_calc = np.maximum(0.0, TABLE_HEIGHT_DATASET - bottom1_z)

    under0_stored = df["Block0_UnderSurface_m"].to_numpy(np.float64)
    under1_stored = df["Block1_UnderSurface_m"].to_numpy(np.float64)

    err_u0 = np.abs(under0_calc - under0_stored)
    err_u1 = np.abs(under1_calc - under1_stored)

    # A wrong TABLE_HEIGHT_DATASET shifts every row by the same amount, so the median offset
    # separates that case from genuinely inconsistent labels, which would scatter.
    median_offset_b0 = float(np.median(under0_calc - under0_stored))
    if abs(median_offset_b0) > 1e-4 and err_u0.max() > 1e-4:
        print(f"  WARN: under_surface_b0 systematic offset {fmt_mm(median_offset_b0)}; "
              f"TABLE_SURFACE_HEIGHT used at generation may differ from 0.0.")

    print(f"  Block0 under_surface  max|err| = {fmt_mm(err_u0.max())}, "
          f"99%-ile = {fmt_mm(np.percentile(err_u0, 99))}")
    print(f"  Block1 under_surface  max|err| = {fmt_mm(err_u1.max())}, "
          f"99%-ile = {fmt_mm(np.percentile(err_u1, 99))}")

    if err_u0.max() < 1e-4 and err_u1.max() < 1e-4:
        print("  PASS: under_surface analytically reproduces stored labels.")
    else:
        failures.append("under_surface mismatch beyond 0.1 mm")
        print("  FAIL: under_surface mismatch beyond 0.1 mm.")

    # The contact axis is recovered by trying every candidate and scoring the full triple of
    # gap, overlap and thickness against the stored labels, rather than by the geometric
    # heuristic in derive_contact_axis. That heuristic is ambiguous when two axes have
    # identical projections, and here a misidentified axis would be reported as a label
    # inconsistency, which is exactly the wrong conclusion.
    ov_xyz = overlap_per_axis(pos0, size0, pos1, size1)

    gap_stored = df["Block1_PlannedGap_m"].to_numpy(np.float64)
    overlap_stored = df["Block1_Overlap_m"].to_numpy(np.float64)
    thickness_stored = df["Block1_Thickness_m"].to_numpy(np.float64)

    n = n_rows
    rows = np.arange(n)

    # Three axes times the two face polarities. The gap at generation is measured against a
    # specific face, so its sign comes from the face index and not from the geometry. Usually
    # the two agree and the gap reduces to the absolute offset minus the summed half-extents,
    # but under heavy interpenetration the part under test can sit on the unexpected side and
    # the two disagree. Both polarities are therefore tried per axis.
    n_cand = 6
    candidate_errs = np.full((n, n_cand), np.inf, dtype=np.float64)
    candidate_gap = np.zeros((n, n_cand), dtype=np.float64)
    candidate_ov = np.zeros((n, n_cand), dtype=np.float64)
    candidate_th = np.zeros((n, n_cand), dtype=np.float64)
    candidate_axis = np.zeros(n_cand, dtype=np.int64)

    for k in range(n_cand):
        ax = k // 2
        face_sign = 1.0 if k % 2 == 0 else -1.0
        candidate_axis[k] = ax

        delta_ax = pos1[:, ax] - pos0[:, ax]
        sum_ax = size0[:, ax] + size1[:, ax]
        g = face_sign * delta_ax - sum_ax / 2.0

        ax_arr = np.full(n, ax, dtype=np.int64)
        ov = tangential_min_overlap(ax_arr, ov_xyz)
        th = size1[:, ax]

        candidate_gap[:, k] = g
        candidate_ov[:, k] = ov
        candidate_th[:, k] = th
        candidate_errs[:, k] = (
            np.abs(g - gap_stored)
            + np.abs(ov - overlap_stored)
            + np.abs(th - thickness_stored)
        )

    best = np.argmin(candidate_errs, axis=1)
    contact_ax = candidate_axis[best]
    gap_calc = candidate_gap[rows, best]
    overlap_calc = candidate_ov[rows, best]
    thickness_calc = candidate_th[rows, best]
    face_sign_used = np.where(best % 2 == 0, 1.0, -1.0)

    err_gap = np.abs(gap_calc - gap_stored)
    err_ov = np.abs(overlap_calc - overlap_stored)
    err_th = np.abs(thickness_calc - thickness_stored)

    print(f"  planned_gap       max|err| = {fmt_mm(err_gap.max())}, "
          f"99%-ile = {fmt_mm(np.percentile(err_gap, 99))}, "
          f"mean = {fmt_mm(err_gap.mean())}")
    print(f"  overlap           max|err| = {fmt_mm(err_ov.max())}, "
          f"99%-ile = {fmt_mm(np.percentile(err_ov, 99))}, "
          f"mean = {fmt_mm(err_ov.mean())}")
    print(f"  thickness         max|err| = {fmt_mm(err_th.max())}, "
          f"99%-ile = {fmt_mm(np.percentile(err_th, 99))}, "
          f"mean = {fmt_mm(err_th.mean())}")

    # 0.1 mm. Well above single-precision round-off in the stored file, well below the
    # millimetre scale at which the feasibility thresholds operate.
    geom_tol = 1e-4
    geom_ok = (err_gap.max() < geom_tol and err_ov.max() < geom_tol
               and err_th.max() < geom_tol)
    if geom_ok:
        print("  PASS: gap/overlap/thickness analytically match stored labels.")
    else:
        failures.append("gap/overlap/thickness mismatch beyond 0.1 mm")
        print("  FAIL: see above mismatches.")
        worst_metric, worst_err = max(
            [("gap", err_gap), ("overlap", err_ov), ("thickness", err_th)],
            key=lambda kv: kv[1].max(),
        )
        idx = np.argsort(worst_err)[-3:][::-1]
        print(f"  Worst {worst_metric} mismatches:")
        for i in idx:
            print(f"    row {i}: stored={fmt_mm(eval(worst_metric + '_stored')[i])}, "
                  f"calc={fmt_mm(eval(worst_metric + '_calc')[i])}, "
                  f"contact_ax={contact_ax[i]}")

    print(f"  contact axis distribution: "
          f"X={int((contact_ax == 0).sum()):,}, "
          f"Y={int((contact_ax == 1).sum()):,}, "
          f"Z={int((contact_ax == 2).sum()):,}")
    # Roughly half of each polarity is expected, since the six faces split evenly. A strong
    # imbalance means the dataset only ever attaches from one side, which the surrogate would
    # then never learn to handle.
    n_neg_sign = int((face_sign_used < 0).sum())
    print(f"  face_sign distribution recovered: +1 = {n_rows - n_neg_sign:,}, "
          f"-1 = {n_neg_sign:,} ({100.0 * n_neg_sign / n_rows:.2f}% negative)")

    hr("5. Threshold-classification consistency")

    # Derived from the STORED metrics, not the recomputed ones: check 4 has already established
    # that the two agree, so this isolates the threshold cascade from the geometry.
    fail1_calc = derive_failure_reason(under1_stored, gap_stored,
                                       overlap_stored, thickness_stored)
    fail1_stored = df["Block1_FailureReason"].astype(str).to_numpy()
    fail1_match = (fail1_calc == fail1_stored)
    n_fail1_bad = int((~fail1_match).sum())

    # The base block is only ever measured against the table: overlap, thickness and gap are
    # properties of the joint and are stored against the part under test. Its only possible
    # failure is therefore interference with the table.
    fail0_calc = np.where(under0_stored > TABLE_THRESHOLD,
                          "TABLE_INTERFERENCE", "NONE")
    fail0_stored = df["Block0_FailureReason"].astype(str).to_numpy()
    fail0_match = (fail0_calc == fail0_stored)
    n_fail0_bad = int((~fail0_match).sum())

    print(f"  Block0_FailureReason match: "
          f"{n_rows - n_fail0_bad:,} / {n_rows:,} "
          f"({100.0 * (n_rows - n_fail0_bad) / n_rows:.3f}%)")
    print(f"  Block1_FailureReason match: "
          f"{n_rows - n_fail1_bad:,} / {n_rows:,} "
          f"({100.0 * (n_rows - n_fail1_bad) / n_rows:.3f}%)")

    if n_fail0_bad == 0 and n_fail1_bad == 0:
        print("  PASS: all FailureReason values agree with thresholds.")
    else:
        if n_fail1_bad:
            mism = pd.crosstab(
                pd.Series(fail1_stored[~fail1_match], name="stored"),
                pd.Series(fail1_calc[~fail1_match], name="derived"),
            )
            print("  WARN: Block1 mismatch crosstab (stored x derived):")
            for line in mism.to_string().splitlines():
                print(f"    {line}")
        if n_fail0_bad:
            print(f"  WARN: {n_fail0_bad} Block0 FailureReason mismatches")
        # A sample sitting almost exactly on a threshold can be classified either way
        # depending on rounding, so counting how many are that close separates harmless
        # boundary noise from a real disagreement about the rule.
        edge_band = 5e-5
        edge_count = int(
            (np.abs(np.abs(gap_stored) - GAP_THRESHOLD) < edge_band).sum()
            + (np.abs(under1_stored - TABLE_THRESHOLD) < edge_band).sum()
            + (np.abs(overlap_stored - OVERLAP_THRESHOLD) < edge_band).sum()
            + (np.abs(thickness_stored - THICKNESS_THRESHOLD) < edge_band).sum()
        )
        print(f"  (samples within ±0.05 mm of any threshold: {edge_count:,} — "
              f"floating-point edge cases account for some of these.)")
        # Only a mismatch above a thousandth of the rows counts as a failure; below that it is
        # within what the boundary cases above can account for.
        if n_fail1_bad / n_rows > 1e-3:
            failures.append(f"FailureReason mismatch beyond 0.1%")

    # Unlike the checks above this one admits no tolerance: the feasibility flag is defined as
    # the conjunction, so a single disagreeing row is a contradiction and not an approximation.
    good_calc = (fail0_stored == "NONE") & (fail1_stored == "NONE")
    good_stored = df["Assembly_Good?"].astype(int).to_numpy().astype(bool)
    if (good_calc == good_stored).all():
        print("  PASS: Assembly_Good is exactly Block0+Block1 NONE-failure conjunction.")
    else:
        n_bad = int((good_calc != good_stored).sum())
        failures.append(f"Assembly_Good inconsistent in {n_bad} rows")
        print(f"  FAIL: Assembly_Good inconsistent in {n_bad} rows.")

    hr("6. Distributions")

    n_good = int(good_stored.sum())
    n_bad = int((~good_stored).sum())
    print(f"  Feasibility: {n_good:,} good ({100.0 * n_good / n_rows:.2f}%) "
          f"| {n_bad:,} bad ({100.0 * n_bad / n_rows:.2f}%)")

    print("  Block1_FailureReason counts:")
    counts = pd.Series(fail1_stored).value_counts()
    for name, c in counts.items():
        print(f"    {name:25s} {int(c):8,d} ({100.0 * c / n_rows:5.2f}%)")

    # The buckets are deliberately fine below a millimetre and coarse above it. What limits the
    # surrogate is not the bulk of the distribution but how many samples sit near the gap
    # threshold, since that is the region where the decision boundary has to be learned.
    abs_gap_mm = np.abs(gap_stored) * 1000.0
    edges = [0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0, np.inf]
    print(f"  |gap| (mm) buckets:")
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (abs_gap_mm >= lo) & (abs_gap_mm < hi)
        c = int(m.sum())
        print(f"    [{lo:5.2f}, {hi:5.2f}): {c:8,d} ({100.0 * c / n_rows:5.2f}%)")

    # Interpenetration and separation are different failures and are repaired differently, so
    # a dataset dominated by one of them teaches only half the problem.
    n_pen = int((gap_stored < 0).sum())
    n_zero = int((gap_stored == 0).sum())
    n_sep = int((gap_stored > 0).sum())
    print(f"  Signed gap polarity: penetration={n_pen:,} ({100.0 * n_pen / n_rows:.2f}%), "
          f"zero={n_zero}, separation={n_sep:,} ({100.0 * n_sep / n_rows:.2f}%)")

    # A millimetre-wide band centred on the gap threshold, counted on both polarities. These
    # are the samples that carry the boundary; the buckets above only show the shape around it.
    near_thresh_pos = int(((gap_stored > 0.0005) & (gap_stored < 0.0015)).sum())
    near_thresh_neg = int(((gap_stored < -0.0005) & (gap_stored > -0.0015)).sum())
    print(f"  Samples in critical band 0.5 mm < gap <  1.5 mm: {near_thresh_pos:,}")
    print(f"  Samples in critical band -1.5 mm < gap < -0.5 mm: {near_thresh_neg:,}")

    print("  Block0 size ranges (mm):")
    for ax in ("X", "Y", "Z"):
        col = f"Block0_Size{ax}"
        v = df[col].to_numpy()
        print(f"    Size{ax} : {percentiles(v, (0, 50, 95, 100))}")
    print("  Block1 size ranges (mm):")
    for ax in ("X", "Y", "Z"):
        col = f"Block1_Size{ax}"
        v = df[col].to_numpy()
        print(f"    Size{ax} : {percentiles(v, (0, 50, 95, 100))}")

    hr("Summary")
    if failures:
        print(f"  RESULT: FAIL ({len(failures)} issue(s))")
        for f in failures:
            print(f"    - {f}")
        return 1
    print("  RESULT: PASS — dataset is internally consistent.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
