"""Reduce the raw simulator output to the two-criterion training set this work is built on.

The simulator records many ways a placement can fail. This work certifies exactly two, at least
``config.gnn.thresh_overlap_min`` of horizontal contact and at most
``config.gnn.thresh_thickness_max`` of joint thickness, and is deliberately silent about the
rest. Feeding the surrogate rows that failed for a reason it has no input to predict would ask
it to model something it cannot see, so those rows are dropped rather than relabelled.

The script concatenates the simulator CSVs matched by the glob, drops the out-of-scope
failures, and recomputes the labels from the recorded geometry: the overlap as the smaller of
the two tangential contacts, the thickness as the extent of the part under test along its
contact axis, and feasibility as both criteria holding. The simulator's own verdict is not
carried over, because it does not include the thickness rule.

Writes one comma-separated file to ``data/<tag>_num_blocks_<n>_samples_<count>_<timestamp>.txt``
in the twenty-column layout the surrogate trainer reads, and prints the label distribution.

Run:
    python tools/csv_to_paper_v0_txt.py
    python tools/csv_to_paper_v0_txt.py --glob "data/num_blocks_2_2026*.csv" --out_tag mysim
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

# Restated here rather than imported, so the produced file records the rule its labels were
# made under. They have to match config.gnn.thresh_overlap_min and thresh_thickness_max.
THRESH_OVERLAP_MIN = 0.010
THRESH_THICKNESS_MAX = 0.020

# Failure modes outside the two criteria certified here. A row with any of these set is
# dropped: its geometry may well satisfy both criteria, so keeping it would teach the surrogate
# to call a valid placement infeasible for a reason its inputs never mention.
DROP_FAILURE_COLS = [
    "Block1_BLOCK_GAP",
    "Block1_TIPPING",
    "Block1_TABLE_INTERFERENCE",
    "Block1_SCREWDRIVER_TABLE_INTERFERENCE",
    "Block1_SCREWDRIVER_BLOCK_INTERFERENCE",
    "Block1_GRIPPER_INTERFERENCE",
    "Block1_SCREWDRIVER_INFO_HIT",
]

# Faces are numbered two per axis, positive side first: 0 and 1 are the x faces, 2 and 3 the y
# faces, 4 and 5 the z faces. The same convention is used throughout the repair.
FACE_TO_AXIS = {0: 0, 1: 0, 2: 1, 3: 1, 4: 2, 5: 2}
AXIS_TO_SIZE_COL = {0: "Block1_SizeX", 1: "Block1_SizeY", 2: "Block1_SizeZ"}


def read_csv_robust(path: Path) -> pd.DataFrame:
    """Read one simulator CSV, skipping the spreadsheet separator hint some exports carry."""
    with path.open("r") as f:
        first = f.readline().strip()
    skiprows = 1 if first.startswith("sep=") else 0
    df = pd.read_csv(path, sep=",", skiprows=skiprows)
    print(f"  read {path.name}: {len(df):>7d} rows, {df.shape[1]} cols")
    return df


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--glob",
        type=str,
        default="data/num_blocks_2_*.csv",
        help="Glob (relative to repo root) to find input CSVs.",
    )
    parser.add_argument(
        "--out_tag",
        type=str,
        default="sim",
        help="Filename prefix for the output .txt (default 'sim').",
    )
    parser.add_argument(
        "--num_blocks",
        type=int,
        default=2,
        help="Number of blocks per assembly (currently only 2 supported).",
    )
    args = parser.parse_args()

    csv_paths = sorted((ROOT).glob(args.glob))
    if not csv_paths:
        print(f"ERROR: no CSVs matched glob {args.glob}")
        return 1

    print(f"Loading {len(csv_paths)} CSV(s) matching {args.glob}:")
    dfs = [read_csv_robust(p) for p in csv_paths]
    df = pd.concat(dfs, ignore_index=True)
    n_raw = len(df)
    print(f"\nTotal raw rows: {n_raw:,}")

    # A missing failure column is reported rather than raised: older simulator exports do not
    # carry every flag, and the remaining ones still filter what they cover.
    missing_cols = [c for c in DROP_FAILURE_COLS if c not in df.columns]
    if missing_cols:
        print(f"WARNING: expected failure columns missing: {missing_cols}")

    drop_mask = pd.Series(False, index=df.index)
    counts = {}
    for col in DROP_FAILURE_COLS:
        if col not in df.columns:
            continue
        flag = df[col].astype(bool)
        counts[col] = int(flag.sum())
        drop_mask = drop_mask | flag
    print("\nOut-of-scope physics failures (samples dropped if any True):")
    for col, n in counts.items():
        print(f"  {col:48s} {n:>7d}  ({100*n/n_raw:.2f} %)")
    n_dropped = int(drop_mask.sum())
    print(f"  -> dropping {n_dropped:,} samples with any out-of-scope failure")

    df = df.loc[~drop_mask].reset_index(drop=True)
    print(f"Remaining: {len(df):,} rows")

    # The smaller of the two tangential contacts, not their mean: a screwdriver needs clearance
    # on both, so the worse one decides.
    overlap_m = np.minimum(
        df["Block1_OverlapTangential1Val"].values.astype(np.float32),
        df["Block1_OverlapTangential2Val"].values.astype(np.float32),
    )
    # The simulator reports a negative value when the blocks are apart on a tangential axis.
    # As a label that distance has no meaning beyond "no contact", so it is clipped to zero.
    overlap_m = np.clip(overlap_m, 0.0, None)

    contact_face = df["Block1_ContactFace"].values.astype(int)
    axis_idx = np.array([FACE_TO_AXIS.get(int(f), -1) for f in contact_face])
    if (axis_idx < 0).any():
        bad = df.loc[axis_idx < 0]
        raise RuntimeError(
            f"Got {len(bad)} rows with ContactFace not in 0..5; first values: "
            f"{bad['Block1_ContactFace'].head().tolist()}"
        )
    thickness_m = np.empty(len(df), dtype=np.float32)
    for a, col in AXIS_TO_SIZE_COL.items():
        mask = axis_idx == a
        thickness_m[mask] = df.loc[mask, col].values.astype(np.float32)

    overlap_fail = overlap_m < THRESH_OVERLAP_MIN
    thick_fail = thickness_m > THRESH_THICKNESS_MAX
    block1_failed = overlap_fail | thick_fail
    assembly_good = ~block1_failed

    # Samples failing both criteria get their own name rather than being folded into one of the
    # two, which keeps the breakdown readable. The string is used for diagnostics only: the
    # trainer reads the regression targets and the boolean, not this column.
    reason = np.full(len(df), "NONE", dtype=object)
    reason[overlap_fail & ~thick_fail] = "OVERLAP_INSUFFICIENT"
    reason[thick_fail & ~overlap_fail] = "BLOCK_TOO_THICK"
    reason[overlap_fail & thick_fail]  = "BOTH"

    out = pd.DataFrame()
    for col in [
        "Block0_PosX", "Block0_PosY", "Block0_PosZ",
        "Block0_SizeX", "Block0_SizeY", "Block0_SizeZ",
        "Block1_PosX", "Block1_PosY", "Block1_PosZ",
        "Block1_SizeX", "Block1_SizeY", "Block1_SizeZ",
    ]:
        out[col] = df[col].values.astype(np.float32)

    # Block 0 is the base. It is never screwed on, so neither criterion applies to it and it
    # cannot fail by construction.
    out["Block0_Failed?"] = False
    out["Block1_Failed?"] = block1_failed
    out["Block0_FailureReason"] = "NONE"
    out["Block1_FailureReason"] = reason
    out["Block1_Overlap_m"] = overlap_m
    out["Block1_Thickness_m"] = thickness_m
    out["Block1_Face"] = contact_face.astype(np.int32)
    out["Assembly_Good?"] = assembly_good

    n = len(out)
    print("\n=== Paper-v0 label distribution ===")
    print(f"  total samples:               {n:,}")
    print(
        f"  Assembly_Good=True:          {int(assembly_good.sum()):>8,}  "
        f"({100*assembly_good.mean():.2f} %)"
    )
    print(
        f"  Block1_Failed=True:          {int(block1_failed.sum()):>8,}  "
        f"({100*block1_failed.mean():.2f} %)"
    )
    print(
        f"    OVERLAP_INSUFFICIENT only: {int((overlap_fail & ~thick_fail).sum()):>8,}"
    )
    print(
        f"    BLOCK_TOO_THICK only:      {int((thick_fail & ~overlap_fail).sum()):>8,}"
    )
    print(
        f"    BOTH (own label):          {int((overlap_fail & thick_fail).sum()):>8,}"
    )

    print("\nOverlap (mm):")
    print(
        f"  mean={1000*overlap_m.mean():.2f}  median={1000*np.median(overlap_m):.2f}  "
        f"min={1000*overlap_m.min():.2f}  max={1000*overlap_m.max():.2f}"
    )
    # Thickness distribution
    print("Thickness (mm):")
    print(
        f"  mean={1000*thickness_m.mean():.2f}  median={1000*np.median(thickness_m):.2f}  "
        f"min={1000*thickness_m.min():.2f}  max={1000*thickness_m.max():.2f}"
    )

    # The overlap label is recomputed here instead of taken from the simulator, so it is worth
    # checking that the two agree. A low agreement would mean the simulator applied a different
    # overlap rule than the one assumed here, and the recomputation would need revisiting.
    if "Block1_OVERLAP_INSUFFICIENT" in df.columns:
        sim_oi = df["Block1_OVERLAP_INSUFFICIENT"].astype(bool).values
        agree = (sim_oi == overlap_fail).mean()
        print(
            f"\nSim OVERLAP_INSUFFICIENT vs our overlap<10mm derivation: "
            f"{100*agree:.2f} % agreement"
        )

    # The row count and timestamp go into the name so that two runs over different glob
    # patterns cannot silently overwrite each other.
    ts = time.strftime("%Y%m%d_%H%M")
    out_name = (
        f"{args.out_tag}_num_blocks_{args.num_blocks}_samples_{n}_{ts}.txt"
    )
    out_path = ROOT / "data" / out_name
    out.to_csv(out_path, sep=",", index=False)
    print(f"\nWrote {out_path}  ({n:,} rows)")
    print(
        "\nNext step: edit config/config.yaml  data.data_file: "
        f'"{out_name}"  and run gnn_training.'
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
