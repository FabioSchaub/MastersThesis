"""Turn the simulator's raw output into the labelled table the surrogate is trained on.

The simulation repository produces one row per simulated pair, with a flag for every failure
mode it models and the measured contact geometry. Part I assesses two of those criteria, so
this script narrows the data to them.

Two things happen, and both are deliberate. Rows that failed for a reason outside the scope --
the assembly tipping over, a collision with the table, the screwdriver or the gripper not
fitting -- are removed rather than relabelled, because a surrogate trained on them would have
to account for a mode it is never shown the cause of. And the remaining rows are relabelled
from the measured geometry rather than inheriting the simulator's own verdict, which does not
include the thickness limit and would therefore call configurations feasible that cannot be
screwed.

    python tools/csv_to_paper_v0_txt.py
    python tools/csv_to_paper_v0_txt.py --glob "data/num_blocks_2_2026*.csv" --out_tag mysim

Writes one comma-separated table into ``data/``, named after the row count and the time, and
prints the label distribution. That distribution is worth reading: it is the class balance the
training will have to work with.
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

# The two limits in metres. They repeat the values in ``config/config.yaml``, so that this
# conversion can run without the model configuration, and have to be kept in step by hand.
THRESH_OVERLAP_MIN = 0.010
THRESH_THICKNESS_MAX = 0.020

# Failure modes the simulator models that Part I does not assess. A row flagged with any of
# them is dropped: it failed for a reason the surrogate is given no way to see.
DROP_FAILURE_COLS = [
    "Block1_BLOCK_GAP",
    "Block1_TIPPING",
    "Block1_TABLE_INTERFERENCE",
    "Block1_SCREWDRIVER_TABLE_INTERFERENCE",
    "Block1_SCREWDRIVER_BLOCK_INTERFERENCE",
    "Block1_GRIPPER_INTERFERENCE",
    "Block1_SCREWDRIVER_INFO_HIT",
]

# The face convention of the simulator: codes run in pairs, plus and minus on each axis in
# turn, so the contact axis is the code halved. The thickness is measured along it.
FACE_TO_AXIS = {0: 0, 1: 0, 2: 1, 3: 1, 4: 2, 5: 2}
AXIS_TO_SIZE_COL = {0: "Block1_SizeX", 1: "Block1_SizeY", 2: "Block1_SizeZ"}


def read_csv_robust(path: Path) -> pd.DataFrame:
    """Read one simulator CSV, skipping the separator hint some exports put on the first line."""
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

    # The contact patch is the smaller of the two tangential overlaps, since the screwdriver
    # needs room in both directions at once.
    overlap_m = np.minimum(
        df["Block1_OverlapTangential1Val"].values.astype(np.float32),
        df["Block1_OverlapTangential2Val"].values.astype(np.float32),
    )
    # The simulator reports a small negative value where the blocks fail to meet on a
    # tangential axis. There is no such thing as a negative contact patch, so it becomes zero.
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

    # A row that violates both criteria gets a class of its own rather than being filed under
    # whichever was checked first, so the two can be told apart when the data is inspected.
    # The training reads the measured quantities, not this string.
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

    # The first block has no joint of its own and is never screwed, so it cannot fail.
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
    print("Thickness (mm):")
    print(
        f"  mean={1000*thickness_m.mean():.2f}  median={1000*np.median(thickness_m):.2f}  "
        f"min={1000*thickness_m.min():.2f}  max={1000*thickness_m.max():.2f}"
    )

    # The overlap criterion is one the simulator also evaluates, so the relabelling can be
    # checked against it. Disagreement would mean the geometry is being read differently here
    # than in the simulation, and the labels could not be trusted.
    if "Block1_OVERLAP_INSUFFICIENT" in df.columns:
        sim_oi = df["Block1_OVERLAP_INSUFFICIENT"].astype(bool).values
        agree = (sim_oi == overlap_fail).mean()
        print(
            f"\nSim OVERLAP_INSUFFICIENT vs our overlap<10mm derivation: "
            f"{100*agree:.2f} % agreement"
        )

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
