"""Generate training data around the designs the generative stage actually proposes.

The simulated dataset covers boxes drawn at random, which is not the distribution the repair
meets in use: proposed designs run to slabs and to tall thin columns, and a surrogate fitted
only on random geometry is least reliable on exactly those. This script mines the proposed
designs for the joints they contain, then produces many variations around each one and labels
them in closed form.

Labelling analytically rather than in the simulator is what makes the volume affordable, and
it is legitimate here because the two criteria of Part I are geometric: for axis-aligned boxes
the closed form is the same rule the simulator applies.

Each variation keeps the joint's contact axis and direction but perturbs both blocks' sizes
and slides the child along the tangential axes, so the variations sweep across the limit
rather than clustering on one side of it.

    python tools/build_finetune_dataset.py
    python tools/build_finetune_dataset.py --variations-per-pair 5000

Reads ``pipeline/*_input.csv`` and writes one table in the training schema, by default
``data/finetune_dataset_10mm.txt``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.analytical_metrics import (  # noqa: E402
    analytical_overlap,
    analytical_thickness,
    infer_face,
)

# The two limits in metres, repeating the values in ``config/config.yaml``.
THRESH_OVERLAP_M = 0.010
THRESH_THICKNESS_M = 0.020

# Edge lengths in metres a generated block may take, matching the range of the simulated data
# so that the two sets can be merged without one covering geometry the other never shows.
SIZE_MIN_M = 0.007
SIZE_MAX_M = 0.200


def _detect_n_blocks(df: pd.DataFrame) -> int:
    """Number of blocks the schema provides for, counted from the position columns."""
    n = 0
    while f"Block{n}_PosX" in df.columns:
        n += 1
    return n


def _connected_pairs(row: pd.Series, n: int) -> list[tuple[int, int]]:
    """Joined pairs of one design, each with the lower index first."""
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            col = f"Block{i}_ConnectsTo_Block{j}"
            if col in row.index and int(row[col]) == 1:
                pairs.append((i, j))
    return pairs


def _block_geom(row: pd.Series, idx: int) -> tuple[np.ndarray, np.ndarray] | None:
    """Centre and edge lengths of one block in metres, or nothing if the slot is unused."""
    pos = np.array([row[f"Block{idx}_PosX"], row[f"Block{idx}_PosY"],
                    row[f"Block{idx}_PosZ"]], dtype=np.float64)
    size = np.array([row[f"Block{idx}_SizeX"], row[f"Block{idx}_SizeY"],
                     row[f"Block{idx}_SizeZ"]], dtype=np.float64)
    if np.isnan(pos).any() or np.isnan(size).any():
        return None
    return pos, size


def extract_base_pairs(csv_paths: list[Path]) -> list[dict]:
    """Collect every joint of every design, as the material the variations are built from.

    Only the two blocks' edge lengths and how they meet are kept. Where the joint sat in the
    original design does not matter: each variation is rebuilt from scratch at the origin.

    Returns:
        One entry per joint, with the file it came from, both blocks' edge lengths in metres,
        the contact axis, and the direction from parent to child along it.
    """
    base_pairs: list[dict] = []
    for path in csv_paths:
        df = pd.read_csv(path)
        row = df.iloc[0]
        n = _detect_n_blocks(df)
        for (a, b) in _connected_pairs(row, n):
            ga = _block_geom(row, a)
            gb = _block_geom(row, b)
            if ga is None or gb is None:
                continue
            pos_a, size_a = ga
            pos_b, size_b = gb
            # The lower block is the parent, as everywhere else on this branch; the two
            # quantities are not symmetric in the pair.
            if pos_a[2] <= pos_b[2]:
                anchor_pos, anchor_size = pos_a, size_a
                active_pos, active_size = pos_b, size_b
            else:
                anchor_pos, anchor_size = pos_b, size_b
                active_pos, active_size = pos_a, size_a
            face = infer_face(anchor_pos, anchor_size / 2.0,
                              active_pos, active_size / 2.0)
            contact_axis = face // 2
            sign = +1 if face % 2 == 0 else -1
            base_pairs.append({
                "source": path.name,
                "anchor_size_m": anchor_size,
                "active_size_m": active_size,
                "contact_axis": int(contact_axis),
                "sign": int(sign),
            })
    return base_pairs


def _failure_reason(overlap_m: float, thickness_m: float) -> str:
    """Which criterion a joint violates, with a class of its own for violating both."""
    ov_fail = overlap_m < THRESH_OVERLAP_M
    th_fail = thickness_m > THRESH_THICKNESS_M
    if ov_fail and th_fail:
        return "BOTH"
    if ov_fail:
        return "OVERLAP_INSUFFICIENT"
    if th_fail:
        return "BLOCK_TOO_THICK"
    return "NONE"


def _synth_one_sample(
    base: dict,
    rng: np.random.Generator,
    contact_size_override_prob: float = 0.5,
    tangential_jitter_factor: float = 1.0,
) -> dict:
    """Build one labelled variation of a joint.

    The parent is placed on the table at the origin and the child in contact with it, so the
    variation is a configuration that could be built. Both blocks' edge lengths are perturbed,
    and the child is slid along the two tangential axes, which is what makes the overlap sweep
    across its limit instead of staying near the value the original design had.

    Args:
        base: One entry from :func:`extract_base_pairs`.
        rng: Random generator.
        contact_size_override_prob: How often the child's thickness is redrawn outright rather
            than perturbed. Without this the thickness would stay near whatever the proposed
            design used, and the variations would say little about the thickness limit.
        tangential_jitter_factor: Scales how far the child may slide, relative to the blocks'
            own extent, so the sweep covers full overlap through to none at all.

    Returns:
        One row in the training schema, labelled in closed form.
    """
    anchor_size = base["anchor_size_m"].copy()
    active_size = base["active_size_m"].copy()
    ca = base["contact_axis"]
    sign = base["sign"]

    # Perturbed rather than redrawn, so the variations stay recognisably the proposed design.
    anchor_size *= rng.uniform(0.8, 1.2, size=3)
    active_size *= rng.uniform(0.8, 1.2, size=3)

    # The redrawn range straddles the thickness limit in both directions, from well under it
    # to several times over, so both sides of the criterion are represented.
    if rng.random() < contact_size_override_prob:
        active_size[ca] = rng.uniform(0.005, 0.080)

    anchor_size = np.clip(anchor_size, SIZE_MIN_M, SIZE_MAX_M)
    active_size = np.clip(active_size, SIZE_MIN_M, SIZE_MAX_M)

    anchor_pos = np.array([0.0, 0.0, anchor_size[2] / 2.0])

    active_pos = anchor_pos.copy()
    active_pos[ca] += sign * (anchor_size[ca] / 2.0 + active_size[ca] / 2.0)
    for ax in range(3):
        if ax == ca:
            continue
        max_tang = max(anchor_size[ax], active_size[ax]) * tangential_jitter_factor
        active_pos[ax] = anchor_pos[ax] + rng.uniform(-max_tang, +max_tang)

    he_a = anchor_size / 2.0
    he_c = active_size / 2.0
    face = infer_face(anchor_pos, he_a, active_pos, he_c)
    overlap_m = float(analytical_overlap(anchor_pos, he_a, active_pos, he_c, face))
    thickness_m = float(analytical_thickness(he_c, face))
    failure = _failure_reason(overlap_m, thickness_m)
    good = failure == "NONE"

    return {
        "Block0_PosX": anchor_pos[0], "Block0_PosY": anchor_pos[1], "Block0_PosZ": anchor_pos[2],
        "Block0_SizeX": anchor_size[0], "Block0_SizeY": anchor_size[1], "Block0_SizeZ": anchor_size[2],
        "Block1_PosX": active_pos[0], "Block1_PosY": active_pos[1], "Block1_PosZ": active_pos[2],
        "Block1_SizeX": active_size[0], "Block1_SizeY": active_size[1], "Block1_SizeZ": active_size[2],
        "Block0_Failed?": False, "Block1_Failed?": (not good),
        "Block0_FailureReason": "NONE", "Block1_FailureReason": failure,
        "Block1_Overlap_m": overlap_m, "Block1_Thickness_m": thickness_m,
        "Block1_Face": int(face),
        "Assembly_Good?": good,
    }


COLUMN_ORDER = [
    "Block0_PosX", "Block0_PosY", "Block0_PosZ",
    "Block0_SizeX", "Block0_SizeY", "Block0_SizeZ",
    "Block1_PosX", "Block1_PosY", "Block1_PosZ",
    "Block1_SizeX", "Block1_SizeY", "Block1_SizeZ",
    "Block0_Failed?", "Block1_Failed?",
    "Block0_FailureReason", "Block1_FailureReason",
    "Block1_Overlap_m", "Block1_Thickness_m",
    "Block1_Face", "Assembly_Good?",
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variations-per-pair", type=int, default=2000,
                        help="Synthetic samples per base pair (default 2000)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str,
                        default="data/finetune_dataset_10mm.txt",
                        help="Output filename (default: data/finetune_dataset_10mm.txt)")
    args = parser.parse_args()

    pipeline_dir = ROOT / "pipeline"
    csv_paths = sorted(pipeline_dir.glob("*_input.csv"))
    if not csv_paths:
        print(f"ERROR: no *_input.csv files in {pipeline_dir}/")
        return 1
    print(f"Found {len(csv_paths)} CSVs")

    print("Extracting base pairs...")
    base_pairs = extract_base_pairs(csv_paths)
    print(f"  Total connected pairs: {len(base_pairs)}")

    by_source: dict[str, int] = {}
    for bp in base_pairs:
        by_source[bp["source"]] = by_source.get(bp["source"], 0) + 1
    for src, count in by_source.items():
        print(f"  - {src}: {count} pairs")

    rng = np.random.default_rng(args.seed)
    n_per_pair = int(args.variations_per_pair)
    n_total = len(base_pairs) * n_per_pair

    print(f"\nGenerating {n_per_pair} variations per pair = {n_total} samples")
    rows: list[dict] = []
    for i, base in enumerate(base_pairs):
        for _ in range(n_per_pair):
            rows.append(_synth_one_sample(base, rng))
        if (i + 1) % 5 == 0 or i == len(base_pairs) - 1:
            print(f"  {(i+1):3d}/{len(base_pairs)} pairs done — {len(rows):>7d} samples")

    df = pd.DataFrame(rows, columns=COLUMN_ORDER)
    n_good = int(df["Assembly_Good?"].sum())
    n_total = len(df)
    print(f"\nTotal samples: {n_total}")
    print(f"  Good: {n_good} ({100*n_good/n_total:.1f}%)")
    print(f"  Failure breakdown:")
    print(df["Block1_FailureReason"].value_counts().to_string())

    output_path = ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\nWrote {output_path}")
    print(f"To train on this dataset:")
    print(f"  1. Set config.data.data_file = \"{output_path.name}\" in config.yaml")
    print(f"  2. Or run: python tools/finetune_gnn.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
