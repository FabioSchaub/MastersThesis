"""Build a training set around the geometry the generative stage actually produces.

The simulated training set covers its own sampling distribution densely, but thinly covers the
aspect ratios the generative stage favours: wide slabs and tall thin columns. This script takes
those designs as a starting point rather than a target. It reads every connected pair out of
``pipeline/*_input.csv``, keeps only the pair's two sizes and its contact face, and then
synthesises many two-block variations around each: sizes jittered per axis, the part under test
placed face to face on the base, and the tangential offset swept wide enough that the samples
land on both sides of the overlap criterion.

Labels are analytical, not simulated, so the file can be regenerated without a simulator. It is
written in the same twenty-column layout as the simulated set, which is what lets
``tools/build_mixed_dataset.py`` concatenate the two and the surrogate trainer read the result
unchanged.

Writes to ``data/finetune_dataset_10mm.txt`` unless ``--output`` says otherwise.

Run:
    python tools/build_finetune_dataset.py
    python tools/build_finetune_dataset.py --variations-per-pair 5000
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

# Restated here rather than imported, so the file records the rule its labels were produced
# under. They have to match config.gnn.thresh_overlap_min and config.gnn.thresh_thickness_max,
# or the labels disagree with the ones the surrogate is scored against.
THRESH_OVERLAP_M = 0.010
THRESH_THICKNESS_M = 0.020

# Sizes are clipped to this band after jitter, keeping the synthetic blocks inside the range
# the rest of the pipeline works in.
SIZE_MIN_M = 0.007
SIZE_MAX_M = 0.200


def _detect_n_blocks(df: pd.DataFrame) -> int:
    """Count blocks by walking the ``Block<i>_PosX`` columns until one is missing."""
    n = 0
    while f"Block{n}_PosX" in df.columns:
        n += 1
    return n


def _connected_pairs(row: pd.Series, n: int) -> list[tuple[int, int]]:
    """Return the index pairs marked as connected, each with the lower index first."""
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            col = f"Block{i}_ConnectsTo_Block{j}"
            if col in row.index and int(row[col]) == 1:
                pairs.append((i, j))
    return pairs


def _block_geom(row: pd.Series, idx: int) -> tuple[np.ndarray, np.ndarray] | None:
    """Read one block's centre and size in metres, or ``None`` if the row has gaps."""
    pos = np.array([row[f"Block{idx}_PosX"], row[f"Block{idx}_PosY"],
                    row[f"Block{idx}_PosZ"]], dtype=np.float64)
    size = np.array([row[f"Block{idx}_SizeX"], row[f"Block{idx}_SizeY"],
                     row[f"Block{idx}_SizeZ"]], dtype=np.float64)
    if np.isnan(pos).any() or np.isnan(size).any():
        return None
    return pos, size


def extract_base_pairs(csv_paths: list[Path]) -> list[dict]:
    """Reduce every connected pair in every design to the seed the synthesis needs.

    Only the two sizes and the contact direction are kept. The absolute placement is not: the
    synthesis rebuilds it from scratch, so that what carries over from the design is its
    proportions and not where in the scene it happened to sit.

    Returns:
        One entry per pair, each holding the source file name, the two sizes in metres, the
        contact axis as an index into x, y, z, and the sign along it, positive when the part
        under test sits on the positive side of the base.
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
            # The lower block is the base and the upper one the part under test, the same
            # convention the surrogate is trained under.
            if pos_a[2] <= pos_b[2]:
                anchor_pos, anchor_size = pos_a, size_a
                active_pos, active_size = pos_b, size_b
            else:
                anchor_pos, anchor_size = pos_b, size_b
                active_pos, active_size = pos_a, size_a
            face = infer_face(anchor_pos, anchor_size / 2.0,
                              active_pos, active_size / 2.0)
            # Faces are numbered two per axis, positive side first, so the axis is the face
            # halved and the direction is its parity.
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
    """Name which of the two criteria a sample violates, or both, or neither."""
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
    """Synthesise one two-block sample around a seed pair, with analytical labels.

    The base is placed at the origin and lifted so that it rests on the table, and the part
    under test is put face to face against it, then displaced tangentially. Both sizes are
    jittered per axis.

    Args:
        base: One entry from :func:`extract_base_pairs`.
        rng: Generator, so the whole file is reproducible from the seed.
        contact_size_override_prob: How often the contact-axis size of the part under test is
            replaced by a draw across the thickness range instead of jittered. Jitter alone
            stays near the seed, which for most designs is far above the thickness threshold,
            and the surrogate would then never see a sample close to the decision boundary.
        tangential_jitter_factor: Scales how far the part may be displaced sideways, in units
            of the larger of the two extents on that axis.
    """
    anchor_size = base["anchor_size_m"].copy()
    active_size = base["active_size_m"].copy()
    ca = base["contact_axis"]
    sign = base["sign"]

    anchor_size *= rng.uniform(0.8, 1.2, size=3)
    active_size *= rng.uniform(0.8, 1.2, size=3)

    # 5 to 80 mm brackets the 20 mm thickness threshold on both sides by a wide margin, so the
    # sample lands anywhere from clearly screwable to clearly too thick.
    if rng.random() < contact_size_override_prob:
        active_size[ca] = rng.uniform(0.005, 0.080)

    anchor_size = np.clip(anchor_size, SIZE_MIN_M, SIZE_MAX_M)
    active_size = np.clip(active_size, SIZE_MIN_M, SIZE_MAX_M)

    # Half its own height above the origin, so the base rests on the table rather than being
    # centred in it.
    anchor_pos = np.array([0.0, 0.0, anchor_size[2] / 2.0])

    active_pos = anchor_pos.copy()
    # Half of each extent apart along the contact axis puts the two faces exactly in contact.
    active_pos[ca] += sign * (anchor_size[ca] / 2.0 + active_size[ca] / 2.0)
    for ax in range(3):
        if ax == ca:
            continue
        # Displacing by up to the full extent reaches complete separation, so the sweep covers
        # both sides of the overlap criterion instead of only the well-overlapped end.
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


# The twenty-column layout of the simulated training set, in its order. It is fixed here so
# that the two files can be concatenated without aligning columns first.
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
