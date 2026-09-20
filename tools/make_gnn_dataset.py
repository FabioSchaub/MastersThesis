"""Generate a fully synthetic training set for the surrogate, with analytical labels.

Two halves: pairs built through ``src/dataset_generation.py``'s own construction, which gives
face variety and natural aspect ratios, and pairs whose active block is forced to be a slab,
which is the shape family the surrogate reads worst. Overlap and thickness are drawn
independently here, unlike the shared bucket configuration of the generator, so a sample can be
near one threshold without being dragged towards the other. Labels come from the analytical
metrics, so no simulation is involved.

Writes ``data/latent_gnn_dataset_v5.txt`` in the comma-separated format the graph builder
reads, and prints the resulting threshold statistics and failure mix.

This is not the training set of the reported surrogate. ``config.data.data_file`` records that
the synthetic slab-enriched datasets were reverted in favour of the simulation-derived one,
because they improved the isolated thickness metric but degraded end-to-end repair.

Run:
    python tools/make_gnn_dataset.py
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config.config import config  # noqa: E402
from src.dataset_generation import (  # noqa: E402
    FAIL_NONE,
    TABLE_CENTER_X,
    TABLE_CENTER_Y,
    TABLE_SURFACE_HEIGHT,
    _classify_from_metrics,
    _compute_planned_overlap,
    _construct_pair,
    _contact_axis,
    _row_from_assembly,
    _sample_block0_position,
    _sample_block0_size,
    _build_header,
    generate_assembly,  # noqa: F401 (kept available)
)

N_STD = 180_000
N_SLAB = 120_000
SEED = 42
OUT_NAME = "latent_gnn_dataset_v5.txt"

# Infeasible overlaps are oversampled far beyond their rate in the generated designs. Whether
# the regression head can fit the mapping at all depends on seeing spread on both sides of the
# threshold, not on the training set matching how rare the failure is in practice.
OVERLAP_FAIL_P = 0.30


def sample_thickness(rng: random.Random) -> float:
    """Draw a thickness target in metres, half of it near the threshold, half over a wide range."""
    # Half the draws land inside a narrow band around the 20 mm threshold so the decision region
    # is dense; the other half spans the full range so the target keeps enough variance for a
    # slope to be identifiable instead of collapsing to the mean.
    if rng.random() < 0.5:
        return rng.uniform(0.016, 0.024)
    return float(np.exp(rng.uniform(np.log(0.008), np.log(0.050))))


def sample_overlap(rng: random.Random) -> float:
    """Draw an overlap target in metres, failing below the threshold with ``OVERLAP_FAIL_P``."""
    if rng.random() < OVERLAP_FAIL_P:
        return rng.uniform(0.0, 0.010)
    return float(np.exp(rng.uniform(np.log(0.010), np.log(0.150))))


def _assembly(size_0, pos_0, size_1, pos_1, face) -> dict:
    """Assemble one labelled pair from its geometry, with analytical overlap and thickness."""
    overlap_real = round(_compute_planned_overlap(pos_0, size_0, pos_1, size_1, face), 6)
    thickness_real = round(float(size_1[_contact_axis(face)]), 6)
    fail = _classify_from_metrics(overlap=overlap_real, thickness=thickness_real)
    return {
        "positions": [pos_0, pos_1], "sizes": [size_0, size_1],
        "faces": [-1, face], "fail_reasons": [FAIL_NONE, fail],
        "overlap": [overlap_real], "thickness": [thickness_real],
    }


def make_standard_sample(rng: random.Random) -> dict:
    """Build one pair through the generator's own construction, with a randomly chosen face."""
    # Face 4 is the vertical stack and gets half the samples on its own; the four side faces
    # share the rest. Stacking is the common case in the generated designs.
    face = 4 if rng.random() < 0.5 else rng.randint(0, 3)
    size_0 = _sample_block0_size()
    pos_0 = _sample_block0_position(size_0[2])
    size_1, pos_1 = _construct_pair(face, size_0, pos_0, sample_overlap(rng), sample_thickness(rng))
    return _assembly(size_0, pos_0, size_1, pos_1, face)


def make_slab_sample(rng: random.Random) -> dict:
    """Build one vertical stack whose active block is forced to an elongated footprint."""
    thickness = sample_thickness(rng)
    tx, ty = rng.uniform(0.080, 0.200), rng.uniform(0.030, 0.120)
    if rng.random() < 0.5:
        tx, ty = ty, tx
    size_1 = np.array([tx, ty, thickness])
    size_0 = np.array([rng.uniform(max(tx, 0.10), 0.20),
                       rng.uniform(max(ty, 0.10), 0.20),
                       rng.uniform(0.05, 0.20)])
    pos_0 = np.array([TABLE_CENTER_X, TABLE_CENTER_Y, TABLE_SURFACE_HEIGHT + size_0[2] / 2.0])
    pos_1 = pos_0.copy()
    pos_1[2] = pos_0[2] + size_0[2] / 2.0 + size_1[2] / 2.0
    # Overlap on a vertical stack is the smaller of the two tangential overlaps, so it can only
    # be set on one axis at a time: one is offset to hit the target, the other stays centred.
    binding = rng.choice([0, 1]); other = 1 - binding
    # An overlap can never exceed the shorter of the two blocks on that axis.
    max_ov = float(min(size_0[binding], size_1[binding]))
    target_ov = min(sample_overlap(rng), max_ov)
    delta = (size_0[binding] + size_1[binding]) / 2.0 - target_ov
    pos_1[binding] = pos_0[binding] + rng.choice([-1.0, 1.0]) * delta
    pos_1[other] = pos_0[other] + rng.uniform(-0.005, 0.005)
    return _assembly(size_0, pos_0, size_1, pos_1, 4)


def main() -> None:
    rng = random.Random(SEED)
    np.random.seed(SEED)
    cols = _build_header(2)

    print(f"Part STD: {N_STD} (face variety, natural aspect)...")
    rows = [_row_from_assembly(make_standard_sample(rng), 2) for _ in range(N_STD)]
    print(f"Part SLAB: {N_SLAB} (forced elongated)...")
    rows += [_row_from_assembly(make_slab_sample(rng), 2) for _ in range(N_SLAB)]

    df = pd.DataFrame(rows, columns=cols).sample(frac=1.0, random_state=SEED).reset_index(drop=True)

    ov = df["Block1_Overlap_m"].astype(float) * 1000
    th = df["Block1_Thickness_m"].astype(float) * 1000
    ov_fail = ov < 10; th_fail = th > 20
    print(f"\nTotal {len(df)} samples")
    print(f"  thickness mean {th.mean():.1f} (std {th.std():.1f})  in[16,24]: {((th>=16)&(th<=24)).mean()*100:.0f}%  fail: {th_fail.mean()*100:.0f}%")
    print(f"  overlap   mean {ov.mean():.1f} (std {ov.std():.1f})  fail: {ov_fail.mean()*100:.1f}%")
    print(f"  failure mix: overlap-only {(ov_fail & ~th_fail).mean()*100:.1f}%  "
          f"both {(ov_fail & th_fail).mean()*100:.1f}%  thickness-only {(~ov_fail & th_fail).mean()*100:.1f}%")
    print(f"  Assembly_Good rate: {(~(ov_fail|th_fail)).mean()*100:.0f}%")

    out = ROOT / "data" / OUT_NAME
    df.to_csv(out, sep=",", index=False)
    print(f"\nWrote -> {out}")


if __name__ == "__main__":
    main()
