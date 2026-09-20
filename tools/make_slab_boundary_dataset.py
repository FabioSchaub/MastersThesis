"""Generate a fine-tuning set that concentrates samples in the neighbourhood of the thresholds.

The surrogate is least reliable where it matters most, within a few millimetres of the overlap
and thickness criteria, and a pair it misreads there is exactly a pair the repair can declare
fixed while the geometry is not. This script attacks that by density: it drives
``src/dataset_generation.py`` with a bucket configuration weighted towards boundary and failure
cases, so the decision region is sampled far more heavily than in the default distribution.
Construction and labelling are the generator's own, which keeps the semantics identical to the
training set, and the aspect ratios follow from its tangential sampling rather than being
forced. Half the pairs are vertical stacks, the common case among the generated designs.

A replay sample of the current training set is mixed in and the result shuffled, so fine-tuning
on this does not overwrite what the surrogate already fits. Labels are analytical throughout.

Writes ``data/finetune_slab_boundary.txt``. Fine-tuning consumes it through
``tools/finetune_gnn.py --data-file finetune_slab_boundary.txt``.

Run:
    python tools/make_slab_boundary_dataset.py
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
    BucketConfig,
    SamplingConfig,
    _build_header,
    _row_from_assembly,
    generate_assembly,
)

N_BOUNDARY = 30_000
# Matched one to one against the generated pairs, so neither half dominates the fine-tune.
N_REPLAY = 30_000
SEED = 42
OUT_NAME = "finetune_slab_boundary.txt"

BOUNDARY_CFG = SamplingConfig(
    bucket=BucketConfig(
        p_fail=0.40,
        p_boundary=0.50,
        # Of the failing samples, most land just past the threshold rather than far beyond it:
        # a grossly infeasible pair teaches nothing about where the boundary lies.
        p_near_within_fail=0.70,
    ),
    p_vertical_face=0.5,
)


def main() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    cols = _build_header(2)

    print(f"Generating {N_BOUNDARY} boundary-heavy 2-block pairs...")
    rows = [_row_from_assembly(generate_assembly(2, BOUNDARY_CFG), 2) for _ in range(N_BOUNDARY)]
    gen_df = pd.DataFrame(rows, columns=cols)

    # Reported so a run can be checked against its intent before the file is used: if the share
    # near the threshold is not clearly above the default distribution, the bucket weights above
    # did not take effect.
    th = gen_df["Block1_Thickness_m"].astype(float) * 1000
    ov = gen_df["Block1_Overlap_m"].astype(float) * 1000
    in_band = ((th >= 16) & (th <= 24)).mean() * 100
    print(f"  thickness in [16,24] mm: {in_band:.0f}%   "
          f"(mean {th.mean():.1f} mm, overlap mean {ov.mean():.1f} mm)")
    print(f"  Assembly_Good rate: {(gen_df['Assembly_Good?']=='True').mean()*100:.0f}%")

    print(f"\nLoading replay sample ({N_REPLAY}) from {config.data.data_file}...")
    orig = pd.read_csv(ROOT / "data" / config.data.data_file, sep=",")
    # The training set carries extra columns that the generated rows do not, so both sides are
    # reduced to the shared header before they are concatenated.
    replay = orig.sample(n=min(N_REPLAY, len(orig)), random_state=SEED)
    replay = replay[[c for c in cols if c in replay.columns]]
    gen_df = gen_df[[c for c in cols if c in gen_df.columns]]

    combined = pd.concat([gen_df, replay], ignore_index=True)
    combined = combined.sample(frac=1.0, random_state=SEED).reset_index(drop=True)

    out_path = ROOT / "data" / OUT_NAME
    combined.to_csv(out_path, sep=",", index=False)
    print(f"\nWrote {len(combined)} rows -> {out_path}")
    print(f"  ({len(gen_df)} boundary-heavy + {len(replay)} replay)")


if __name__ == "__main__":
    main()
