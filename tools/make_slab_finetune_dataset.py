"""Generate a fine-tuning set of elongated blocks spanning a wide thickness range.

The counterpart to ``tools/make_slab_boundary_dataset.py``, built on the opposite trade-off.
Concentrating samples at the threshold makes the decision region dense but leaves the target
almost constant, and a regression head fitted on a target with no spread can do no better than
predict its mean. Here the thickness is drawn log-uniformly across a wide range instead, so the
band around the threshold is still well covered while the target keeps enough variance for the
slope to be identifiable. What is forced is the footprint: every active block is an elongated
slab, the aspect ratio the surrogate reads worst, and the overlap is held comfortably feasible
so thickness alone decides the label.

Unlike the boundary script this one constructs its pairs directly rather than through
``generate_assembly``, but it labels them with the generator's own analytical helpers, so the
semantics still match the training set. A replay sample of the current training set is mixed in
and the result shuffled.

Writes ``data/finetune_slab.txt``. Fine-tuning consumes it through
``tools/finetune_gnn.py --data-file finetune_slab.txt``.

Run:
    python tools/make_slab_finetune_dataset.py
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
    _build_header,
    _classify_from_metrics,
    _compute_planned_overlap,
    _row_from_assembly,
)

N_SLAB = 30_000
N_REPLAY = 30_000
SEED = 42
OUT_NAME = "finetune_slab.txt"

# Every pair is a vertical stack, which fixes the contact axis to z and makes the thickness the
# z edge length of the active block for the whole file.
FACE = 4


def make_slab_sample(rng: random.Random) -> dict:
    """Build one stack of an elongated slab on a larger base block."""
    # Log-uniform rather than uniform: it puts proportionally more mass on the thin end, where
    # the 20 mm threshold sits, without narrowing the range and flattening the target.
    thickness = float(np.exp(rng.uniform(np.log(0.008), np.log(0.050))))
    # The two tangential ranges barely overlap, so the footprint is elongated whichever way the
    # swap below falls; the swap only decides which axis is the long one.
    tx = rng.uniform(0.080, 0.200)
    ty = rng.uniform(0.030, 0.120)
    if rng.random() < 0.5:
        tx, ty = ty, tx
    size_1 = np.array([tx, ty, thickness])

    # The base is drawn at least as large as the slab on both tangential axes, so the overlap is
    # bounded by the slab and stays feasible. Thickness is then the only criterion in play.
    size_0 = np.array([
        rng.uniform(max(tx, 0.10), 0.20),
        rng.uniform(max(ty, 0.10), 0.20),
        rng.uniform(0.05, 0.20),
    ])
    pos_0 = np.array([TABLE_CENTER_X, TABLE_CENTER_Y,
                      TABLE_SURFACE_HEIGHT + size_0[2] / 2.0])
    # Resting exactly on the base: half of each block's z size above the shared contact plane.
    pos_1 = pos_0.copy()
    pos_1[2] = pos_0[2] + size_0[2] / 2.0 + size_1[2] / 2.0
    # A jitter of at most 10 mm per axis, small against the tangential sizes above, so the
    # overlap varies from pair to pair without any pair being pushed past the criterion.
    pos_1[0] += rng.uniform(-0.01, 0.01)
    pos_1[1] += rng.uniform(-0.01, 0.01)

    overlap_real = round(_compute_planned_overlap(pos_0, size_0, pos_1, size_1, FACE), 6)
    thickness_real = round(float(size_1[2]), 6)
    fail = _classify_from_metrics(overlap=overlap_real, thickness=thickness_real)

    return {
        "positions": [pos_0, pos_1],
        "sizes": [size_0, size_1],
        "faces": [-1, FACE],
        "fail_reasons": [FAIL_NONE, fail],
        "overlap": [overlap_real],
        "thickness": [thickness_real],
    }


def main() -> None:
    rng = random.Random(SEED)
    np.random.seed(SEED)
    cols = _build_header(2)

    print(f"Generating {N_SLAB} slab pairs (wide thickness, elongated footprint)...")
    rows = [_row_from_assembly(make_slab_sample(rng), 2) for _ in range(N_SLAB)]
    gen = pd.DataFrame(rows, columns=cols)

    th = gen["Block1_Thickness_m"].astype(float) * 1000
    ov = gen["Block1_Overlap_m"].astype(float) * 1000
    ar = (gen["Block1_SizeX"].astype(float).clip(lower=1e-9)
          .combine(gen["Block1_SizeY"].astype(float), max)
          / gen["Block1_SizeZ"].astype(float))
    print(f"  thickness mean {th.mean():.1f} mm (std {th.std():.1f}, range {th.min():.0f}-{th.max():.0f})")
    print(f"  overlap  mean {ov.mean():.1f} mm  (>=10mm: {(ov>=10).mean()*100:.0f}%)")
    print(f"  aspect ratio (maxtang/thickness) mean {ar.mean():.1f}")
    print(f"  Assembly_Good rate: {(gen['Assembly_Good?']=='True').mean()*100:.0f}%")

    print(f"\nReplay {N_REPLAY} from {config.data.data_file}...")
    orig = pd.read_csv(ROOT / "data" / config.data.data_file, sep=",")
    # The training set carries extra columns that the generated rows do not, so both sides are
    # reduced to the shared header before they are concatenated.
    replay = orig.sample(n=min(N_REPLAY, len(orig)), random_state=SEED)
    replay = replay[[c for c in cols if c in replay.columns]]
    gen = gen[[c for c in cols if c in gen.columns]]

    combined = pd.concat([gen, replay], ignore_index=True)
    combined = combined.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    out = ROOT / "data" / OUT_NAME
    combined.to_csv(out, sep=",", index=False)
    print(f"\nWrote {len(combined)} rows -> {out}  ({len(gen)} slab + {len(replay)} replay)")


if __name__ == "__main__":
    main()
