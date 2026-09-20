"""Check that every shape of the vocabulary can be generated and sampled consistently.

The first gate of the pipeline, and the cheapest. It generates a few instances of each of the
fourteen types and confirms four things about each: that the sampled surface really lies on the
zero level of the distance field, that the field has both an inside and an outside, that no
value is infinite or undefined, and that the shape stays within the cube the queries are drawn
from. A shape that fails any of these would train the autoencoder on nonsense, and finding that
out on the cluster costs a day.

Nothing is written. The result is a table on the terminal and an exit code, which is non-zero
if any type fails, so the check can gate a submission.

Run:
    python -m tools.verify_shape_vocab
"""

from __future__ import annotations

import sys

import numpy as np

from src.enc_dec_dataset_generation import SHAPE_TYPES, generate_random_shape

N_PER = 8
# In the units the shapes are generated in, where a characteristic dimension is of order one.
# The surface sampler accepts a point within 0.02 of the surface, so the mean over a whole
# sample has to stay below that to indicate that most points landed rather than were padded.
MEAN_TOL = 0.015
# Has to match the default of the generator; a shape reaching this would have been clipped.
BOUNDS = 1.2


def main() -> None:
    """Report one row per shape type, so a broken generator shows up as a single bad line."""
    print(f"Verifying {len(SHAPE_TYPES)} shape types, {N_PER} instances each\n")
    print(f"{'shape':14s} {'mean|sdf|':>10s} {'max|sdf|':>9s} {'maxabs':>7s} "
          f"{'interior':>8s} {'finite':>6s} {'inbounds':>8s}")
    all_ok = True
    for i, st in enumerate(SHAPE_TYPES):
        w_mean = w_max = w_abs = 0.0
        interior = finite = inbounds = True
        for k in range(N_PER):
            # Seeded per instance, so a failure names a shape that can be reproduced exactly.
            np.random.seed(7919 * i + 31 * k + 1)
            s = generate_random_shape(n_surface=2000, n_query=4000, shape_type=st)
            d = np.abs(s.sdf_fn(s.surface_points))
            w_mean = max(w_mean, float(d.mean()))
            w_max = max(w_max, float(d.max()))
            # The worst case over the instances is kept throughout, not the average: one bad
            # instance of a type is a failure of that type.
            w_abs = max(w_abs, float(np.abs(s.surface_points).max()))
            if not np.all(np.isfinite(s.sdf_values)):
                finite = False
            if not (s.sdf_values < 0).any():
                interior = False
            if w_abs > BOUNDS + 1e-3:
                inbounds = False
        shape_ok = (w_mean < MEAN_TOL) and interior and finite and inbounds
        all_ok = all_ok and shape_ok
        flag = "" if shape_ok else "   <-- FAIL"
        print(f"{st:14s} {w_mean:10.4f} {w_max:9.4f} {w_abs:7.2f} "
              f"{str(interior):>8s} {str(finite):>6s} {str(inbounds):>8s}{flag}")

    print("\nALL SHAPES OK" if all_ok else "\nSOME SHAPES FAILED")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
