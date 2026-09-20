"""Put the code drift of the shape repair on a scale that can be interpreted.

The repair reports how far the shape code moved, but that number means nothing on its own: the
codes are not normalised, so a drift of 0.08 is only large or small relative to the geometry of
the code set itself. This script measures the three references the drift has to be read against.

    - How long a shape code is, so the drift can be read as a fraction of it.
    - How far apart the codes the surrogate was trained on lie, that is the sampling density of
      the region on which it has evidence.
    - How far the shape continuum displaces a code from its base shape, which is the radius of
      the region the training set actually covers.

It then reports how many repaired codes end up outside that region. That is the quantity the
argument of Part III rests on: not that the optimiser produces implausible shapes, but that it
leaves the set of codes the surrogate has ever been asked about.

Run from the repository root with the exported repair configurations in place:

    python tools/z_drift_scales.py --configs data_and_latents/repair_configs.json
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch

# The jitter that generated the shape continuum, as a fraction of the per-dimension spread of
# the fourteen base codes. It has to match scripts/z_continuum/generate_meshes.py in the
# simulation repository, where the continuum was produced.
CONTINUUM_SIGMA = 0.15


def _matrix(path: str) -> tuple[np.ndarray, list[str]]:
    """Load a mapping from mesh name to shape code as a matrix, one code per row."""
    obj = torch.load(path, map_location="cpu")
    names = list(obj.keys())
    rows = [np.asarray(obj[k]).ravel() for k in names]
    return np.stack(rows).astype(np.float64), names


def main() -> None:
    """Print the reference scales and how far the repaired codes sit outside them."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", default="encoder_decoder_model/shape_latents_lv_v14_lam005.pt")
    ap.add_argument("--continuum", default="encoder_decoder_model/latents_continuum.pth")
    ap.add_argument("--configs", default="repair_configs.json")
    args = ap.parse_args()

    base, base_names = _matrix(args.base)
    cont, _ = _matrix(args.continuum)
    train = np.vstack([base, cont])

    # 1. The length of a code.
    norms = np.linalg.norm(base, axis=1)

    # 2. The spacing of the training codes. The diagonal is excluded so that the nearest
    #    neighbour of a code is never the code itself.
    d_train = np.linalg.norm(train[:, None, :] - train[None, :, :], axis=2)
    np.fill_diagonal(d_train, np.inf)
    spacing = d_train.min(axis=1)

    # 3. The reach of the continuum: how far its members sit from the base shape they came from.
    reach = np.linalg.norm(cont[:, None, :] - base[None, :, :], axis=2).min(axis=1)

    # The distance between vocabulary members, which is the scale at which one shape turns into
    # a different shape rather than a distorted version of the same one.
    d_base = np.linalg.norm(base[:, None, :] - base[None, :, :], axis=2)
    np.fill_diagonal(d_base, np.inf)
    vocab_nn = d_base.min(axis=1)
    i, j = np.unravel_index(np.argmin(d_base), d_base.shape)

    print("code length         median %.3f" % np.median(norms))
    print("training spacing    median %.4f" % np.median(spacing))
    print("continuum reach     median %.4f  max %.4f" % (np.median(reach), reach.max()))
    print("vocabulary nearest  median %.3f  min %.3f (%s / %s)"
          % (np.median(vocab_nn), d_base[i, j], base_names[i], base_names[j]))

    cfg = json.load(open(args.configs))
    repaired = [a for a in cfg["assemblies"] if a["mode"] == "scale_z"]
    z_out = np.array([a["obj1"]["z"] for a in repaired], dtype=np.float64)
    drift = np.array([a["obj1"]["z_drift"] for a in repaired])

    # The decisive measurement: distance from each repaired code to the closest code the
    # surrogate was ever trained on.
    d_min = np.linalg.norm(z_out[:, None, :] - train[None, :, :], axis=2).min(axis=1)
    outside = int((d_min > reach.max()).sum())

    print("\nrepaired codes      n %d" % len(repaired))
    print("drift               mean %.4f  median %.4f  max %.4f"
          % (drift.mean(), np.median(drift), drift.max()))
    print("distance to nearest training code   median %.4f" % np.median(d_min))
    print("  as a fraction of the code length  %.0f%%" % (100 * np.median(drift) / np.median(norms)))
    print("  in units of the training spacing  %.1fx" % (np.median(drift) / np.median(spacing)))
    print("  in units of the continuum reach   %.1fx" % (np.median(drift) / np.median(reach)))
    print("outside the continuum's reach of any training code: %d of %d" % (outside, len(d_min)))


if __name__ == "__main__":
    main()
