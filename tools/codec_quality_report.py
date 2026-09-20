"""Report what a trained autoencoder's set of latent codes looks like after compression.

Where the reconstruction check asks whether the shapes come back, this asks how much of the code
they need. It encodes instances of every shape type and reports four things about the resulting
set of codes:

    the spectrum      the spread of each entry, sorted. Where it drops is how many entries the
                      vocabulary actually occupies.
    the correlation   over all entries, and over the varying ones alone. The emptied entries all
                      sit at nearly the same value for every shape, which by itself makes any two
                      codes look alike; restricting the measure to the varying entries shows how
                      distinct the codes really are.
    the spread        of the code magnitudes.
    the prunability   the reconstruction error before and after every emptied entry is replaced
                      by its mean over the set. If it barely moves, those entries genuinely carry
                      nothing and the compression can be taken at face value. This is the claim
                      the compression is worth making, and the only one of the four that tests
                      it rather than describing the codes.

Run:  python -m tools.codec_quality_report --checkpoint <path to the autoencoder>

Writes nothing; the report goes to the terminal.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.config import config
from src.enc_dec_dataset_generation import CANON_EXTENT, SHAPE_TYPES, generate_random_shape
from src.validate_shape_encoder import decode_sdf_at, encode, load_models

BASE = Path(__file__).resolve().parent.parent


def mean_pairwise_cos(Z: np.ndarray) -> float:
    """Mean cosine similarity between distinct rows of a set of codes."""
    n = np.linalg.norm(Z, axis=1, keepdims=True)
    n[n == 0] = 1.0
    U = Z / n
    C = U @ U.T
    off = ~np.eye(len(Z), dtype=bool)
    return float(C[off].mean())


def main() -> int:
    """Encode a sample of every shape type and report the structure of the resulting codes."""
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--k_per", type=int, default=15, help="shapes per type")
    p.add_argument("--frac", type=float, default=0.01, help="active-dim threshold (fraction of max sigma)")
    args = p.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = Path(args.checkpoint)
    ck = ck if ck.is_absolute() else BASE / ck
    if not ck.exists():
        print(f"ERROR: checkpoint not found: {ck}")
        return 1
    enc, dec, ld = load_models(ck, dev)
    ns = config.autoencoder.n_surface
    # The same reference part size the reconstruction check uses by default, so the errors
    # printed by the two tools are on the same scale.
    to_mm = 135.0 / CANON_EXTENT

    Z, surfs = [], []
    for i, st in enumerate(SHAPE_TYPES):
        for k in range(args.k_per):
            np.random.seed(4441 * i + 17 * k + 3)
            s = generate_random_shape(n_surface=ns, n_query=64, shape_type=st, canonical=True)
            Z.append(encode(enc, s.surface_points, dev).cpu().numpy())
            # The surface is kept as well, since the prunability is measured on the same points
            # the code was produced from.
            surfs.append(s.surface_points)
    Z = np.stack(Z)
    sigma = Z.std(0)
    order = np.argsort(sigma)[::-1]
    active = sigma > args.frac * sigma.max()
    n_active = int(active.sum())
    norms = np.linalg.norm(Z, axis=1)

    print(f"\n==== {ck.name} ====")
    print(f"latent_dim={ld}   codes={len(Z)}   ACTIVE dims (sigma > {args.frac}*max) = {n_active}/{ld}")
    print("sigma spectrum (sorted): " + " ".join(f"{sigma[o]:.3f}" for o in order))
    print(f"z-norm:  mean={norms.mean():.3f}  std={norms.std():.3f}  min={norms.min():.3f}  max={norms.max():.3f}")
    print(f"cos (all {ld} dims):        {mean_pairwise_cos(Z):+.3f}")
    print(f"cos (active {n_active} dims):  {mean_pairwise_cos(Z[:, active]):+.3f}   <- the meaningful subspace")

    # Replaced by the mean rather than by zero: an emptied entry sits at whatever constant the
    # training left it at, and forcing it to zero would test a different question.
    dead = ~active
    zmean = Z.mean(0)
    ef, ep = [], []
    for z, surf in zip(Z, surfs):
        zp = z.copy()
        zp[dead] = zmean[dead]
        ef.append(np.abs(decode_sdf_at(dec, torch.from_numpy(z.astype(np.float32)).to(dev), surf, dev)).mean())
        ep.append(np.abs(decode_sdf_at(dec, torch.from_numpy(zp.astype(np.float32)).to(dev), surf, dev)).mean())
    ef_mm, ep_mm = float(np.mean(ef)) * to_mm, float(np.mean(ep)) * to_mm
    print(f"prunability: drop {int(dead.sum())} dead dims -> recon {ef_mm:.2f} mm -> {ep_mm:.2f} mm "
          f"(+{ep_mm - ef_mm:.2f} mm)   [small = active dims carry the shape]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
