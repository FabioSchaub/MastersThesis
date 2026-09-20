"""Report how well a trained autoencoder reconstructs each of the fourteen shapes.

The validation in ``src.validate_shape_encoder`` uses parts from externally generated
assemblies, and those cover only a few of the vocabulary's members. This tool closes the gap by
generating random instances of every type itself and measuring the same quantity: the magnitude
of the decoded distance field at the input surface points.

It is the capacity gate of the pipeline. If a type cannot be reconstructed before the
compression is applied, the code is too narrow for the vocabulary and has to be widened, or the
first stage has to be trained further. What must not happen is that the compression is asked to
repair it: the penalty removes unused entries, it does not create representational capacity.

Errors are reported in the units of the canonical frame, where they are comparable with the
training loss, and in millimetres at a stated reference size, which is where the sub-millimetre
criterion is applied.

Run:
    python -m tools.validate_vocab_recon --checkpoint <path to the autoencoder>

Writes nothing. The exit code is zero if every type passes and two otherwise, so it can gate
the next step.
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

BASE_DIR = Path(__file__).resolve().parent.parent


def main() -> int:
    """Measure the reconstruction of every shape type and decide whether all of them pass."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", default=str(
        BASE_DIR / config.autoencoder.autoencoder_folder /
        f"best_encoder_decoder_general_canon_v14_latentdim{config.autoencoder.latent_dim}.pth"))
    p.add_argument("--n_per", type=int, default=20, help="random instances per shape type")
    p.add_argument("--ref_mm", type=float, default=135.0,
                   help="reference real max-extent (mm) for the norm->mm conversion (typical Gemini block)")
    p.add_argument("--submm_mm", type=float, default=1.0, help="pass threshold in mm")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = Path(args.checkpoint)
    if not ckpt.is_absolute():
        ckpt = BASE_DIR / ckpt
    if not ckpt.exists():
        print(f"ERROR: checkpoint not found: {ckpt}")
        return 1

    encoder, decoder, latent_dim = load_models(ckpt, device)
    n_surface = config.autoencoder.n_surface
    # A shape normalised to the canonical extent stands for a real part of the reference size,
    # so this ratio converts a distance in the canonical frame into millimetres on that part.
    to_mm = args.ref_mm / CANON_EXTENT

    print(f"\nReconstruction by shape type — {args.n_per} instances each, "
          f"latent_dim {latent_dim}, mm @ {args.ref_mm:.0f}mm block:\n")
    rows = []
    all_ok = True
    for i, st in enumerate(SHAPE_TYPES):
        means, worsts = [], []
        for k in range(args.n_per):
            # Seeded per instance, so two checkpoints are compared on exactly the same shapes.
            np.random.seed(9973 * i + 41 * k + 1)
            # Generated in the canonical frame, which is the frame the encoder was trained in.
            s = generate_random_shape(n_surface=n_surface, n_query=256,
                                      shape_type=st, canonical=True)
            z = encode(encoder, s.surface_points, device)
            err = np.abs(decode_sdf_at(decoder, z, s.surface_points, device))
            means.append(float(err.mean()))
            worsts.append(float(err.max()))
        mean_norm = float(np.mean(means))
        worst_norm = float(np.max(worsts))
        # Judged on the mean rather than the worst point. A single badly reconstructed point,
        # typically at an edge, does not make a shape unusable.
        ok = (mean_norm * to_mm) < args.submm_mm
        all_ok = all_ok and ok
        rows.append((st, mean_norm, mean_norm * to_mm, worst_norm * to_mm, ok))

    print(f"{'shape':14s} {'mean|SDF|':>10s} {'mean mm':>8s} {'worst mm':>9s}  {'pass':>4s}")
    print("-" * 52)
    for st, mn, mm, wmm, ok in sorted(rows, key=lambda r: r[2], reverse=True):
        print(f"{st:14s} {mn:10.4f} {mm:8.2f} {wmm:9.2f}  {'OK' if ok else 'FAIL':>4s}")

    print(f"\n{'ALL SUB-MM' if all_ok else 'SOME TYPES OVER THRESHOLD'} "
          f"(threshold {args.submm_mm:.1f}mm @ {args.ref_mm:.0f}mm block)")
    if not all_ok:
        print("Over-threshold types: latent_dim 16 is cramped for the richer family ->\n"
              "  bump config.autoencoder.latent_dim (24/32) OR lower stage1_recon_threshold,\n"
              "  then re-train the baseline. Do NOT compensate with the LV penalty.")
    return 0 if all_ok else 2


if __name__ == "__main__":
    sys.exit(main())
