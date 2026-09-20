"""Ask whether the autoencoder still resolves the thickness axis once a block becomes a slab.

The surrogate under-predicts thickness on elongated blocks, and that can have two causes. It
may sit in the surrogate, in which case fine-tuning reaches it, or it may sit one level below,
in the latent representation: if squeezing three sizes into one code lets the large tangential
extents dominate and blurs the small contact-axis extent, then no fine-tune can help, because
the surrogate never sees anything but the code.

The test holds the contact-axis thickness at a few values, sweeps the tangential footprint from
near-cube to extreme slab, and reports the encoder-decoder roundtrip error on the contact axis
alone. It uses the box encoder and the box decoder from code to half-extents, not the auxiliary
signed-distance decoder. Output is a printed table and a verdict; nothing is written to disk.

Run:
    python tools/codec_aspect_test.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.dec_box import get_box_decoder  # noqa: E402
from src.gnn_dataset_preparation import get_box_encoder  # noqa: E402

CONTACT_AXIS = 2  # the block is placed on top, so thickness is measured along z
# Bracketing config.gnn.thresh_thickness_max (20 mm): one value below, one on it, one above.
THICKNESSES_MM = [16.0, 20.0, 24.0]
# (label, tangential X mm, tangential Y mm), ordered from near-cube to extreme slab.
FOOTPRINTS = [
    ("cube-ish", None, None),   # None means: take the thickness, giving a cube
    ("40x40", 40.0, 40.0),
    ("100x40", 100.0, 40.0),
    ("200x40", 200.0, 40.0),
    ("200x100", 200.0, 100.0),
    ("300x60", 300.0, 60.0),
]


def main():
    enc = get_box_encoder(device="cpu")
    dec = get_box_decoder(device="cpu")

    print("\nRoundtrip thickness error vs. footprint (contact axis = z)\n")
    print(f"  {'thickness':>10} {'footprint':>10} {'recon_th[mm]':>13} {'err[mm]':>9} {'err[%]':>8}")
    rows = []
    for t_mm in THICKNESSES_MM:
        for label, tx, ty in FOOTPRINTS:
            txx = tx if tx is not None else t_mm
            tyy = ty if ty is not None else t_mm
            size = torch.tensor([[txx, tyy, t_mm]], dtype=torch.float) / 1000.0
            half = size / 2.0
            with torch.no_grad():
                z = enc(half)
                # The decoder returns half-extents; doubling gives the edge length.
                recon = dec(z) * 2.0
            recon_th = float(recon[0, CONTACT_AXIS]) * 1000.0
            err = recon_th - t_mm
            rel = 100.0 * err / t_mm
            rows.append((t_mm, label, recon_th, err, rel))
            print(f"  {t_mm:>9.0f}m {label:>10} {recon_th:>13.1f} {err:>+9.2f} {rel:>+7.1f}%")
        print()

    # The verdict is read at the threshold itself, because that is the only thickness at which
    # a roundtrip error changes the feasibility decision.
    at20 = [r for r in rows if r[0] == 20.0]
    cube_err = next(r[3] for r in at20 if r[1] == "cube-ish")
    slab_err = next(r[3] for r in at20 if r[1] == "200x40")
    print("=== VERDICT (at 20 mm thickness) ===")
    print(f"  cube roundtrip error:      {cube_err:+.2f} mm")
    print(f"  slab 200x40 roundtrip err: {slab_err:+.2f} mm")
    if abs(slab_err) > abs(cube_err) + 1.0:
        print(f"  => Codec LOSES thickness for slabs (err grows {abs(slab_err)-abs(cube_err):.1f} mm).")
        print(f"     The slab bias is partly a LATENT-REPRESENTATION problem; GNN")
        print(f"     fine-tuning alone cannot fully fix it — the encoder/decoder")
        print(f"     would need attention (e.g. higher latent_dim or aspect-aware training).")
    else:
        print(f"  => Codec preserves thickness across aspect ratios "
              f"(slab err ~ cube err).")
        print(f"     The slab bias lives in the GNN, not the latent => fine-tuning")
        print(f"     on slab geometries CAN fix it.")


if __name__ == "__main__":
    main()
