"""Test whether the thickness of an elongated box is readable from its latent code at all.

The surrogate under-reads thickness near the 20 mm threshold for slab-like aspect ratios, which
``tools/gnn_thickness_bias.py`` measures. That leaves two candidate causes, and they call for
opposite fixes: either the code of the frozen encoder does not carry the thickness of an
elongated box, in which case the latent width has to grow, or it does and the surrogate simply
fails to read it while fitting four heads at once, in which case the code is not the problem.

This script separates the two by training a small MLP on nothing but the code and the thickness
of slab boxes, then running the same footprint sweep as the bias script. A probe that stays
sharp at the threshold shows the code is sufficient and moves the blame to the surrogate; a
probe that is biased in the same way points at the latent width.

Console output only, no artefact. It reads the frozen encoder and never the decoder, so it says
nothing about how well a code can be turned back into a size.

Run:
    python tools/latent_thickness_probe.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.gnn_dataset_preparation import get_box_encoder  # noqa: E402

DEVICE = "cpu"
N_TRAIN = 80_000
EPOCHS = 300
LR = 1e-3


def sample_slab_boxes(n, rng):
    """Draw elongated boxes and return their half-extents and thickness label in millimetres."""
    # Thickness is drawn log-uniformly so the band around the 20 mm threshold is dense without
    # the range collapsing onto it, which would leave the probe nothing to fit a slope to.
    th = np.exp(rng.uniform(np.log(0.008), np.log(0.050), size=n))
    tx = rng.uniform(0.030, 0.200, size=n)
    ty = rng.uniform(0.030, 0.200, size=n)
    full = np.stack([tx, ty, th], axis=1).astype(np.float32)
    # The encoder takes half-extents; the label stays a full edge length, which is what the
    # thickness criterion is stated in.
    half = full / 2.0
    return torch.from_numpy(half), torch.from_numpy((th * 1000.0).astype(np.float32))


class Probe(nn.Module):
    """Single-task MLP from a latent code to a thickness in millimetres."""

    def __init__(self, d=8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, z):
        return self.net(z).squeeze(-1)


def main():
    rng = np.random.default_rng(42)
    torch.manual_seed(42)
    enc = get_box_encoder(device=DEVICE)

    half, th_mm = sample_slab_boxes(N_TRAIN, rng)
    with torch.no_grad():
        z = enc(half)
    probe = Probe(z.shape[1])
    opt = torch.optim.Adam(probe.parameters(), lr=LR)
    print(f"Training probe z->thickness on {N_TRAIN} slab boxes ({EPOCHS} epochs)...")
    for ep in range(EPOCHS):
        opt.zero_grad()
        pred = probe(z)
        loss = nn.functional.mse_loss(pred, th_mm)
        loss.backward()
        opt.step()
        if (ep + 1) % 100 == 0:
            print(f"  epoch {ep+1}: train MAE {loss.sqrt().item():.2f} mm (rmse)")

    # The same four footprints as tools/gnn_thickness_bias.py, so probe and surrogate are read
    # off identical geometry and the two tables can be compared row by row.
    CONFIGS = [("cube 100x100", 100, 100), ("slab 200x40", 200, 40),
               ("small 40x40", 40, 40), ("thin 60x24", 60, 24)]
    targets = np.arange(12.0, 28.01, 0.5)
    print("\n  config            slope[16-24]   bias@20mm   probe-MAE@band")
    probe.eval()
    for label, tx, ty in CONFIGS:
        full = np.stack([
            np.full_like(targets, tx / 1000.0),
            np.full_like(targets, ty / 1000.0),
            targets / 1000.0,
        ], axis=1).astype(np.float32)
        with torch.no_grad():
            zc = enc(torch.from_numpy(full / 2.0))
            pr = probe(zc).numpy()
        band = (targets >= 16) & (targets <= 24)
        slope = float(np.polyfit(targets[band], pr[band], 1)[0])
        bias20 = float(np.interp(20.0, targets, pr - targets))
        mae_band = float(np.abs(pr[band] - targets[band]).mean())
        d = "UNDER" if bias20 < 0 else "over"
        print(f"  {label:>14}   {slope:>10.2f}   {bias20:>+7.2f} ({d:>5})   {mae_band:>6.2f} mm")

    print("\n  Interpretation:")
    print("   slopes ~1 and |bias@20| small for SLABS => z holds thickness, a model")
    print("   CAN read it => GNN bottleneck is learnability/loss, not dim-8.")
    print("   slab still biased here => genuine representation limit (raise latent_dim).")


if __name__ == "__main__":
    main()
