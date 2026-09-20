"""Measure how thin a block the box encoder and box decoder can still represent.

The repair moves a code, and the only way it can make a block thin enough to be screwed is to
drive the decoded contact-axis size below ``config.gnn.thresh_thickness_max``. Whether that is
reachable at all is a property of the autoencoder, not of the repair: the pair was trained on
half-extents between ``config.data.sampling_min`` and ``config.data.sampling_max``, and if the
smallest representable edge length lies above the thickness threshold, then the whole feasible
region sits below the floor and no amount of optimisation gets there. The two decoders must not
be confused here: this probes the decoder from code to half-extents, the one in the repair
loop, not the auxiliary signed-distance decoder of stage 1.

Two measurements, both printed and plotted to ``results/codec_floor/``:

Test A sweeps a target contact-axis size, runs it through encoder and decoder, and compares
what comes back. A reconstruction that flattens out above the threshold is the floor.

Test B iterates the projection the repair itself applies, clamping the contact axis to a target
and pushing the result through encoder and decoder again, and reports the fixed point. This is
the number that matters, because a floor the repair can only escape at codes far outside the
training range would not be a usable thin block.

Run:
    python tools/codec_floor_test.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
from config.config import config  # noqa: E402
from src.dec_box import get_box_decoder  # noqa: E402
from src.gnn_dataset_preparation import get_box_encoder  # noqa: E402

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

DEVICE = "cpu"
OUT_DIR = Path(__file__).parent.parent / "results" / "codec_floor"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SAMP_MIN_M = float(config.data.sampling_min)           # half-extent, metres
SAMP_MAX_M = float(config.data.sampling_max)
# The configuration bounds the half-extent, the thickness rule bounds the edge length, so the
# two are put on the same footing before anything is compared.
FULL_MIN_MM = SAMP_MIN_M * 2.0 * 1000.0
FULL_MAX_MM = SAMP_MAX_M * 2.0 * 1000.0
THRESH_THICKNESS_MM = float(config.gnn.thresh_thickness_max) * 1000.0

# Which axis is called the thickness axis is arbitrary for this test; the encoder treats the
# three symmetrically. The other two are held at a size well inside the training range so that
# only the swept axis can be responsible for what is measured.
CONTACT_AXIS = 0
OTHER_SIZE_MM = 120.0


def _size_to_half(size_m: torch.Tensor) -> torch.Tensor:
    return size_m / 2.0


def _decode_to_size(decoder, z: torch.Tensor) -> torch.Tensor:
    """Decode a code to edge lengths in metres; the decoder itself returns half-extents."""
    return decoder(z) * 2.0


def test_a_roundtrip(encoder, decoder) -> float:
    """Sweep the target contact-axis size and measure what the roundtrip gives back.

    Returns:
        The reconstructed size for the thinnest target, which is the floor thin boxes are
        pulled up to.
    """
    # 5 to 60 mm in 1 mm steps: the sweep has to start well below the threshold to show where
    # the reconstruction stops following the target, and end well above it to show it recover.
    targets_mm = np.linspace(5.0, 60.0, 56)
    other_m = OTHER_SIZE_MM / 1000.0

    sizes = np.tile([other_m, other_m, other_m], (len(targets_mm), 1)).astype(np.float32)
    sizes[:, CONTACT_AXIS] = targets_mm / 1000.0
    size_t = torch.from_numpy(sizes)

    with torch.no_grad():
        z = encoder(_size_to_half(size_t))
        size_rec = _decode_to_size(decoder, z)
    rec_mm = size_rec[:, CONTACT_AXIS].cpu().numpy() * 1000.0

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(targets_mm, rec_mm, "o-", ms=3, label="reconstructed contact size")
    ax.plot(targets_mm, targets_mm, "k--", lw=1, label="perfect (y = x)")
    ax.axhline(FULL_MIN_MM, color="red", ls=":", label=f"codec floor ({FULL_MIN_MM:.0f} mm full)")
    ax.axvline(THRESH_THICKNESS_MM, color="green", ls=":", label=f"thickness thresh ({THRESH_THICKNESS_MM:.0f} mm)")
    ax.set_xlabel("target contact-axis full size [mm]")
    ax.set_ylabel("reconstructed contact-axis full size [mm]")
    ax.set_title("Test A — codec roundtrip on thin boxes\n(other axes held at %.0f mm)" % OTHER_SIZE_MM)
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = OUT_DIR / "test_a_roundtrip.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)

    print("\n=== Test A — roundtrip (contact axis) ===")
    print(f"  codec trained on full sizes [{FULL_MIN_MM:.0f}, {FULL_MAX_MM:.0f}] mm")
    print(f"  thickness threshold = {THRESH_THICKNESS_MM:.0f} mm\n")
    print(f"  {'target [mm]':>12} {'recon [mm]':>12} {'abs err [mm]':>13}")
    for t, r in zip(targets_mm, rec_mm):
        if abs(t - round(t)) < 1e-6 and int(round(t)) in (5, 10, 15, 20, 30, 40, 50, 60):
            print(f"  {t:>12.1f} {r:>12.1f} {abs(r - t):>13.1f}")
    print(f"\n  -> plot: {out}")
    return float(rec_mm[0])


def test_b_roundtrip_fixedpoint(encoder, decoder, roundtrip_floor_mm: float) -> None:
    """Iterate the projection the repair applies and report the size it settles on.

    Every step, the repair clamps the decoded size and re-encodes it. Iterating that operator
    on its own shows the fixed point of the contact-axis size: the size that actually results
    when the repair asks for a given target. Re-clamping to a concrete size before every
    encode is what keeps this measurement honest, because it never leaves the region the
    encoder was trained on, whereas an unconstrained search over codes could reach a small
    decoded size at a code where nothing else in the pipeline is meaningful.
    """
    print("\n=== Test B — hard-cap roundtrip fixed point ===")
    print("  (iterate: set contact axis to target T, then encode->decode)\n")

    targets_mm = [10.0, 15.0, 20.0, 25.0, 30.0, 40.0]
    other_m = OTHER_SIZE_MM / 1000.0
    n_iter = 8

    print(f"  {'target [mm]':>12} {'fixed pt [mm]':>14}")
    fixed_pts = {}
    for t_mm in targets_mm:
        size = torch.tensor([[other_m, other_m, other_m]])
        size[0, CONTACT_AXIS] = t_mm / 1000.0
        with torch.no_grad():
            for _ in range(n_iter):
                # The clamp has to be reapplied inside the loop, not once before it: the
                # repair also re-imposes it after every step, and applying it only once would
                # measure a plain roundtrip instead of the operator's fixed point.
                size[0, CONTACT_AXIS] = t_mm / 1000.0
                size = _decode_to_size(decoder, encoder(_size_to_half(size)))
        fp_mm = size[0, CONTACT_AXIS].item() * 1000.0
        fixed_pts[t_mm] = fp_mm
        print(f"  {t_mm:>12.1f} {fp_mm:>14.1f}")

    fig, ax = plt.subplots(figsize=(7, 5))
    ts = list(fixed_pts.keys())
    fps = list(fixed_pts.values())
    ax.plot(ts, fps, "o-", ms=4, label="roundtrip fixed point")
    ax.plot(ts, ts, "k--", lw=1, label="perfect (y = x)")
    ax.axhline(THRESH_THICKNESS_MM, color="green", ls=":", label=f"thickness thresh ({THRESH_THICKNESS_MM:.0f} mm)")
    ax.axvline(THRESH_THICKNESS_MM, color="green", ls=":")
    ax.axhline(FULL_MIN_MM, color="red", ls=":", label=f"codec floor ({FULL_MIN_MM:.0f} mm full)")
    ax.set_xlabel("target contact size the optimiser asks for [mm]")
    ax.set_ylabel("contact size that actually results [mm]")
    ax.set_title("Test B — hard-cap roundtrip fixed point")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = OUT_DIR / "test_b_roundtrip_fixedpoint.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"\n  -> plot: {out}")

    # The verdict is read at the threshold, because that is the only target whose fixed point
    # decides whether a pair can be repaired at all.
    fp_at_thresh = fixed_pts.get(20.0, roundtrip_floor_mm)
    print("\n=== VERDICT ===")
    if fp_at_thresh > THRESH_THICKNESS_MM + 1.0:
        print(f"  WALL CONFIRMED.")
        print(f"  - Asking the codec for a {THRESH_THICKNESS_MM:.0f} mm beam yields "
              f"{fp_at_thresh:.1f} mm (Test A & B agree).")
        print(f"  - Sub-{FULL_MIN_MM:.0f} mm targets are pulled up to a ~"
              f"{roundtrip_floor_mm:.0f} mm floor; the hard-cap roundtrip silently")
        print(f"    re-inflates any clamped-thin size, so the constraint never holds.")
        print(f"  - decode(z) CAN reach 0 mm, but only at OOD z where the GNN is")
        print(f"    blind — that is a bluff channel, not a usable thin beam.")
        print(f"  => Root cause: codec trained on full sizes >= {FULL_MIN_MM:.0f} mm "
              f"(config.data.sampling_min={SAMP_MIN_M}).")
        print(f"     Fix: retrain encoder+decoder with sampling_min low enough to")
        print(f"     cover the feasible thickness range (<= {THRESH_THICKNESS_MM:.0f} mm).")
    else:
        print(f"  No hard wall: asking for {THRESH_THICKNESS_MM:.0f} mm yields "
              f"{fp_at_thresh:.1f} mm. Look elsewhere (GNN / constraints).")


def main() -> None:
    encoder = get_box_encoder(device=DEVICE)
    decoder = get_box_decoder(device=DEVICE)
    roundtrip_floor_mm = test_a_roundtrip(encoder, decoder)
    test_b_roundtrip_fixedpoint(encoder, decoder, roundtrip_floor_mm)


if __name__ == "__main__":
    main()
