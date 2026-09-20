"""Report how far the surrogate's thickness regression is off at the 20 mm threshold.

The counterpart of ``tools/gnn_overlap_bias.py`` for the second criterion. It builds fully
overlapping stacked pairs whose contact-axis size, which is the thickness, sweeps 12 to 28 mm,
and repeats the sweep for four tangential footprints of the active block: a bias that depends
on the footprint means the surrogate reads the same thickness differently depending on the
aspect ratio of the block it belongs to. A prediction that falls below the true value near the
threshold is the mechanism behind a bluff, since it lets a block that is in fact too thick pass
the gate.

Prints a table of local slope, signed bias at the threshold and the true thickness at which the
prediction crosses it, and writes ``results/gnn_bias/thickness_bias_by_footprint.png``.

Run:
    python tools/gnn_thickness_bias.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.repair_process import THRESH_THICKNESS, load_models, predict_feasibility  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUT_DIR = ROOT / "results" / "gnn_bias"
OUT_DIR.mkdir(parents=True, exist_ok=True)
THR_MM = THRESH_THICKNESS * 1000.0

ANCHOR_MM = 100.0

# Label with the two tangential edge lengths of the active block in millimetres. The thickness
# swept below is identical across these four, so any spread between the curves is attributable
# to the footprint alone.
CONFIGS = [
    ("cube 100x100", 100.0, 100.0),
    ("slab 200x40", 200.0, 40.0),
    ("small 40x40", 40.0, 40.0),
    ("thin 60x24", 60.0, 24.0),
]


def _t(x, device):
    """Wrap a nested sequence as a float tensor on the given device."""
    return torch.tensor(np.asarray(x, dtype=np.float32), dtype=torch.float, device=device)


def make_pair(thickness_mm, tx_mm, ty_mm, device):
    """Build a centred vertical stack whose active block has the requested footprint."""
    a = ANCHOR_MM / 1000.0
    t = thickness_mm / 1000.0
    size_anchor = _t([a, a, a], device)
    pos_anchor = _t([0.0, 0.0, a / 2.0], device)
    size_active = _t([tx_mm / 1000.0, ty_mm / 1000.0, t], device)
    pos_active = _t([0.0, 0.0, a + t / 2.0], device)
    return size_anchor, size_active, pos_anchor, pos_active


def sweep_config(label, tx, ty, targets, gnn_model, device):
    """Sweep the thickness for one footprint and return predictions and threshold statistics."""
    preds = []
    for t_mm in targets:
        sa, sc, pa, pc = make_pair(t_mm, tx, ty, device)
        _, reg = predict_feasibility(sa, sc, pa, pc, gnn_model)
        preds.append(reg[1] * 1000.0)
    preds = np.array(preds)
    bias = preds - targets
    # Fit the slope on a narrow band around the threshold rather than the whole sweep: only the
    # local behaviour decides whether the gate is sharp.
    band = (targets >= 16) & (targets <= 24)
    slope = float(np.polyfit(targets[band], preds[band], 1)[0])
    bias20 = float(np.interp(THR_MM, targets, bias))
    # Interpolating targets over preds inverts the curve, giving the true thickness up to which
    # the surrogate still passes the pair. Only defined while the sweep brackets the threshold.
    cross = float(np.interp(THR_MM, preds, targets)) if preds[0] < THR_MM < preds[-1] else float("nan")
    return preds, bias, slope, bias20, cross


def main():
    device = torch.device("cpu")
    gnn_model, _ = load_models(device)
    gnn_model = gnn_model.to(device)

    targets = np.arange(12.0, 28.01, 0.5)
    print(f"\nthreshold = {THR_MM:.0f} mm   (overlap held ~feasible)\n")
    print(f"  {'config':>14} {'slope[16-24]':>13} {'bias@20mm':>11} {'green-lit-until':>16}")

    fig, ax = plt.subplots(figsize=(7.5, 6))
    ax.plot(targets, targets, "k--", lw=1, label="unbiased (y = x)")
    for label, tx, ty in CONFIGS:
        preds, bias, slope, bias20, cross = sweep_config(
            label, tx, ty, targets, gnn_model, device
        )
        direction = "UNDER" if bias20 < 0 else "over"
        crossing = f"~{cross:.1f} mm" if not np.isnan(cross) else "n/a"
        print(f"  {label:>14} {slope:>13.2f} {bias20:>+9.2f} ({direction:>5}) {crossing:>10}")
        ax.plot(targets, preds, "o-", ms=2.5, label=f"{label}")

    ax.axhline(THR_MM, color="green", ls=":", label=f"threshold ({THR_MM:.0f} mm)")
    ax.axvline(THR_MM, color="green", ls=":")
    ax.set_xlabel("true thickness [mm]")
    ax.set_ylabel("GNN predicted thickness [mm]")
    ax.set_title("GNN thickness bias vs. active-block footprint (aspect ratio)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = OUT_DIR / "thickness_bias_by_footprint.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"\n  bias@20mm < 0 => GNN under-predicts => green-lights too-thick beams (BLUFF).")
    print(f"  -> plot: {out}")


if __name__ == "__main__":
    main()
