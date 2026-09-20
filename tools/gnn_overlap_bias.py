"""Report how far the surrogate's overlap regression is off right at the 10 mm threshold.

Overlap is regressed in logarithmic space, so an aggregate error is dominated by the large
overlaps, where a few millimetres never change the verdict. What decides feasibility, and what
the repair follows, is the accuracy in the immediate neighbourhood of the threshold. The script
therefore synthesises stacked pairs whose overlap sweeps 4 to 20 mm with the thickness held at
a feasible value, so overlap is the only variable, and prints predicted against true overlap
together with the local slope, the signed bias at the threshold, and the true overlap at which
the surrogate starts calling the pair feasible.

Console output only, no artefact. The checkpoint is the one ``src/repair_process.py`` selects,
which the ``GNN_MODEL`` environment variable overrides.

Run:
    python tools/gnn_overlap_bias.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.repair_process import THRESH_OVERLAP, load_models, predict_feasibility  # noqa: E402

THR_MM = THRESH_OVERLAP * 1000.0
W_MM = 100.0
# Comfortably below the 20 mm thickness threshold, so the pair can only ever fail on overlap
# and the sweep isolates one criterion.
THICK_MM = 15.0


def _t(x, device):
    """Wrap a nested sequence as a float tensor on the given device."""
    return torch.tensor(np.asarray(x, dtype=np.float32), dtype=torch.float, device=device)


def make_pair(overlap_mm, device):
    """Build a vertically stacked pair whose x-overlap is the requested value."""
    a = W_MM / 1000.0
    t = THICK_MM / 1000.0
    size_anchor = _t([a, a, a], device)
    pos_anchor = _t([0.0, 0.0, a / 2.0], device)
    size_active = _t([a, a, t], device)
    # Both blocks are W_MM wide on x, so an offset of W - overlap leaves exactly the requested
    # overlap. y stays centred and is therefore never the binding axis.
    dx = (W_MM - overlap_mm) / 1000.0
    pos_active = _t([dx, 0.0, a + t / 2.0], device)
    return size_anchor, size_active, pos_anchor, pos_active


def main():
    device = torch.device("cpu")
    gnn_model, _ = load_models(device)
    gnn_model = gnn_model.to(device)

    targets = np.arange(4.0, 20.01, 1.0)
    preds = []
    for ov in targets:
        sa, sc, pa, pc = make_pair(ov, device)
        _, reg = predict_feasibility(sa, sc, pa, pc, gnn_model)
        preds.append(reg[0] * 1000.0)
    preds = np.array(preds)
    bias = preds - targets

    print(f"\noverlap threshold = {THR_MM:.0f} mm  (thickness held feasible)\n")
    print(f"  {'true[mm]':>9} {'pred[mm]':>9} {'bias[mm]':>9}")
    for tt, pr, b in zip(targets, preds, bias):
        mark = "  <-- threshold" if abs(tt - THR_MM) < 1e-6 else ""
        print(f"  {tt:>9.1f} {pr:>9.1f} {b:>+9.2f}{mark}")

    # Fit the slope on a narrow band around the threshold rather than the whole sweep: only the
    # local behaviour decides whether the gate is sharp.
    band = (targets >= 6) & (targets <= 14)
    slope = float(np.polyfit(targets[band], preds[band], 1)[0])
    bias10 = float(np.interp(THR_MM, targets, bias))
    cross = float(np.interp(THR_MM, preds, targets)) if preds[0] < THR_MM < preds[-1] else float("nan")
    print(f"\n  slope in [6,14] mm: {slope:.2f}")
    print(f"  bias@10mm: {bias10:+.2f} mm ({'over' if bias10>0 else 'UNDER'}-estimates)")
    if not np.isnan(cross):
        print(f"  GNN calls overlap feasible (>=10mm pred) once TRUE overlap >= ~{cross:.1f} mm")
        if cross < THR_MM - 1:
            print(f"  => OVER-optimistic: green-lights overlaps as low as {cross:.1f}mm (bluff risk).")
        elif cross > THR_MM + 1:
            print(f"  => conservative: rejects feasible overlaps up to {cross:.1f}mm.")
        else:
            print(f"  => sharp at the threshold; overlap is FINE near 10mm (big MAE is log-inflation).")


if __name__ == "__main__":
    main()
