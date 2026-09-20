"""Trace one repair step by step to see whether the latent optimiser can actually thin a beam.

Builds synthetic stacked pairs that fail for exactly one reason: a wide, well-aligned footprint
so overlap is never in question, and a contact-axis size above
``config.gnn.thresh_thickness_max``. Each is handed to the real batched repair with its
per-step trace hook enabled, so the run is the production one and not a reimplementation.

Three quantities are logged per step: the contact-axis size of the decoded box, which is the
truth; the thickness the surrogate predicts for the same code; and the feasibility probability.
Separating the first two is the whole purpose. If the decoded thickness falls below the
threshold the optimiser can genuinely thin the member. If it does not, the trace says why: a
prediction well below the decoded value is the surrogate bluffing, a curve flat from the first
step means no gradient reaches the thickness, and a dip that springs back means the drift
penalty or the projection is pulling against it.

Writes one PNG per case to ``results/repair_trace/trace_start_<N>mm.png``, thickness over steps
in the upper panel and the feasibility probability in the lower one.

Run:
    python tools/repair_thickness_trace.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.repair_optimizer import adam_repair_batched  # noqa: E402
from src.repair_process import load_models  # noqa: E402
from config.config import config  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUT_DIR = ROOT / "results" / "repair_trace"
OUT_DIR.mkdir(parents=True, exist_ok=True)
THRESH_TH_MM = float(config.gnn.thresh_thickness_max) * 1000.0
THRESH_OV_MM = float(config.gnn.thresh_overlap_min) * 1000.0

# All three start above the thickness threshold and have to be pushed below it. The spread is
# deliberate: a case just over the threshold and one several times over it fail differently.
START_THICKNESS_MM = [30.0, 50.0, 80.0]
# Both footprints are far wider than the overlap threshold and perfectly aligned, so overlap
# can never become the binding constraint and the trace isolates thickness.
TANGENTIAL_MM = 100.0
ANCHOR_MM = 100.0


def make_pair(thickness_mm: float, device: torch.device):
    """A cube resting on the table with a too-thick block on top; contact axis is z."""
    a = ANCHOR_MM / 1000.0
    t = thickness_mm / 1000.0
    w = TANGENTIAL_MM / 1000.0
    # Both blocks are placed by their centre, so resting on a surface means half the own
    # extent above it: the base sits on the table, the part under test on the base.
    size_anchor = torch.tensor([a, a, a], dtype=torch.float, device=device)
    pos_anchor = torch.tensor([0.0, 0.0, a / 2.0], dtype=torch.float, device=device)
    size_active = torch.tensor([w, w, t], dtype=torch.float, device=device)
    pos_active = torch.tensor([0.0, 0.0, a + t / 2.0], dtype=torch.float, device=device)
    return size_anchor, size_active, pos_anchor, pos_active


def run_case(thickness_mm, gnn_model, scale_factor, device):
    """Repair one synthetic pair with tracing on and return the trace and the final state."""
    size_anchor, size_active, pos_anchor, pos_active = make_pair(thickness_mm, device)
    trace: list[dict] = []
    out = adam_repair_batched(
        [size_anchor],
        [size_active],
        [pos_anchor],
        [pos_active],
        gnn_model,
        scale_factor,
        verbose=False,
        trace=trace,
    )
    final_size = out["size_active_out"][0].detach().cpu().numpy()
    final_reg = out["reg_out_m"][0]
    final_p = float(out["p_binary"][0])
    success = bool(out["success_mask"][0])
    # The repair determines the contact axis once before its loop and repeats it in every
    # trace entry, so the first step is enough. The fallback is the z axis these cases stack on.
    ca = trace[0]["contact_axis"] if trace else 2
    return trace, final_size, final_reg, final_p, success, ca


def summarise(thickness_mm, trace, final_size, final_reg, final_p, success, ca):
    """Print a few sampled steps and the verdict, and return the three logged series."""
    decoded = [r["thickness_decoded_mm"] for r in trace]
    pred = [r["thickness_pred_mm"] for r in trace]
    pb = [r["p_binary"] for r in trace]
    final_decoded = final_size[ca] * 1000.0
    print(f"\n===== START THICKNESS {thickness_mm:.0f} mm  (contact axis = {ca}) =====")
    print(f"  steps logged: {len(trace)}")
    print(f"  {'step':>6} {'decoded[mm]':>12} {'GNN-pred[mm]':>13} {'p_binary':>9}")
    n = len(trace)
    # The trace has one entry per optimisation step, far too many to print; quarters plus the
    # endpoints are enough to see the shape of the curve, and the plot carries the rest.
    idxs = sorted(set([0, n // 4, n // 2, 3 * n // 4, n - 1])) if n else []
    for i in idxs:
        r = trace[i]
        print(f"  {r['step']:>6} {r['thickness_decoded_mm']:>12.1f} "
              f"{r['thickness_pred_mm']:>13.1f} {r['p_binary']:>9.3f}")
    min_decoded = min(decoded) if decoded else float("nan")
    print(f"  --> final decoded thickness: {final_decoded:.1f} mm "
          f"(min reached: {min_decoded:.1f} mm)")
    print(f"  --> final GNN thickness/overlap: {final_reg[1]*1000:.1f} / "
          f"{final_reg[0]*1000:.1f} mm   p_binary={final_p:.3f}   success={success}")
    # The three outcomes the trace is meant to tell apart, decided on the last step: the
    # surrogate believing a thin member that is not there, a genuine repair, or no progress.
    if trace:
        gap = pred[-1] - decoded[-1]
        if decoded[-1] > THRESH_TH_MM and pred[-1] <= THRESH_TH_MM:
            print(f"  !! BLUFF: GNN predicts {pred[-1]:.1f} mm (<= thresh) but actual "
                  f"is {decoded[-1]:.1f} mm. gap={gap:.1f} mm")
        elif final_decoded <= THRESH_TH_MM:
            print(f"  OK: actual thickness {final_decoded:.1f} mm <= {THRESH_TH_MM:.0f} mm "
                  f"threshold — beam thinned successfully.")
        else:
            print(f"  STUCK: actual {final_decoded:.1f} mm still > {THRESH_TH_MM:.0f} mm.")
    return decoded, pred, pb


def plot_case(thickness_mm, decoded, pred, pb):
    """Write the two-panel trace figure for one case."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    steps = list(range(1, len(decoded) + 1))
    ax1.plot(steps, decoded, label="actual decoded thickness", lw=1.6)
    ax1.plot(steps, pred, label="GNN-predicted thickness", lw=1.2, ls="--")
    ax1.axhline(THRESH_TH_MM, color="green", ls=":", label=f"threshold ({THRESH_TH_MM:.0f} mm)")
    ax1.set_ylabel("thickness [mm]")
    ax1.set_title(f"Repair trace — start {thickness_mm:.0f} mm")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)
    ax2.plot(steps, pb, color="purple", lw=1.4, label="p_binary")
    ax2.axhline(0.5, color="grey", ls=":", label="commit gate 0.5")
    ax2.set_ylabel("p_binary")
    ax2.set_xlabel("Adam step")
    ax2.set_ylim(-0.05, 1.05)
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    out = OUT_DIR / f"trace_start_{int(thickness_mm)}mm.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"  -> plot: {out}")


def main():
    device = torch.device("cpu")
    gnn_model, scale_factor = load_models(device)
    gnn_model = gnn_model.to(device)
    print(f"scale_factor={scale_factor}  thickness threshold={THRESH_TH_MM:.0f} mm  "
          f"overlap threshold={THRESH_OV_MM:.0f} mm")

    for t_mm in START_THICKNESS_MM:
        trace, final_size, final_reg, final_p, success, ca = run_case(
            t_mm, gnn_model, scale_factor, device
        )
        decoded, pred, pb = summarise(
            t_mm, trace, final_size, final_reg, final_p, success, ca
        )
        plot_case(t_mm, decoded, pred, pb)


if __name__ == "__main__":
    main()
