"""Report how well a trained surrogate predicts, on a split it was not fitted on.

This is where the surrogate figures quoted for Part I come from. A checkpoint is loaded, the
split is reconstructed from the same seed the training used, and the model is scored twice
over: as a regressor on the two quantities, and as a classifier of feasibility.

Feasibility is scored three ways on purpose. Derived from the two predicted quantities against
their limits, which is what the repair actually relies on; taken from the feasibility head
directly; and per failure mode. Where these disagree is informative -- a model can be a good
classifier and a poor regressor near a limit, and only the second matters for a repair -- so
the agreement between the first two is reported as well.

The regression numbers are stratified by how far the target sits from its threshold, since an
average over the whole range would hide exactly the region the repair operates in.

    python src/label_analysis/analyze_paper_v0.py

Prints the tables and writes a calibration plot and a residual histogram per quantity into
``results/paper_v0_analysis/``. ``GNN_MODEL`` selects the checkpoint, ``GNN_SPLIT`` the split,
and ``ANALYSIS_SEED`` the seed the split is drawn with.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch_geometric.loader import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from config.config import config  # noqa: E402
from src.gnn import GNN  # noqa: E402
from src.gnn_dataset_preparation import (  # noqa: E402
    build_graph_dataset,
    compute_target_stats,
    split_graphs,
)
from src.gnn_training import random_seed  # noqa: E402

GNN_MODEL_FOLDER = ROOT / config.gnn.gnn_folder
OUT_DIR = ROOT / "results" / "paper_v0_analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)

LABELS = ["overlap", "thickness"]
THRESH = {
    "overlap": float(config.gnn.thresh_overlap_min),
    "thickness": float(config.gnn.thresh_thickness_max),
}

# Bands of the target range the error is reported over, laid out around the limits of the
# configuration so that the region the repair works in is resolved.
OVERLAP_BUCKETS = [
    (0.000, 0.005, "0_5mm_fail_far"),
    (0.005, 0.010, "5_10mm_fail_near"),
    (0.010, 0.015, "10_15mm_boundary"),
    (0.015, 0.025, "15_25mm_far"),
    (0.025, float("inf"), "25mm_inf"),
]
THICKNESS_BUCKETS = [
    (0.000, 0.016, "0_16mm_far"),
    (0.016, 0.020, "16_20mm_boundary"),
    (0.020, 0.024, "20_24mm_fail_near"),
    (0.024, float("inf"), "24mm_inf_fail_far"),
]


def find_paper_v0_checkpoint() -> Path:
    """Most recent checkpoint in ``gnn_models/``, used when none is named."""
    candidates = sorted(
        GNN_MODEL_FOLDER.glob("gnn_paper-v0_*.pth"),
        key=lambda p: p.stat().st_mtime,
    )
    if not candidates:
        candidates = sorted(
            GNN_MODEL_FOLDER.glob("gnn_*.pth"),
            key=lambda p: p.stat().st_mtime,
        )
    if not candidates:
        raise SystemExit(f"No GNN checkpoint found in {GNN_MODEL_FOLDER}")
    return candidates[-1]


def load_model(ckpt_path: Path, target_stats: dict, device: torch.device) -> GNN:
    """Restore a checkpoint, standardising with statistics recomputed from the training split."""
    model = GNN(
        node_dim=config.gnn.node_dim,
        edge_dim=config.gnn.edge_dim,
        hidden_dim=config.gnn.hidden_dim,
        heads=config.gnn.heads,
        dropout=config.gnn.dropout,
    ).to(device)
    model.register_buffer("target_mean", target_stats["mean"].to(device))
    model.register_buffer("target_std", target_stats["std"].to(device))
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


def predict(model: GNN, dataloader: DataLoader, device: torch.device):
    """Run the model over a split and collect everything the report needs.

    Returns:
        The predicted and the true overlap and thickness in metres, both ``(N, 2)``; the
        feasibility probability ``(N,)``; the two failure mode probabilities ``(N, 2)``; and
        the feasibility label ``(N,)``.
    """
    OVERLAP_LOG_EPS = 1e-5
    reg_preds, reg_targets, p_binary, p_modes, y_good = [], [], [], [], []
    target_mean = model.target_mean
    target_std = model.target_std
    log_space_overlap = bool(target_mean[0].item() < 0.0)
    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            # Checkpoints from earlier revisions of the model return a different number of
            # heads; only the first three are read, so both load.
            out = model(batch.x, batch.edge_index, batch.edge_attr)
            if len(out) == 4:
                pred_std, bin_logit, mode_logits, _contact_logits = out
            else:
                pred_std, bin_logit, mode_logits = out
            pred = pred_std * target_std + target_mean
            if log_space_overlap:
                pred = pred.clone()
                pred[:, 0] = torch.exp(pred[:, 0]) - OVERLAP_LOG_EPS
            reg_preds.append(pred[:, :2].cpu().numpy())
            reg_targets.append(batch.y.squeeze(1)[:, :2].cpu().numpy())
            p_binary.append(torch.sigmoid(bin_logit).cpu().numpy())
            p_modes.append(torch.sigmoid(mode_logits).cpu().numpy())
            y_good.append(batch.y_good.cpu().numpy())
    return (
        np.concatenate(reg_preds),
        np.concatenate(reg_targets),
        np.concatenate(p_binary).ravel(),
        np.concatenate(p_modes),
        np.concatenate(y_good).ravel(),
    )


def overall_stats(pred: np.ndarray, target: np.ndarray) -> dict:
    """Error, fit and spread of one quantity over a whole split, with lengths in millimetres.

    The slope and the ratio of the two spreads are reported alongside the error because a
    model that hedges towards the mean can still have a respectable mean error.
    """
    err = pred - target
    abs_err = np.abs(err)
    mae = float(abs_err.mean())
    rmse = float(np.sqrt((err ** 2).mean()))
    if target.std() > 1e-9:
        slope = float(np.cov(pred, target, bias=True)[0, 1] / target.var())
        intercept = float(pred.mean() - slope * target.mean())
        ss_res = float(((pred - target) ** 2).sum())
        ss_tot = float(((target - target.mean()) ** 2).sum())
        r2 = 1.0 - ss_res / max(ss_tot, 1e-12)
        corr = float(np.corrcoef(pred, target)[0, 1])
    else:
        slope = intercept = r2 = corr = float("nan")
    std_ratio = float(pred.std() / max(target.std(), 1e-12))
    return {
        "n": int(len(target)),
        "mae_mm": mae * 1000,
        "rmse_mm": rmse * 1000,
        "slope": slope,
        "intercept_mm": intercept * 1000,
        "r2": r2,
        "corr": corr,
        "std_ratio": std_ratio,
        "bias_mm": float(err.mean()) * 1000,
    }


def stratified_bucket_stats(pred: np.ndarray, target: np.ndarray, buckets: list):
    """The same statistics per band of the target range. Thin bands are reported as absent."""
    rows = []
    for lo, hi, tag in buckets:
        m = (target >= lo) & (target < hi)
        n = int(m.sum())
        if n < 50:
            rows.append({"bucket": tag, "n": n, "mae_mm": float("nan"),
                         "slope": float("nan"), "bias_mm": float("nan")})
            continue
        p, t = pred[m], target[m]
        err = p - t
        mae = float(np.abs(err).mean()) * 1000
        slope = (
            float(np.cov(p, t, bias=True)[0, 1] / t.var()) if t.std() > 1e-9
            else float("nan")
        )
        rows.append({"bucket": tag, "n": n, "mae_mm": mae,
                     "slope": slope, "bias_mm": float(err.mean()) * 1000})
    return rows


def classifier_stats(true: np.ndarray, pred: np.ndarray) -> dict:
    """Classification scores with the full confusion matrix, since the classes are unbalanced."""
    cm = confusion_matrix(true, pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return {
        "f1": float(f1_score(true, pred, zero_division=0)),
        "precision": float(precision_score(true, pred, zero_division=0)),
        "recall": float(recall_score(true, pred, zero_division=0)),
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
    }


def plot_calibration(pred, target, name, out_path, threshold):
    """Scatter predicted against true, with the diagonal and the threshold drawn in."""
    target_mm = target * 1000
    pred_mm = pred * 1000
    fig, ax = plt.subplots(figsize=(6, 6))
    if len(target) > 10000:
        idx = np.random.default_rng(0).choice(len(target), 10000, replace=False)
        ax.scatter(target_mm[idx], pred_mm[idx], s=2, alpha=0.25, color="#3b82f6")
    else:
        ax.scatter(target_mm, pred_mm, s=2, alpha=0.25, color="#3b82f6")
    lo = min(float(target_mm.min()), float(pred_mm.min()))
    hi = max(float(target_mm.max()), float(pred_mm.max()))
    ax.plot([lo, hi], [lo, hi], color="black", linestyle="--", linewidth=1, label="y=x")
    t_mm = threshold * 1000
    ax.axvline(t_mm, color="red", linestyle=":", linewidth=1, label=f"thresh={t_mm:.1f} mm")
    ax.axhline(t_mm, color="red", linestyle=":", linewidth=1)
    ax.set_xlabel(f"true {name} (mm)")
    ax.set_ylabel(f"predicted {name} (mm)")
    ax.set_title(f"{name}: predicted vs true")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_residuals(pred, target, name, out_path):
    """Histogram of the residual in millimetres, with its mean marked."""
    err_mm = (pred - target) * 1000
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(err_mm, bins=80, color="#3b82f6", alpha=0.85)
    ax.axvline(0, color="black", linewidth=1)
    ax.axvline(err_mm.mean(), color="red", linestyle="--", linewidth=1,
               label=f"mean={err_mm.mean():+.3f} mm")
    ax.set_xlabel(f"residual {name} = pred − true (mm)")
    ax.set_ylabel("count")
    ax.set_title(f"{name}: residual histogram")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main():
    seed = int(os.environ.get("ANALYSIS_SEED", config.general.random_seed or 42))
    random_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Building graph dataset...")
    graphs = build_graph_dataset(device=str(device))
    train_g, val_g, test_g = split_graphs(graphs)

    split = os.environ.get("GNN_SPLIT", "test").lower()
    eval_g = {"train": train_g, "val": val_g, "test": test_g}[split]
    print(f"Evaluating on '{split}' split: {len(eval_g)} graphs")

    # Recomputed from the training split rather than read from the checkpoint. Both routes
    # agree as long as the dataset and the seed are the ones the run used.
    target_stats = compute_target_stats(train_g)

    ckpt_name = os.environ.get("GNN_MODEL")
    ckpt_path = (
        GNN_MODEL_FOLDER / ckpt_name if ckpt_name else find_paper_v0_checkpoint()
    )
    print(f"Loading checkpoint: {ckpt_path.name}")
    model = load_model(ckpt_path, target_stats, device)

    loader = DataLoader(eval_g, batch_size=4096)
    print("Running inference...")
    reg_pred, reg_target, p_binary, p_modes, y_good = predict(model, loader, device)
    print(f"  predictions shape: {reg_pred.shape}")

    print("\n" + "=" * 84)
    print(f" PAPER-V0 GNN ANALYSIS")
    print(f" Checkpoint: {ckpt_path.name}")
    print(f" Split: {split}  |  N = {len(reg_target)}")
    print("=" * 84)

    # --- per-target regression quality
    print("\n--- Per-target regression quality (units in mm) ---")
    print(f"{'target':<12} {'n':>7} {'MAE':>8} {'RMSE':>8} {'slope':>7} "
          f"{'intercept':>10} {'R²':>7} {'corr':>6} {'bias':>9} {'std_ratio':>10}")
    for i, name in enumerate(LABELS):
        s = overall_stats(reg_pred[:, i], reg_target[:, i])
        print(f"{name:<12} {s['n']:>7} {s['mae_mm']:>8.4f} {s['rmse_mm']:>8.4f} "
              f"{s['slope']:>7.3f} {s['intercept_mm']:>+10.4f} {s['r2']:>7.3f} "
              f"{s['corr']:>6.3f} {s['bias_mm']:>+9.4f} {s['std_ratio']:>10.3f}")

    # --- stratified
    print("\n--- Overlap stratified by target bucket ---")
    print(f"  {'bucket':<22} {'n':>7} {'MAE(mm)':>9} {'slope':>7} {'bias(mm)':>10}")
    for r in stratified_bucket_stats(reg_pred[:, 0], reg_target[:, 0], OVERLAP_BUCKETS):
        slope_str = f"{r['slope']:>7.3f}" if not np.isnan(r['slope']) else "      —"
        bias_str = f"{r['bias_mm']:>+10.4f}" if not np.isnan(r['bias_mm']) else "         —"
        mae_str = f"{r['mae_mm']:>9.4f}" if not np.isnan(r['mae_mm']) else "        —"
        print(f"  {r['bucket']:<22} {r['n']:>7} {mae_str} {slope_str} {bias_str}")

    print("\n--- Thickness stratified by target bucket ---")
    print(f"  {'bucket':<22} {'n':>7} {'MAE(mm)':>9} {'slope':>7} {'bias(mm)':>10}")
    for r in stratified_bucket_stats(reg_pred[:, 1], reg_target[:, 1], THICKNESS_BUCKETS):
        slope_str = f"{r['slope']:>7.3f}" if not np.isnan(r['slope']) else "      —"
        bias_str = f"{r['bias_mm']:>+10.4f}" if not np.isnan(r['bias_mm']) else "         —"
        mae_str = f"{r['mae_mm']:>9.4f}" if not np.isnan(r['mae_mm']) else "        —"
        print(f"  {r['bucket']:<22} {r['n']:>7} {mae_str} {slope_str} {bias_str}")

    # --- classification
    print("\n--- Feasibility classification ---")
    thresh_pred = (
        (reg_pred[:, 0] >= THRESH["overlap"])
        & (reg_pred[:, 1] <= THRESH["thickness"])
    ).astype(int)
    bin_pred = (p_binary > 0.5).astype(int)

    blocks = [
        ("threshold-derived (regression heads)", thresh_pred, y_good.astype(int)),
        ("binary-head (p_binary > 0.5)", bin_pred, y_good.astype(int)),
        ("mode/overlap_fail (p > 0.5)",
         (p_modes[:, 0] > 0.5).astype(int),
         (reg_target[:, 0] < THRESH["overlap"]).astype(int)),
        ("mode/thickness_fail (p > 0.5)",
         (p_modes[:, 1] > 0.5).astype(int),
         (reg_target[:, 1] > THRESH["thickness"]).astype(int)),
    ]
    for name, pred, true in blocks:
        m = classifier_stats(true, pred)
        print(f"\n  [{name}]")
        print(f"    F1 = {m['f1']:.4f}   Precision = {m['precision']:.4f}   "
              f"Recall = {m['recall']:.4f}")
        print(f"    Confusion (rows=true, cols=pred):")
        print(f"               pred=0   pred=1")
        print(f"      true=0  {m['TN']:>7}  {m['FP']:>7}")
        print(f"      true=1  {m['FN']:>7}  {m['TP']:>7}")

    # --- agreement between threshold-pred and binary-pred
    print("\n--- Agreement: threshold-derived vs binary-head ---")
    agree = (thresh_pred == bin_pred).mean()
    print(f"  Agreement rate: {agree:.4f}")
    print(f"  Disagreement (thresh=feasible, binary=infeasible): "
          f"{int(((thresh_pred == 1) & (bin_pred == 0)).sum())}")
    print(f"  Disagreement (thresh=infeasible, binary=feasible): "
          f"{int(((thresh_pred == 0) & (bin_pred == 1)).sum())}")

    # The two mode heads are independent, so a configuration that fails both is not a class
    # of its own to the model; it is recovered here by combining the two.
    print("\n--- 4-way fail-class breakdown (true vs predicted) ---")
    true_ov = reg_target[:, 0] < THRESH["overlap"]
    true_th = reg_target[:, 1] > THRESH["thickness"]
    pred_ov = p_modes[:, 0] > 0.5
    pred_th = p_modes[:, 1] > 0.5

    def class_id(ov, th):
        """Combine the two failure flags into one of four classes: none, one, the other, both."""
        return (ov & ~th).astype(int) * 1 + (~ov & th).astype(int) * 2 + (ov & th).astype(int) * 3

    true_c = class_id(true_ov, true_th)
    pred_c = class_id(pred_ov, pred_th)
    names = ["NONE", "OVERLAP", "THICK", "BOTH"]
    header_label = "true | pred"
    print(f"  {header_label:>14s}  " + "  ".join(f"{n:>8s}" for n in names) + "    total")
    for ti in range(4):
        row = [int(((true_c == ti) & (pred_c == pj)).sum()) for pj in range(4)]
        total = sum(row)
        print(f"  {names[ti]:>14s}  " + "  ".join(f"{c:>8d}" for c in row) + f"  {total:>8d}")
    # Per-class recall / precision
    print(f"\n  Per-class recall (= P(pred=c | true=c)):")
    for ci, n in enumerate(names):
        nt = int((true_c == ci).sum())
        if nt > 0:
            rec = int(((true_c == ci) & (pred_c == ci)).sum()) / nt
            print(f"    {n:>10s}: {rec:.4f}  (n_true={nt})")

    # --- plots
    print("\n--- Saving plots ---")
    for i, name in enumerate(LABELS):
        plot_calibration(
            reg_pred[:, i], reg_target[:, i], name,
            OUT_DIR / f"calibration_{name}_{split}.png",
            threshold=THRESH[name],
        )
        plot_residuals(
            reg_pred[:, i], reg_target[:, i], name,
            OUT_DIR / f"residuals_{name}_{split}.png",
        )
        print(f"  {name}: calibration + residuals")
    print(f"\nResults saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
