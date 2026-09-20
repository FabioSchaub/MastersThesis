"""Check how a surrogate predicts across the range of each quantity, band by band.

Complements ``analyze_paper_v0.py`` and answers a narrower question: within each band of the
target range, does the model track the quantity or only order it? That is what the fitted
slope per band says, and a slope well below one near a limit means a gradient the repair
cannot follow, however good the overall error looks.

Unlike the other analysis, this one works from the table directly rather than from prepared
graphs, and reads the model's shape out of the checkpoint instead of the configuration, so a
run with different widths can be inspected without editing anything.

    python src/label_analysis/analyze_label_generalization.py

Prints per-band statistics and writes one figure per quantity into ``results/label_analysis/``.
``GNN_MODEL`` selects the checkpoint, ``GNN_SPLIT`` the split and ``GNN_DATA_FILE`` the table.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from config.config import config  # noqa: E402
from src.gnn import GNN  # noqa: E402
from src.simulation_dataset import (  # noqa: E402
    get_scale_factor,
    manipulate_data,
    read_txt_file,
)

DATA_FOLDER = ROOT / config.data.data_folder
GNN_MODEL_FOLDER = ROOT / config.gnn.gnn_folder

DATA_FILE = os.environ.get("GNN_DATA_FILE", config.data.data_file)
BATCH_SIZE = 4096
OUT_DIR = ROOT / "results" / "label_analysis"

# The checkpoint the results of Part I were produced with; ``GNN_MODEL`` overrides it.
PLACEHOLDER_GNN_NAME: str = "gnn_small-mixed-10mm_node5_batchsize2048_20260519-092003.pth"
MODEL_NAME = os.environ.get("GNN_MODEL", PLACEHOLDER_GNN_NAME)


def _find_latest_gnn() -> str:
    """Most recent checkpoint in ``gnn_models/``, preferring the more specific name patterns."""
    candidates = sorted(
        GNN_MODEL_FOLDER.glob("gnn_paramter-optimization_*.pth"),
        key=lambda p: p.stat().st_mtime,
    )
    if not candidates:
        candidates = sorted(
            GNN_MODEL_FOLDER.glob("gnn_*node5*.pth"),
            key=lambda p: p.stat().st_mtime,
        )
    if not candidates:
        candidates = sorted(
            GNN_MODEL_FOLDER.glob("gnn_*.pth"),
            key=lambda p: p.stat().st_mtime,
        )
    if not candidates:
        raise SystemExit(
            f"No GNN checkpoint found in {GNN_MODEL_FOLDER}. "
            f"Expected something like {PLACEHOLDER_GNN_NAME}."
        )
    return candidates[-1].name


# Held-out by default. The other two splits are for inspecting the fit itself, and any number
# read off them is in-sample.
SPLIT = os.environ.get("GNN_SPLIT", "test").lower()

LABEL_TO_COLUMN = {
    "overlap": "Block1_Overlap_m",
    "thickness": "Block1_Thickness_m",
}
THRESH = {
    "overlap": float(config.gnn.thresh_overlap_min),
    "thickness": float(config.gnn.thresh_thickness_max),
}
OVERLAP_LOG_EPS = 1e-5  # must equal gnn_dataset_preparation.OVERLAP_LOG_EPS

LABELS: list[str] = []
TRUE_COLS: list[str] = []


def _fmt_mm(v_m: float) -> str:
    """Format a length given in metres as millimetres."""
    return f"{v_m * 1000.0:.3f} mm"


def load_gnn(device: torch.device) -> tuple[GNN, torch.Tensor, torch.Tensor, str]:
    """Restore a checkpoint, reading its widths off the weights rather than the configuration.

    Returns:
        The model in evaluation mode, the standardisation from the checkpoint, and the file
        name it came from.

    Raises:
        RuntimeError: If the standardisation is missing, or has a width other than two. Both
            mean the checkpoint does not match what this script knows how to read, and
            de-standardising it would produce plausible-looking nonsense.
    """
    name = MODEL_NAME or _find_latest_gnn()
    ckpt_path = GNN_MODEL_FOLDER / name
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt.get("model", ckpt)

    node_dim = int(state["gat1.lin_l.weight"].shape[1])
    edge_dim = int(state["gat1.lin_edge.weight"].shape[1])
    heads = int(state["gat1.att"].shape[1])
    hidden_per_head = int(state["gat1.att"].shape[2])
    hidden_dim = int(heads * hidden_per_head)

    # Removed from the state before loading, because they are registered on the instance
    # separately; the forward pass does not read them, this script does.
    state = dict(state)
    target_mean = state.pop("target_mean", None)
    target_std = state.pop("target_std", None)
    if target_mean is None or target_std is None:
        raise RuntimeError(
            "Checkpoint missing target_mean/target_std buffers. "
            "Re-train with the parameter-optimization gnn_training.py."
        )

    if target_mean.shape[0] != 2:
        raise RuntimeError(
            f"Expected target_mean of shape (2,) for [overlap, thickness] in "
            f"the 164136-revert state; got {tuple(target_mean.shape)}. "
            f"Old multi-head (5-target) checkpoint? Switch architectures or "
            f"point GNN_MODEL to a 2-target checkpoint."
        )

    gnn = GNN(
        node_dim=node_dim,
        edge_dim=edge_dim,
        hidden_dim=hidden_dim,
        heads=heads,
        dropout=config.gnn.dropout,
    )
    gnn.register_buffer("target_mean", target_mean.clone())
    gnn.register_buffer("target_std", target_std.clone())
    gnn.load_state_dict(state, strict=False)
    gnn.eval().to(device)

    print(f"Loaded GNN checkpoint: {ckpt_path.name}")
    print(
        f"  architecture: node_dim={node_dim}, edge_dim={edge_dim}, "
        f"hidden_dim={hidden_dim}, heads={heads}"
    )
    print(f"  target_mean: {target_mean.cpu().numpy()}")
    print(f"  target_std:  {target_std.cpu().numpy()}")
    return gnn, target_mean.to(device), target_std.to(device), name


def build_batch_graph_inputs(
    size0_scaled: torch.Tensor,  # (B, 3)
    size1_scaled: torch.Tensor,  # (B, 3)
    pos0_scaled: torch.Tensor,  # (B, 3)
    pos1_scaled: torch.Tensor,  # (B, 3)
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Assemble the surrogate's inputs for a batch of pairs, built here rather than per graph.

    Args:
        size0_scaled, size1_scaled: Edge lengths of the two blocks, shape ``(B, 3)`` each.
        pos0_scaled, pos1_scaled: Their centres, shape ``(B, 3)`` each.

    Returns:
        Node features of shape ``(2 * B, 5)`` with the two nodes of a pair adjacent, the edge
        list, and edge features of shape ``(2 * B, 6)``.
    """
    bsz = size0_scaled.shape[0]
    dev = size0_scaled.device

    type0 = torch.tensor([1.0, 0.0], device=dev).unsqueeze(0).expand(bsz, -1)
    type1 = torch.tensor([0.0, 1.0], device=dev).unsqueeze(0).expand(bsz, -1)
    node0 = torch.cat([size0_scaled, type0], dim=1)
    node1 = torch.cat([size1_scaled, type1], dim=1)
    # Interleaved, not concatenated: the model reads the part under test off the odd indices.
    x = torch.stack([node0, node1], dim=1).reshape(2 * bsz, -1)

    base = torch.arange(bsz, device=dev, dtype=torch.long) * 2
    edge_index = torch.stack(
        [torch.cat([base, base + 1]), torch.cat([base + 1, base])],
        dim=0,
    )

    delta = pos1_scaled - pos0_scaled
    dx = delta[:, 0]
    dy = delta[:, 1]
    dz = delta[:, 2]
    eps = 1e-8
    dist_xy = torch.sqrt(dx**2 + dy**2 + eps).unsqueeze(1)
    dist_xz = torch.sqrt(dx**2 + dz**2 + eps).unsqueeze(1)
    dist_yz = torch.sqrt(dy**2 + dz**2 + eps).unsqueeze(1)

    edge_fwd = torch.cat([delta, dist_xy, dist_xz, dist_yz], dim=1)
    edge_bwd = torch.cat([-delta, dist_xy, dist_xz, dist_yz], dim=1)
    edge_attr = torch.cat([edge_fwd, edge_bwd], dim=0)

    return x, edge_index, edge_attr


def _fit_line(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Least-squares slope and intercept, or not-a-number for fewer than two points."""
    if x.size < 2:
        return float("nan"), float("nan")
    a, b = np.polyfit(x.astype(np.float64), y.astype(np.float64), 1)
    return float(a), float(b)


def _bucket_edges_label(label: str) -> tuple[list[float], list[str], bool]:
    """Band edges in millimetres, and their names, for one quantity.

    They follow the bands ``gnn_training`` reports on, so the two outputs can be compared.
    """
    if label == "overlap":
        return (
            [0.0, 8.0, 16.0, 20.0, 30.0, float("inf")],
            ["[0,8)", "[8,16)", "[16,20)", "[20,30)", "[30,inf)"],
            False,
        )
    return (
        [0.0, 16.0, 20.0, 24.0, 30.0, float("inf")],
        ["[0,16)", "[16,20)", "[20,24)", "[24,30)", "[30,inf)"],
        False,
    )


def _bucketize(values_m: np.ndarray, label: str) -> tuple[np.ndarray, list[str]]:
    """Assign each value to a band, returning the per-value names and the band order."""
    edges_mm, names, abs_mode = _bucket_edges_label(label)
    x_mm = np.abs(values_m) * 1000.0 if abs_mode else values_m * 1000.0
    bucket_idx = np.digitize(
        x_mm, bins=np.array(edges_mm[1:-1], dtype=float), right=False
    )
    bucket_names = np.array([names[i] for i in bucket_idx], dtype=object)
    return bucket_names, names


def _save_scatter(
    label: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    bucket_names: np.ndarray,
    order: list[str],
    model_filename: str,
) -> None:
    """Draw predicted against true for one quantity, one panel per band plus the whole split."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    safe_name = model_filename.replace("/", "_").replace("<", "_").replace(">", "_")
    out_path = OUT_DIR / f"{label}_true_vs_pred_buckets_{safe_name}.png"

    y_true_mm = y_true * 1000.0
    y_pred_mm = y_pred * 1000.0

    fig, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
    axes_flat = axes.ravel()

    lo = float(min(y_true_mm.min(), y_pred_mm.min()))
    hi = float(max(y_true_mm.max(), y_pred_mm.max()))
    pad = 0.05 * (hi - lo + 1e-9)
    lo, hi = lo - pad, hi + pad

    for i, b in enumerate(order):
        ax = axes_flat[i]
        m = bucket_names == b
        t = y_true_mm[m]
        p = y_pred_mm[m]

        ax.scatter(t, p, s=6, alpha=0.35, color="#1f77b4", edgecolors="none")
        ax.plot([lo, hi], [lo, hi], "k--", linewidth=1.0)

        if t.size >= 2:
            a, c = _fit_line(t, p)
            x_line = np.array([lo, hi], dtype=np.float64)
            ax.plot(x_line, a * x_line + c, color="#d62728", linewidth=1.4)
            ax.set_title(f"{b} | n={t.size} | slope={a:.3f}")
        else:
            ax.set_title(f"{b} | n={t.size}")

        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_xlabel(f"True {label} (mm)")
        ax.set_ylabel(f"Pred {label} (mm)")
        ax.grid(alpha=0.2)

    ax_all = axes_flat[5]
    ax_all.scatter(
        y_true_mm, y_pred_mm, s=4, alpha=0.25, color="#2ca02c", edgecolors="none"
    )
    ax_all.plot([lo, hi], [lo, hi], "k--", linewidth=1.0)
    a_all, c_all = _fit_line(y_true_mm, y_pred_mm)
    x_line = np.array([lo, hi], dtype=np.float64)
    ax_all.plot(x_line, a_all * x_line + c_all, color="#d62728", linewidth=1.4)
    ax_all.set_title(f"ALL | n={y_true.size} | slope={a_all:.3f}")
    ax_all.set_xlim(lo, hi)
    ax_all.set_ylim(lo, hi)
    ax_all.set_xlabel(f"True {label} (mm)")
    ax_all.set_ylabel(f"Pred {label} (mm)")
    ax_all.grid(alpha=0.2)

    fig.suptitle(f"True vs Predicted {label} by bucket", fontsize=14)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    print(f"Saved plot: {out_path}")


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    df = read_txt_file(DATA_FOLDER, DATA_FILE)
    if df.empty:
        print("Dataset empty or unreadable.")
        return
    df = manipulate_data(df)

    global LABELS, TRUE_COLS
    LABELS = []
    TRUE_COLS = []
    for label, col in LABEL_TO_COLUMN.items():
        if col in df.columns:
            LABELS.append(label)
            TRUE_COLS.append(col)
    if not LABELS:
        raise KeyError(
            f"No labels found in dataset. Expected any of: {list(LABEL_TO_COLUMN.keys())}"
        )
    print(f"Detected labels in dataset: {LABELS}")

    required = [
        "Block0_SizeX",
        "Block0_SizeY",
        "Block0_SizeZ",
        "Block1_SizeX",
        "Block1_SizeY",
        "Block1_SizeZ",
        "Block0_PosX",
        "Block0_PosY",
        "Block0_PosZ",
        "Block1_PosX",
        "Block1_PosY",
        "Block1_PosZ",
        *TRUE_COLS,
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    # Must match what the checkpoint was trained on, see prepare_simulation_dataset.
    scale_factor: float = 1.0
    print(f"Scale factor: {scale_factor:.4f} (raw metres, matches 164136 ckpt)")

    gnn, target_mean, target_std, model_filename = load_gnn(device)

    # Same seed and same ratios as split_graphs, so this reproduces the held-out set the
    # surrogate was never fitted on. Diverging here would quietly report an in-sample number.
    torch.manual_seed(config.general.random_seed)
    n_all = len(df)
    perm = torch.randperm(n_all).tolist()
    train_end = int(config.training.train_split * n_all)
    val_end = int((config.training.train_split + config.training.val_split) * n_all)

    if SPLIT == "train":
        indices = perm[:train_end]
    elif SPLIT == "val":
        indices = perm[train_end:val_end]
    elif SPLIT == "test":
        indices = perm[val_end:]
    else:
        raise ValueError(
            f"Unknown GNN_SPLIT={SPLIT!r}; expected 'train', 'val', or 'test'."
        )
    eval_df = df.iloc[indices].reset_index(drop=True)
    print(f"Evaluation split: {SPLIT!r}, size: {len(eval_df)}")

    size0 = eval_df[["Block0_SizeX", "Block0_SizeY", "Block0_SizeZ"]].to_numpy(
        np.float32
    )
    size1 = eval_df[["Block1_SizeX", "Block1_SizeY", "Block1_SizeZ"]].to_numpy(
        np.float32
    )
    pos0 = eval_df[["Block0_PosX", "Block0_PosY", "Block0_PosZ"]].to_numpy(np.float32)
    pos1 = eval_df[["Block1_PosX", "Block1_PosY", "Block1_PosZ"]].to_numpy(np.float32)

    y_true = eval_df[TRUE_COLS].to_numpy(np.float32)

    size0_s = (size0 * scale_factor).astype(np.float32)
    size1_s = (size1 * scale_factor).astype(np.float32)
    pos0_s = (pos0 * scale_factor).astype(np.float32)
    pos1_s = (pos1 * scale_factor).astype(np.float32)

    y_pred = np.zeros((len(eval_df), len(LABELS)), dtype=np.float32)

    overlap_idx_model = 0
    thickness_idx_model = 1

    with torch.no_grad():
        for start in range(0, len(eval_df), BATCH_SIZE):
            end = min(start + BATCH_SIZE, len(eval_df))

            s0 = torch.from_numpy(size0_s[start:end]).to(device)
            s1 = torch.from_numpy(size1_s[start:end]).to(device)
            p0 = torch.from_numpy(pos0_s[start:end]).to(device)
            p1 = torch.from_numpy(pos1_s[start:end]).to(device)

            x, edge_index, edge_attr = build_batch_graph_inputs(s0, s1, p0, p1)

            reg_out_std, _binary_logit, *_ = gnn(x, edge_index, edge_attr)

            reg_out = reg_out_std * target_std + target_mean
            reg_out = reg_out.clone()
            if target_mean[0].item() < 0.0:
                reg_out[:, 0] = torch.exp(reg_out[:, 0]) - OVERLAP_LOG_EPS

            reg_out_np = reg_out.cpu().numpy()
            for j, label in enumerate(LABELS):
                model_idx = (
                    overlap_idx_model if label == "overlap" else thickness_idx_model
                )
                y_pred[start:end, j] = reg_out_np[:, model_idx]

    for i, label in enumerate(LABELS):
        t = y_true[:, i]
        p = y_pred[:, i]
        e = p - t

        mae = float(np.abs(e).mean())
        rmse = float(np.sqrt(np.mean(e**2)))
        corr = float(np.corrcoef(t, p)[0, 1]) if t.size > 1 else float("nan")
        slope, intercept = _fit_line(t, p)

        print(f"\n=== {label.upper()} ===")
        print(f"MAE: {_fmt_mm(mae)}")
        print(f"RMSE: {_fmt_mm(rmse)}")
        print(f"Corr: {corr:.4f}")
        print(f"Fit: pred = {slope:.4f} * true + {intercept * 1000.0:+.3f} mm")

        # The two criteria point in opposite directions: overlap has to reach its limit,
        # thickness has to stay under it.
        thr = THRESH[label]
        if label == "overlap":
            true_ok = t >= thr
            pred_ok = p >= thr
        else:
            true_ok = t < thr
            pred_ok = p < thr

        agree = float((true_ok == pred_ok).mean())
        print(f"Threshold agreement @ {_fmt_mm(thr)}: {100.0 * agree:.2f}%")

        bnames, border = _bucketize(t, label)
        print("Bucket stats:")
        for b in border:
            m = bnames == b
            n_b = int(m.sum())
            if n_b == 0:
                print(f"  {b:>10}: n=0")
                continue
            tb = t[m]
            pb = p[m]
            eb = pb - tb
            b_mae = float(np.abs(eb).mean())
            b_rmse = float(np.sqrt(np.mean(eb**2)))
            b_corr = float(np.corrcoef(tb, pb)[0, 1]) if n_b > 1 else float("nan")
            b_slope, b_int = _fit_line(tb, pb)
            print(
                f"  {b:>10}: n={n_b:6d} | MAE={_fmt_mm(b_mae)} | "
                f"RMSE={_fmt_mm(b_rmse)} | "
                f"Slope={b_slope:7.3f} | Int={b_int * 1000.0:+.3f} mm | "
                f"Corr={b_corr:7.4f}"
            )

        _save_scatter(label, t, p, bnames, border, model_filename)


if __name__ == "__main__":
    main()
