"""Training of the feasibility surrogate on the latent graph.

Reads the labelled two-block dataset through ``src/gnn_dataset_preparation.py``, which encodes
every block into a latent code with the frozen box encoder, and trains the model of
``src/gnn.py`` to regress the two screwdriving criteria and to classify feasibility. The loss
weights the region around each threshold, since the repair is steered by the gradient exactly
where a pair is close to passing, and an average error over the whole range says little about
that region.

The best epoch is written to ``config.gnn.gnn_folder`` as
``gnn_<tag>_node<node_dim>_batchsize<batch_size>_<timestamp>.pth``, holding the state dict under
the key ``model``. The standardisation of the two targets travels inside it as the buffers
``target_mean`` and ``target_std``: every consumer of a prediction has to undo that
standardisation to get metres, so it belongs with the weights rather than in a configuration
file that could drift away from them.

On the cluster the entry point is ``gnn_train_latentspace.slurm``.
"""

import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import wandb
from sklearn.metrics import f1_score, precision_score, recall_score
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from tqdm import tqdm

# Settings of the auxiliary failure-mode head. It is supervision only: the repair reads the
# regression outputs and the feasibility logit and never looks at these logits.
# Insufficient overlap is much rarer than excessive thickness in this dataset, so its positive
# weight would grow without bound; the clip stops one rare mode from dominating the loss.
MODE_POS_WEIGHT_CLIP: float = 50.0
LABEL_SMOOTH_EPS: float = 0.05
MODE_NAMES: list[str] = ["overlap_fail", "thickness_fail"]
MODE_LOSS_WEIGHT: float = 0.1


sys.path.insert(0, str(Path(__file__).parent.parent))
from config.config import config
from src.gnn import GNN
from src.gnn_dataset_preparation import (
    OVERLAP_LOG_EPS,
    build_graph_dataset,
    calculate_class_weights,
    compute_target_stats,
    split_graphs,
)

MODEL_FOLDER = Path(__file__).parent.parent / config.gnn.gnn_folder
MODEL_FOLDER.mkdir(exist_ok=True)

# The two screwdriving criteria in metres. Only these two can fail on this dataset: the
# generator places every child on its parent's face and above the table, so gap and table
# clearance are satisfied by construction.
THRESHOLDS = {
    "overlap": config.gnn.thresh_overlap_min,
    "thickness": config.gnn.thresh_thickness_max,
}


def build_mode_targets(reg_targets: torch.Tensor) -> torch.Tensor:
    """Derive the two failure-mode labels by applying the thresholds to the true criteria.

    Args:
        reg_targets: True overlap and thickness in metres, shape ``(B, 2)``.

    Returns:
        Shape ``(B, 2)``, one where the criterion is violated and zero otherwise, in the order
        insufficient overlap then excessive thickness.
    """
    overlap_fail = (reg_targets[:, 0] < THRESHOLDS["overlap"]).float()
    thickness_fail = (reg_targets[:, 1] > THRESHOLDS["thickness"]).float()
    return torch.stack([overlap_fail, thickness_fail], dim=1)


def calculate_mode_pos_weights(train_graphs: list[Data]) -> torch.Tensor:
    """Compute how far each failure mode must be up-weighted to offset its rarity.

    The weight is the ratio of negatives to positives, capped at
    :data:`MODE_POS_WEIGHT_CLIP`. Statistics are taken on the training split only.

    Returns:
        Shape ``(2,)``, in the order of :data:`MODE_NAMES`, for the ``pos_weight`` argument of
        the binary cross-entropy.
    """
    reg_targets = torch.stack([g.y.squeeze(0) for g in train_graphs])
    mode_targets = build_mode_targets(reg_targets)
    num_pos = mode_targets.sum(dim=0)
    num_neg = mode_targets.size(0) - num_pos
    pw = num_neg / torch.clamp(num_pos, min=1.0)
    pw = torch.clamp(pw, max=MODE_POS_WEIGHT_CLIP)
    for i, name in enumerate(MODE_NAMES):
        print(
            f"  mode={name}: pos={num_pos[i].item():.0f}, "
            f"neg={num_neg[i].item():.0f}, pos_weight={pw[i].item():.3f}"
        )
    return pw


def mode_bce_with_smoothing(
    logits: torch.Tensor, targets: torch.Tensor, pos_weight: torch.Tensor,
) -> torch.Tensor:
    """Binary cross-entropy on the failure modes, with per-mode weighting and smoothed labels.

    A mode label is derived by comparing a continuous quantity against a threshold, so a pair
    sitting a fraction of a millimetre from it is labelled as confidently as one far away.
    Moving the targets off zero and one by :data:`LABEL_SMOOTH_EPS` keeps the loss from
    rewarding certainty the labels do not support.

    Args:
        logits: Raw mode logits, shape ``(B, 2)``.
        targets: Mode labels from :func:`build_mode_targets`, shape ``(B, 2)``.
        pos_weight: Per-mode weighting, shape ``(2,)``.
    """
    smooth = targets * (1.0 - LABEL_SMOOTH_EPS) + (1.0 - targets) * LABEL_SMOOTH_EPS
    return F.binary_cross_entropy_with_logits(
        logits, smooth, pos_weight=pos_weight, reduction="mean",
    )


def random_seed(seed: int = config.general.random_seed or 42):
    """Seed every generator the run uses and put cuDNN into deterministic mode.

    The split of ``src/gnn_dataset_preparation.py`` draws from the global torch generator, so
    this also fixes which graphs end up in which split.
    """
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def compute_reg_loss(
    reg_pred_std: torch.Tensor,
    reg_targets: torch.Tensor,
    target_mean: torch.Tensor,
    target_std: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Regression loss that concentrates on the neighbourhood of the two thresholds.

    Both criteria are compared in the standardised space the model predicts in. Each pair's
    contribution is scaled by how close its true value lies to the threshold, so accuracy is
    bought where the feasible and the infeasible side meet. That is the region the repair
    operates in: a pair well inside either side needs no gradient, while one near the boundary
    decides whether the repair stops or keeps pushing.

    Args:
        reg_pred_std: Standardised predictions, shape ``(B, 2)``, overlap then thickness.
        reg_targets: True values in metres, shape ``(B, 2)``. The overlap column is put into
            logarithmic space here, matching what ``compute_target_stats`` standardised.
        target_mean, target_std: The standardisation, shape ``(2,)`` each, with the first entry
            in logarithmic space.

    Returns:
        A pair ``(reg_loss, parts)``, the second holding the two summands separately for
        logging.
    """
    reg_targets_for_std = reg_targets.clone()
    # Overlap spans orders of magnitude and the threshold sits near the low end of that range,
    # so it is standardised in logarithmic space; thickness is standardised directly. The
    # asymmetry is undone at inference by src.repair_optimizer.destandardize.
    reg_targets_for_std[:, 0] = torch.log(reg_targets[:, 0] + OVERLAP_LOG_EPS)
    reg_targets_std = (reg_targets_for_std - target_mean) / target_std

    # Width of the emphasised band and its strength, both in standardised units, and the
    # Huber knee. A pair on the threshold weighs 1 + boundary_gain, one sigma_std away weighs
    # about 1 + boundary_gain / e, and one far from it weighs 1.
    sigma_std = 0.5
    boundary_gain = 3.0
    huber_delta = 1.0

    # The threshold has to be carried through the same logarithm and the same standardisation
    # as the targets, otherwise the distance below is measured against the wrong point.
    overlap_thresh_log = math.log(float(THRESHOLDS["overlap"]) + OVERLAP_LOG_EPS)
    overlap_thresh_std = (
        overlap_thresh_log - target_mean[0].item()
    ) / target_std[0].item()
    dist_overlap_std = (reg_targets_std[:, 0] - overlap_thresh_std).abs()
    w_overlap = 1.0 + boundary_gain * torch.exp(-dist_overlap_std / sigma_std)
    loss_overlap_per = F.huber_loss(
        reg_pred_std[:, 0], reg_targets_std[:, 0],
        delta=huber_delta, reduction="none",
    )
    loss_overlap = (w_overlap * loss_overlap_per).mean()

    thickness_thresh_std = (
        float(THRESHOLDS["thickness"]) - target_mean[1].item()
    ) / target_std[1].item()
    dist_thick_std = (reg_targets_std[:, 1] - thickness_thresh_std).abs()
    w_thick = 1.0 + boundary_gain * torch.exp(-dist_thick_std / sigma_std)
    loss_thick_per = F.huber_loss(
        reg_pred_std[:, 1], reg_targets_std[:, 1],
        delta=huber_delta, reduction="none",
    )
    loss_thickness = (w_thick * loss_thick_per).mean()

    reg_loss = loss_overlap + loss_thickness
    parts = {
        "loss_reg_overlap": float(loss_overlap.item()),
        "loss_reg_thickness": float(loss_thickness.item()),
    }
    return reg_loss, parts


def train_one_epoch(
    model: GNN,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    pos_weight: torch.Tensor,
    mode_pos_weight: torch.Tensor,
    alpha: float = config.gnn.alpha,
    beta: float = MODE_LOSS_WEIGHT,
) -> tuple[float, dict[str, float]]:
    """Run one epoch over the training split and return the mean loss and its parts.

    The total is the regression loss plus ``alpha`` times the feasibility cross-entropy plus
    ``beta`` times the auxiliary mode loss. The regression term is unweighted because the
    repair follows its gradient; the other two are supervision that shapes the representation.

    Args:
        pos_weight: Positive class weight for the feasibility label, shape ``(1,)``.
        mode_pos_weight: Per-mode weighting for the auxiliary head, shape ``(2,)``.
        alpha: Weight of the feasibility term, ``config.gnn.alpha``.
        beta: Weight of the auxiliary mode term.

    Returns:
        A pair ``(mean_loss, parts)``, the second holding the individual terms averaged over
        batches.

    Raises:
        RuntimeError: If the model carries no target standardisation buffers, without which the
            regression loss cannot be formed.
    """
    model.train()
    total_loss: float = 0.0
    total_binary_loss: float = 0.0
    total_mode_loss: float = 0.0
    total_reg_overlap: float = 0.0
    total_reg_thickness: float = 0.0
    target_mean = getattr(model, "target_mean", None)
    target_std = getattr(model, "target_std", None)
    if target_mean is None or target_std is None:
        raise RuntimeError(
            "Model is missing target_mean/target_std buffers. "
            "Register them in main() before training."
        )

    pos_weight = pos_weight.to(device)
    mode_pos_weight = mode_pos_weight.to(device)

    for batch in tqdm(dataloader):
        batch = batch.to(device)

        reg_pred_std, binary_logit, mode_logits = model(
            batch.x, batch.edge_index, batch.edge_attr
        )
        reg_targets = batch.y.squeeze(1)
        binary_targets = batch.y_good.unsqueeze(1)
        # Derived from the same targets rather than stored, so the mode labels can never
        # disagree with the criteria the regression is fitted to.
        mode_targets = build_mode_targets(reg_targets)

        reg_loss, reg_parts = compute_reg_loss(
            reg_pred_std, reg_targets, target_mean, target_std
        )
        binary_loss = F.binary_cross_entropy_with_logits(
            binary_logit, binary_targets, pos_weight=pos_weight
        )
        mode_loss = mode_bce_with_smoothing(
            mode_logits, mode_targets, pos_weight=mode_pos_weight
        )

        loss = reg_loss + alpha * binary_loss + beta * mode_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item())
        total_binary_loss += float(binary_loss.item())
        total_mode_loss += float(mode_loss.item())
        total_reg_overlap += reg_parts["loss_reg_overlap"]
        total_reg_thickness += reg_parts["loss_reg_thickness"]

    n_batches = len(dataloader)
    train_parts = {
        "loss_reg_overlap": total_reg_overlap / n_batches,
        "loss_reg_thickness": total_reg_thickness / n_batches,
        "binary_loss": total_binary_loss / n_batches,
        "mode_loss": total_mode_loss / n_batches,
    }
    return total_loss / n_batches, train_parts


# Fixed ranges in metres, used to report the regression error separately near and far from a
# threshold. A single average hides exactly the failure that matters, a model that is accurate
# overall but biased in the millimetre either side of the boundary.
# The bucket edges are laid out around a 16 mm overlap threshold, whereas
# config.gnn.thresh_overlap_min is 10 mm; the bucket named as the boundary is therefore not the
# one containing the configured threshold. The buckets are diagnostics only and enter neither
# the loss nor the checkpoint selection.
OVERLAP_BUCKETS = [
    (0.000, 0.008, "0_8mm_fail_far"),
    (0.008, 0.016, "8_16mm_fail_near"),
    (0.016, 0.020, "16_20mm_boundary"),
    (0.020, 0.030, "20_30mm_far"),
    (0.030, float("inf"), "30mm_inf_far"),
]
# These edges do bracket the configured thickness threshold of 20 mm.
THICKNESS_BUCKETS = [
    (0.000, 0.016, "0_16mm_far"),
    (0.016, 0.020, "16_20mm_boundary"),
    (0.020, 0.024, "20_24mm_fail_near"),
    (0.024, float("inf"), "24mm_inf_fail_far"),
]


def _safe_std_ratio(pred: np.ndarray, target: np.ndarray) -> float:
    """Spread of the predictions over the spread of the targets; not a number when the targets
    are nearly constant."""
    denom = float(target.std())
    if denom < 1e-12:
        return float("nan")
    return float(pred.std() / denom)


def stratified_metrics(
    pred: np.ndarray,
    target: np.ndarray,
    buckets: list,
    name: str,
    use_abs: bool = False,
) -> dict[str, float]:
    """Report the mean absolute error and the regression slope inside each bucket.

    The slope is what makes this more informative than the error alone: a model that has
    learnt to predict the mean of a bucket has a small error there but a slope near zero, and
    the repair cannot be steered by a prediction that does not move with the geometry. A slope
    near one is the healthy case.

    Args:
        pred, target: Predictions and true values in metres, shape ``(N,)`` each.
        buckets: Triples of lower bound, upper bound and name, bounds in metres.
        name: Prefix the metric keys are grouped under.
        use_abs: Whether the bucket is selected on the magnitude of the target.

    Returns:
        Keys of the form ``strat/<name>/<tag>/{mae,slope,n}``. A bucket is reported only if it
        has enough samples for the slope to mean anything.
    """
    out: dict[str, float] = {}
    base = np.abs(target) if use_abs else target
    for lo, hi, tag in buckets:
        m = (base >= lo) & (base < hi)
        n = int(m.sum())
        # Below this the slope is dominated by noise and would be misread as a real trend.
        if n < 20:
            continue
        p, t = pred[m], target[m]
        mae = float(np.abs(p - t).mean())
        if t.std() > 1e-9:
            slope = float(np.cov(p, t, bias=True)[0, 1] / t.var())
        else:
            slope = float("nan")
        out[f"strat/{name}/{tag}/mae"] = mae
        out[f"strat/{name}/{tag}/slope"] = slope
        out[f"strat/{name}/{tag}/n"] = n
    return out


def validate(
    model: GNN,
    dataloader: DataLoader,
    device: torch.device,
    pos_weight: torch.Tensor,
    mode_pos_weight: torch.Tensor,
    alpha: float = config.gnn.alpha,
    beta: float = MODE_LOSS_WEIGHT,
) -> tuple[float, dict]:
    """Evaluate one split and return the mean loss together with the full metric dictionary.

    Two feasibility figures are reported and they answer different questions. The
    ``thresh_*`` metrics apply the two criteria to the regressed values, so they say whether
    the quantities the repair actually follows are good enough to decide feasibility.
    ``binary_f1`` reads the classification head instead. The head can be right where the
    regression is wrong, which is why the repair is not allowed to rely on it alone.

    Returns:
        A pair ``(mean_loss, metrics)``. The metrics carry the per-criterion mean absolute
        error in metres, the threshold-derived and direct classification scores, the per-mode
        scores, the stratified diagnostics under ``strat/``, the spread ratios, and
        ``composite_critical_mae``, the mean of the two errors that drives both the learning
        rate schedule and the choice of best checkpoint.

    Raises:
        RuntimeError: If the model carries no target standardisation buffers.
    """
    model.eval()
    total_loss: float = 0.0
    total_binary_loss: float = 0.0
    total_mode_loss: float = 0.0
    total_reg_overlap: float = 0.0
    total_reg_thickness: float = 0.0
    target_mean = getattr(model, "target_mean", None)
    target_std = getattr(model, "target_std", None)
    if target_mean is None or target_std is None:
        raise RuntimeError(
            "Model is missing target_mean/target_std buffers. "
            "Register them in main() before validation."
        )

    all_reg_preds: list[np.ndarray] = []
    all_reg_targets: list[np.ndarray] = []
    all_bin_preds: list[np.ndarray] = []
    all_bin_targets: list[np.ndarray] = []
    all_mode_preds: list[np.ndarray] = []
    all_mode_targets: list[np.ndarray] = []
    pos_weight = pos_weight.to(device)
    mode_pos_weight = mode_pos_weight.to(device)

    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            reg_pred_std, binary_logit, mode_logits = model(
                batch.x, batch.edge_index, batch.edge_attr
            )
            reg_targets = batch.y.squeeze(1)
            binary_targets = batch.y_good.unsqueeze(1)
            mode_targets = build_mode_targets(reg_targets)

            reg_loss, reg_parts = compute_reg_loss(
                reg_pred_std, reg_targets, target_mean, target_std
            )
            binary_loss = F.binary_cross_entropy_with_logits(
                binary_logit, binary_targets, pos_weight=pos_weight
            )
            mode_loss = mode_bce_with_smoothing(
                mode_logits, mode_targets, pos_weight=mode_pos_weight
            )

            total_loss += float(
                (reg_loss + alpha * binary_loss + beta * mode_loss).item()
            )
            total_binary_loss += float(binary_loss.item())
            total_mode_loss += float(mode_loss.item())
            total_reg_overlap += reg_parts["loss_reg_overlap"]
            total_reg_thickness += reg_parts["loss_reg_thickness"]

            # Reported errors are in metres, so the predictions are brought back to that scale
            # here. The same inverse is applied at inference by
            # src.repair_optimizer.destandardize, and the two must stay in step.
            reg_pred = reg_pred_std * target_std + target_mean
            reg_pred = reg_pred.clone()
            reg_pred[:, 0] = torch.exp(reg_pred[:, 0]) - OVERLAP_LOG_EPS

            all_reg_preds.append(reg_pred.cpu().numpy())
            all_reg_targets.append(reg_targets.cpu().numpy())
            all_bin_preds.append((torch.sigmoid(binary_logit) > 0.5).cpu().numpy())
            all_bin_targets.append(binary_targets.cpu().numpy())
            all_mode_preds.append((torch.sigmoid(mode_logits) > 0.5).cpu().numpy())
            all_mode_targets.append(mode_targets.cpu().numpy())

    reg_preds = np.concatenate(all_reg_preds)
    reg_targets = np.concatenate(all_reg_targets)
    bin_preds = np.concatenate(all_bin_preds).ravel()
    bin_targets = np.concatenate(all_bin_targets).ravel()
    mode_preds = np.concatenate(all_mode_preds)
    mode_targets_np = np.concatenate(all_mode_targets)

    reg_names = ["overlap", "thickness"]
    metrics: dict = {}
    for i, name in enumerate(reg_names):
        metrics[f"mae_{name}"] = float(
            np.abs(reg_preds[:, i] - reg_targets[:, i]).mean()
        )

    # Feasibility as the repair would derive it, from the regressed criteria rather than from
    # the classification head.
    feasible_pred = feasible_from_regression(reg_preds)
    metrics["thresh_f1"] = f1_score(bin_targets, feasible_pred, zero_division=0)
    metrics["thresh_precision"] = precision_score(
        bin_targets, feasible_pred, zero_division=0
    )
    metrics["thresh_recall"] = recall_score(bin_targets, feasible_pred, zero_division=0)

    metrics["binary_f1"] = f1_score(bin_targets, bin_preds, zero_division=0)

    mode_f1_values: list[float] = []
    for i, name in enumerate(MODE_NAMES):
        f1_mode = f1_score(
            mode_targets_np[:, i], mode_preds[:, i], zero_division=0
        )
        metrics[f"mode_f1_{name}"] = float(f1_mode)
        mode_f1_values.append(float(f1_mode))
    metrics["mode_f1_macro"] = float(np.mean(mode_f1_values))

    metrics.update(
        stratified_metrics(
            reg_preds[:, 0], reg_targets[:, 0], OVERLAP_BUCKETS,
            name="overlap", use_abs=False,
        )
    )
    metrics.update(
        stratified_metrics(
            reg_preds[:, 1], reg_targets[:, 1], THICKNESS_BUCKETS,
            name="thickness", use_abs=False,
        )
    )

    # A ratio well below one means the model is predicting close to a constant, which a mean
    # absolute error alone would not reveal and which would leave the repair without a gradient
    # to follow.
    metrics["std_ratio_overlap"] = _safe_std_ratio(reg_preds[:, 0], reg_targets[:, 0])
    metrics["std_ratio_thickness"] = _safe_std_ratio(reg_preds[:, 1], reg_targets[:, 1])

    # The checkpoint is selected on regression accuracy, not on a classification score. The
    # repair follows these two quantities, so a model that classified well while regressing
    # poorly would be the wrong one to keep.
    metrics["composite_critical_mae"] = float(
        np.mean([metrics["mae_overlap"], metrics["mae_thickness"]])
    )

    n_batches = len(dataloader)
    metrics["loss_reg_overlap"] = total_reg_overlap / n_batches
    metrics["loss_reg_thickness"] = total_reg_thickness / n_batches
    metrics["binary_loss"] = total_binary_loss / n_batches
    metrics["mode_loss"] = total_mode_loss / n_batches

    return total_loss / n_batches, metrics


def feasible_from_regression(reg_pred: np.ndarray) -> np.ndarray:
    """Apply the two criteria to regressed values to obtain a feasibility label.

    Args:
        reg_pred: Predictions in metres, shape ``(B, 2)``, overlap then thickness.

    Returns:
        Shape ``(B,)``, one where both criteria hold and zero otherwise.
    """
    fail = (
        (reg_pred[:, 0] < THRESHOLDS["overlap"])
        | (reg_pred[:, 1] > THRESHOLDS["thickness"])
    )
    return (~fail).astype(float)


def get_monitor_value(
    val_loss: float,
    val_metrics: dict,
    monitor_metric: str,
) -> float:
    """Look up the quantity the checkpoint selection watches.

    Raises:
        KeyError: If the named metric is neither ``val_loss`` nor a key of the metric
            dictionary, listing what is available.
    """
    if monitor_metric == "val_loss":
        return float(val_loss)
    if monitor_metric in val_metrics:
        return float(val_metrics[monitor_metric])
    available_keys = ", ".join(sorted(val_metrics.keys()))
    raise KeyError(
        f"Unknown monitor metric '{monitor_metric}'. Use 'val_loss' or one of: {available_keys}"
    )


def main():
    """Build the dataset, train the surrogate, keep the best epoch, and score it on the test split."""
    import os
    import subprocess

    # The checkpoint filename carries a tag so that surrogates of the different branches cannot
    # be confused in one folder. It falls back to the branch name, since the branch is what
    # distinguishes the parameter formulation from the latent one.
    exp_name: str = os.environ.get("GNN_EXP_NAME", "")
    if not exp_name:
        try:
            branch = subprocess.check_output(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                cwd=str(Path(__file__).resolve().parent.parent),
                stderr=subprocess.DEVNULL,
                text=True,
            ).strip()
            for prefix in ("feature/gnn-", "feature/", "feat/"):
                if branch.startswith(prefix):
                    branch = branch[len(prefix):]
                    break
            exp_name = branch.replace("/", "_")
        except (subprocess.SubprocessError, FileNotFoundError):
            exp_name = ""
    exp_tag: str = f"{exp_name}_" if exp_name else ""
    if exp_name:
        print(f"Checkpoint tag: '{exp_name}'")

    # Quick mode is for checking that the pipeline runs, not for producing a surrogate: it
    # subsets the data and disables early stopping, so its checkpoints are not comparable.
    quick_mode: bool = bool(int(os.environ.get("GNN_QUICK", "0")))
    quick_n: int = int(os.environ.get("GNN_QUICK_N", "5000"))
    effective_epochs: int = (
        int(os.environ.get("GNN_QUICK_EPOCHS", "5"))
        if quick_mode
        else config.training.epochs
    )
    effective_patience: int = 10**6 if quick_mode else config.training.patience

    random_seed()
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if quick_mode:
        print(
            f"GNN_QUICK enabled: subset to {quick_n} graphs, "
            f"{effective_epochs} epochs, patience disabled."
        )

    run_name = (
        f"{exp_name}_{time.strftime('%Y%m%d-%H%M')}" if exp_name else None
    )
    run_tags = [exp_name] if exp_name else None

    run = wandb.init(
        project="train_gnn_feasibility",
        name=run_name,
        tags=run_tags,
        config={
            "node_dim": config.gnn.node_dim,
            "edge_dim": config.gnn.edge_dim,
            "hidden_dim": config.gnn.hidden_dim,
            "heads": config.gnn.heads,
            "dropout": config.gnn.dropout,
            "output_dim": config.gnn.output_dim,
            "lr": config.training.learning_rate,
            "batch_size": config.training.batch_size_gnn,
            "epochs": effective_epochs,
            "patience": effective_patience,
            "quick_mode": quick_mode,
            "label_names": config.data.label_names,
            "alpha": config.gnn.alpha,
            "monitor_metric": "composite_critical_mae",
            "monitor_mode": "min",
            "exp_name": exp_name,
        },
    )

    print("Building graph dataset...")
    start_time: float = time.time()

    # Each graph has two nodes, node 0 the parent and node 1 the part under test, node features
    # of width config.gnn.node_dim built from the frozen box encoder, targets in metres and one
    # feasibility label.
    graphs: list[Data] = build_graph_dataset(device=str(device))

    data_time: float = time.time() - start_time
    print(f"Built {len(graphs)} graphs in {data_time:.2f}s")

    if quick_mode:
        graphs = graphs[:quick_n]
        print(f"GNN_QUICK: truncated to {len(graphs)} graphs.")

    train_graphs, val_graphs, test_graphs = split_graphs(graphs)

    # Statistics and class weights come from the training split alone, so that nothing about
    # the validation or test split enters the model through its standardisation.
    target_stats: dict[str, torch.Tensor] = compute_target_stats(train_graphs)
    pos_weight: torch.Tensor = calculate_class_weights(train_graphs)
    mode_pos_weight: torch.Tensor = calculate_mode_pos_weights(train_graphs)

    batch_size: int = config.training.batch_size_gnn
    train_loader: DataLoader = DataLoader(
        train_graphs, batch_size=batch_size, shuffle=True
    )
    val_loader: DataLoader = DataLoader(val_graphs, batch_size=batch_size)
    test_loader: DataLoader = DataLoader(test_graphs, batch_size=batch_size)

    print("Initializing GNN model...")
    model: GNN = GNN(
        node_dim=config.gnn.node_dim,
        edge_dim=config.gnn.edge_dim,
        hidden_dim=config.gnn.hidden_dim,
        heads=config.gnn.heads,
        dropout=config.gnn.dropout,
    ).to(device)

    # Registered as buffers rather than kept beside the model, so they are saved with the
    # weights. The repair loads a checkpoint and immediately needs to read predictions in
    # metres; carrying the standardisation separately would let the two drift apart silently.
    model.register_buffer("target_mean", target_stats["mean"].to(device))
    model.register_buffer("target_std", target_stats["std"].to(device))

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  node_dim: {config.gnn.node_dim}")
    print(f"  edge_dim: {config.gnn.edge_dim}")
    print(f"  hidden_dim: {config.gnn.hidden_dim}")
    print(f"  heads: {config.gnn.heads}")
    print(f"  dropout: {config.gnn.dropout}")
    print(f"  learning_rate: {config.training.learning_rate}")
    print(f"  batch_size: {config.training.batch_size_gnn}")
    print(f"  alpha (binary loss weight): {config.gnn.alpha}")
    print(f"  beta  (mode loss weight):   {MODE_LOSS_WEIGHT}")
    print(f"  epochs: {effective_epochs}, patience: {effective_patience}")

    optimizer: torch.optim.Optimizer = torch.optim.Adam(
        model.parameters(), lr=config.training.learning_rate
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.7, patience=12, min_lr=1e-6
    )

    print(f"Starting training for {effective_epochs} epochs...")
    monitor_metric: str = "composite_critical_mae"
    best_monitor_value: float = float("inf")
    patience_counter: int = 0
    timestamp = time.strftime("%Y%m%d-%H%M%S")

    for epoch in range(effective_epochs):
        start_time = time.time()

        train_loss, train_parts = train_one_epoch(
            model, train_loader, optimizer, device, pos_weight, mode_pos_weight,
        )
        val_loss, val_metrics = validate(
            model, val_loader, device, pos_weight, mode_pos_weight,
        )

        epoch_time: float = time.time() - start_time

        print(
            f"Epoch {epoch+1}/{effective_epochs} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Loss: {val_loss:.6f} | "
            f"composite_critical_mae: {val_metrics['composite_critical_mae']:.6f} | "
            f"monitor({monitor_metric}): {get_monitor_value(val_loss, val_metrics, monitor_metric):.4f} | "
            f"threshold_F1: {val_metrics['thresh_f1']:.4f} | "
            f"threshold_P: {val_metrics['thresh_precision']:.4f} | "
            f"threshold_R: {val_metrics['thresh_recall']:.4f} | "
            f"binary_F1: {val_metrics['binary_f1']:.4f} | "
            f"mode_F1: {val_metrics['mode_f1_macro']:.4f} | "
            f"MAE_overlap: {val_metrics['mae_overlap']:.4f} | "
            f"MAE_thick: {val_metrics['mae_thickness']:.4f} | "
            f"Time per epoch: {epoch_time:.2f}s"
        )

        run.log(
            {
                "train_loss": train_loss,
                "val_loss": val_loss,
                "train_loss_reg_overlap": train_parts["loss_reg_overlap"],
                "train_loss_reg_thickness": train_parts["loss_reg_thickness"],
                "train_binary_loss": train_parts["binary_loss"],
                "train_mode_loss": train_parts["mode_loss"],
                "val_loss_reg_overlap": val_metrics["loss_reg_overlap"],
                "val_loss_reg_thickness": val_metrics["loss_reg_thickness"],
                "val_binary_loss": val_metrics["binary_loss"],
                "val_mode_loss": val_metrics["mode_loss"],
                "loss_reg_overlap": val_metrics["loss_reg_overlap"],
                "loss_reg_thickness": val_metrics["loss_reg_thickness"],
                "binary_loss": val_metrics["binary_loss"],
                "mode_loss": val_metrics["mode_loss"],
                "composite_critical_mae": val_metrics["composite_critical_mae"],
                "std_ratio_overlap": val_metrics["std_ratio_overlap"],
                "std_ratio_thickness": val_metrics["std_ratio_thickness"],
                "val_threshold_f1": val_metrics["thresh_f1"],
                "val_threshold_precision": val_metrics["thresh_precision"],
                "val_threshold_recall": val_metrics["thresh_recall"],
                "val_binary_f1": val_metrics["binary_f1"],
                "val_mode_f1_overlap_fail": val_metrics["mode_f1_overlap_fail"],
                "val_mode_f1_thickness_fail": val_metrics["mode_f1_thickness_fail"],
                "val_mode_f1_macro": val_metrics["mode_f1_macro"],
                "val_mae_overlap": val_metrics["mae_overlap"],
                "val_mae_thickness": val_metrics["mae_thickness"],
                "learning_rate": optimizer.param_groups[0]["lr"],
                "epoch": epoch,
            }
        )

        strat_logs = {k: v for k, v in val_metrics.items() if k.startswith("strat/")}
        if strat_logs:
            run.log(strat_logs)

        scheduler.step(val_metrics["composite_critical_mae"])

        current_monitor_value: float = get_monitor_value(
            val_loss, val_metrics, monitor_metric
        )
        improved: bool = current_monitor_value < best_monitor_value

        if improved:
            best_monitor_value = current_monitor_value
            patience_counter = 0

            model_dir: Path = MODEL_FOLDER
            model_dir.mkdir(parents=True, exist_ok=True)
            timestamp = time.strftime("%Y%m%d-%H%M%S")

            torch.save(
                {"model": model.state_dict()},
                model_dir
                / f"gnn_{exp_tag}node{config.gnn.node_dim}_batchsize{config.training.batch_size_gnn}_{timestamp}.pth",
            )
            print(
                f"  -> Saved best model | monitor({monitor_metric})={current_monitor_value:.6f} | "
                f"Val Loss: {val_loss:.6f} | threshold_F1: {val_metrics['thresh_f1']:.4f}"
            )
        else:
            patience_counter += 1
            if patience_counter >= effective_patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    # The test score is taken on the best checkpoint, not on whatever epoch the loop ended on.
    print("Loading best model and evaluating on test set...")

    MODEL_FOLDER.mkdir(parents=True, exist_ok=True)
    ckpt: dict = torch.load(
        MODEL_FOLDER
        / f"gnn_{exp_tag}node{config.gnn.node_dim}_batchsize{config.training.batch_size_gnn}_{timestamp}.pth"
    )
    model.load_state_dict(ckpt["model"])

    test_loader = DataLoader(test_graphs, batch_size=batch_size)
    test_loss, test_metrics = validate(
        model, test_loader, device, pos_weight, mode_pos_weight,
    )

    print(f"\n{'='*50}")
    print("Test Results:")
    print(f"  Loss:                {test_loss:.6f}")
    print(f"  threshold_F1:        {test_metrics['thresh_f1']:.4f}")
    print(f"  threshold_Precision: {test_metrics['thresh_precision']:.4f}")
    print(f"  threshold_Recall:    {test_metrics['thresh_recall']:.4f}")
    print(f"  binary_F1:           {test_metrics['binary_f1']:.4f}")
    print(f"  mode_F1_macro:       {test_metrics['mode_f1_macro']:.4f}")
    print(f"  mode_F1_overlap:     {test_metrics['mode_f1_overlap_fail']:.4f}")
    print(f"  mode_F1_thickness:   {test_metrics['mode_f1_thickness_fail']:.4f}")
    print(f"  MAE overlap:         {test_metrics['mae_overlap']:.4f}")
    print(f"  MAE thickness:       {test_metrics['mae_thickness']:.4f}")
    print(f"{'='*50}")

    run.log(
        {
            "test_loss": test_loss,
            "test_threshold_f1": test_metrics["thresh_f1"],
            "test_threshold_precision": test_metrics["thresh_precision"],
            "test_threshold_recall": test_metrics["thresh_recall"],
            "test_binary_f1": test_metrics["binary_f1"],
            "test_mode_f1_macro": test_metrics["mode_f1_macro"],
            "test_mode_f1_overlap_fail": test_metrics["mode_f1_overlap_fail"],
            "test_mode_f1_thickness_fail": test_metrics["mode_f1_thickness_fail"],
            "test_mae_overlap": test_metrics["mae_overlap"],
            "test_mae_thickness": test_metrics["mae_thickness"],
        }
    )

    run.finish()


if __name__ == "__main__":
    main()
