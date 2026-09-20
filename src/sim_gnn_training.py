"""Train the feasibility surrogate of Part III on the configurations labelled in simulation.

Four binary targets are learned at once: the three failure modes and, separately, the overall
verdict. Each is treated as its own decision rather than as one class of a common outcome,
because a configuration can fail in more than one way and the modes are informative about why.
Nothing is regressed; the quantities the earlier parts regressed are not defined for the curved
members of the vocabulary.

The targets are rare to differing degrees, so each is weighted by the ratio of negatives to
positives in the training split. Without that the model would reach a good loss by predicting
that nothing ever fails.

What the training run leaves behind is more than the weights. The checkpoint also holds the
standardisation of the inputs and, per target, the decision threshold tuned on the validation
split. Both have to travel with the model: the repair builds inputs itself and reads the
threshold, and neither would otherwise be reproducible.

Run:
    python -m src.sim_gnn_training
    python -m src.sim_gnn_training --limit 6000 --epochs 2 --batch-size 512
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)
from torch_geometric.loader import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.config import config  # noqa: E402
from src.gnn import SimAssemblyGNN  # noqa: E402
from src.sim_gnn_dataset import (  # noqa: E402
    _TARGET_FLAGS,
    load_shape_latents,
    node_dim,
    txt_to_graphs,
)

FAIL_NAMES = [f.replace("ObjNew_", "") for f in _TARGET_FLAGS]
NUM_FAIL = len(FAIL_NAMES)
MODEL_FOLDER = Path(__file__).resolve().parent.parent / config.gnn.gnn_folder
DEFAULT_TXT = "data/sim_shape_dataset_1613.txt"
DEFAULT_LATENTS = "encoder_decoder_model/shape_latents_lv_v14_lam005.pt"


def set_seed(seed: int = 42) -> None:
    """Seed the random number generators of torch, the standard library and numpy."""
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def split_graphs(graphs, seed: int = 42):
    """Split the graphs into four fifths for training and one tenth each for validation and test.

    The order is shuffled from a fixed seed, so the same file always yields the same split. The
    per-shape evaluation reproduces this split from the seed alone, which is what makes its
    numbers comparable with the ones reported at the end of training.
    """
    idx = list(range(len(graphs)))
    rng = random.Random(seed)
    rng.shuffle(idx)
    n = len(idx)
    n_tr = int(0.8 * n)
    n_va = int(0.1 * n)
    tr = [graphs[i] for i in idx[:n_tr]]
    va = [graphs[i] for i in idx[n_tr:n_tr + n_va]]
    te = [graphs[i] for i in idx[n_tr + n_va:]]
    return tr, va, te


def pos_weights(graphs, key: str, width: int) -> torch.Tensor:
    """Weight of the positive class per target, the ratio of negatives to positives.

    Computed on the training split only, so the balance of the validation and test splits does
    not leak into the objective.

    Args:
        key: Which label tensor of a graph to read, the failure modes or the verdict.
        width: Number of targets in that tensor.

    Returns:
        One weight per target, shape ``(width,)``.
    """
    Y = torch.cat([getattr(g, key) for g in graphs]).view(-1, width)
    n_pos = Y.sum(dim=0)
    n_neg = Y.size(0) - n_pos
    return n_neg / torch.clamp(n_pos, min=1.0)


def run_epoch(model, loader, device, fail_pw, good_pw, optimizer=None,
              grad_clip: float | None = None):
    """Run one pass over a loader, training if an optimiser is given and evaluating otherwise.

    Probabilities are returned rather than decisions, so that the threshold can be chosen
    afterwards from the same pass instead of being fixed in advance.

    Args:
        optimizer: Given for a training pass, omitted for an evaluation pass. It also decides
            whether dropout is active and whether gradients are recorded.
        grad_clip: Maximum gradient norm, or ``None`` for no clipping.

    Returns:
        The mean loss over the batches, then the predicted probabilities and the targets for
        the failure modes, shape ``(N, num_fail)`` each, and for the verdict, shape ``(N,)``.
    """
    train = optimizer is not None
    model.train(train)
    total_loss, n_batches = 0.0, 0
    fp, ft, gp, gt = [], [], [], []
    fail_pw = fail_pw.to(device)
    good_pw = good_pw.to(device)

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for batch in loader:
            batch = batch.to(device)
            fail_logits, good_logit = model(batch.x, batch.edge_index, batch.edge_attr)
            fail_t = batch.y.view(-1, NUM_FAIL)
            good_t = batch.y_good.view(-1, 1)

            # The two heads contribute equally. The verdict is not derived from the modes here;
            # it is its own target, because the simulator can reject a configuration for
            # reasons none of the three modes names.
            loss_fail = F.binary_cross_entropy_with_logits(
                fail_logits, fail_t, pos_weight=fail_pw
            )
            loss_good = F.binary_cross_entropy_with_logits(
                good_logit, good_t, pos_weight=good_pw
            )
            loss = loss_fail + loss_good

            if train:
                optimizer.zero_grad()
                loss.backward()
                if grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

            total_loss += float(loss.item())
            n_batches += 1
            fp.append(torch.sigmoid(fail_logits).detach().cpu().numpy())
            ft.append(fail_t.cpu().numpy())
            gp.append(torch.sigmoid(good_logit).detach().cpu().numpy())
            gt.append(good_t.cpu().numpy())

    return (
        total_loss / max(n_batches, 1),
        np.concatenate(fp), np.concatenate(ft),
        np.concatenate(gp).ravel(), np.concatenate(gt).ravel(),
    )


def input_stats(graphs, skip_last_x: int = 2):
    """Mean and standard deviation of every node and edge entry, over the training split.

    A node mixes lengths in metres with code entries of order one, so without this the size
    would dominate and the shape would count for almost nothing. The statistics are handed to
    the model, which applies them itself.

    Args:
        graphs: The training split only; using more would leak into the standardisation.
        skip_last_x: Number of trailing node entries left untouched, the role indicator. It is
            already zero or one and standardising it would only obscure the distinction.

    Returns:
        The mean and standard deviation for the node entries, shape ``(node_dim,)`` each, and
        for the edge entries, shape ``(edge_dim,)`` each.
    """
    X = torch.cat([g.x for g in graphs], dim=0)
    E = torch.cat([g.edge_attr for g in graphs], dim=0)
    x_mean, x_std = X.mean(0), X.std(0)
    e_mean, e_std = E.mean(0), E.std(0)
    # An entry that never varies would otherwise be divided by something near zero. The code
    # entries that the compression has emptied are exactly such entries.
    x_std = torch.where(x_std < 1e-6, torch.ones_like(x_std), x_std)
    e_std = torch.where(e_std < 1e-6, torch.ones_like(e_std), e_std)
    if skip_last_x > 0:
        x_mean[-skip_last_x:] = 0.0
        x_std[-skip_last_x:] = 1.0
    return x_mean, x_std, e_mean, e_std


def tune_threshold(prob, tgt) -> tuple[float, float]:
    """Choose the decision threshold of one target as the one with the best balanced score.

    A threshold of one half would be arbitrary here: the targets are rare and were trained under
    a class weight, so the probabilities are not calibrated. The threshold is chosen on the
    validation split and then applied unchanged to the test split.

    Returns:
        The threshold and the score it reaches. A target that is entirely one class in this
        split has no meaningful threshold and is given one half and a score of zero.
    """
    if tgt.max() == tgt.min():
        return 0.5, 0.0
    p, r, th = precision_recall_curve(tgt, prob)
    f1 = 2 * p * r / (p + r + 1e-12)
    i = int(np.nanargmax(f1))
    t = float(th[min(i, len(th) - 1)]) if len(th) else 0.5
    return t, float(f1[i])


def _safe(fn, prob, tgt) -> float:
    """Evaluate a ranking measure, returning not-a-number where the target has only one class."""
    return float("nan") if tgt.max() == tgt.min() else float(fn(tgt, prob))


def evaluate(fp, ft, gp, gt, thresholds: dict | None = None) -> tuple[dict, dict]:
    """Score every target, both at a decision threshold and independently of one.

    The two ranking measures are reported alongside the thresholded score because they are not
    affected by the choice of threshold, and because the second of them is the informative one
    when a target is rare.

    Args:
        thresholds: Thresholds to apply. If omitted, each is tuned on the data passed in, which
            is what the validation pass does; the test pass passes the tuned ones in instead.

    Returns:
        The scores, and the thresholds that were used.
    """
    m, fail_thr = {}, []
    for i, name in enumerate(FAIL_NAMES):
        t = thresholds["fail"][i] if thresholds else tune_threshold(fp[:, i], ft[:, i])[0]
        fail_thr.append(t)
        m[f"f1_{name}"] = f1_score(ft[:, i], fp[:, i] > t, zero_division=0)
        m[f"auroc_{name}"] = _safe(roc_auc_score, fp[:, i], ft[:, i])
        m[f"ap_{name}"] = _safe(average_precision_score, fp[:, i], ft[:, i])
    m["f1_macro_fail"] = float(np.mean([m[f"f1_{n}"] for n in FAIL_NAMES]))
    g_thr = thresholds["good"] if thresholds else tune_threshold(gp, gt)[0]
    m["f1_good"] = f1_score(gt, gp > g_thr, zero_division=0)
    m["auroc_good"] = _safe(roc_auc_score, gp, gt)
    m["ap_good"] = _safe(average_precision_score, gp, gt)
    return m, {"fail": fail_thr, "good": g_thr}


def main():
    """Read the dataset, train the surrogate, and report it on the held-out split."""
    ap = argparse.ArgumentParser()
    ap.add_argument("txt", nargs="?", default=DEFAULT_TXT)
    ap.add_argument("--latents", nargs="+", default=[DEFAULT_LATENTS],
                    help="one or more {name:z} tables, merged (base-14 + z-continuum)")
    ap.add_argument("--limit", type=int, default=None,
                    help="max rows parsed (smoke test)")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--patience", type=int, default=12)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Logging is optional and offline. The cluster nodes have no outbound network, and a run
    # must not fail because the logger cannot start.
    run = None
    try:
        import wandb
        run = wandb.init(
            project="sim_gnn_feasibility",
            config=vars(args),
            mode="offline",
        )
    except Exception as e:  # noqa: BLE001
        print(f"wandb disabled ({e})")

    root = Path(__file__).resolve().parent.parent
    latents = load_shape_latents([root / p for p in args.latents])
    nd = node_dim(latents)
    latent_dim = nd - 3 - 2
    print(f"latents: {len(latents)} shapes | node_dim={nd} (z{latent_dim}+bbox3+type2)")

    t0 = time.time()
    graphs, stats = txt_to_graphs(root / args.txt, latents, limit=args.limit)
    print(f"built {len(graphs)} graphs in {time.time() - t0:.1f}s | stats={stats}")

    tr, va, te = split_graphs(graphs, seed=args.seed)
    print(f"split: train={len(tr)} val={len(va)} test={len(te)}")

    fail_pw = pos_weights(tr, "y", NUM_FAIL)
    good_pw = pos_weights(tr, "y_good", 1)
    print("train pos_weight (n_neg/n_pos):")
    for name, w in zip(FAIL_NAMES, fail_pw.tolist()):
        print(f"  {name:<22} {w:8.3f}")
    print(f"  {'Assembly_Good':<22} {good_pw.item():8.3f}")

    train_loader = DataLoader(tr, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(va, batch_size=args.batch_size)
    test_loader = DataLoader(te, batch_size=args.batch_size)

    model = SimAssemblyGNN(
        node_dim=nd,
        edge_dim=config.gnn.edge_dim,
        hidden_dim=config.gnn.hidden_dim,
        heads=config.gnn.heads,
        dropout=config.gnn.dropout,
        head_hidden=config.gnn.head_hidden,
        num_fail=NUM_FAIL,
    ).to(device)
    print(f"model params: {sum(p.numel() for p in model.parameters()):,}")

    # The statistics are stored in the model rather than applied to the data, so that they are
    # saved with the weights and anything querying the model later cannot use different ones.
    model.set_input_stats(*input_stats(tr))

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    MODEL_FOLDER.mkdir(parents=True, exist_ok=True)
    ckpt_path = MODEL_FOLDER / f"sim_gnn_shape_node{nd}_{latent_dim}.pth"

    best_val = float("inf")
    best_thr = None
    patience = 0
    for epoch in range(args.epochs):
        t0 = time.time()
        tr_loss, *_ = run_epoch(
            model, train_loader, device, fail_pw, good_pw,
            optimizer=optimizer, grad_clip=args.grad_clip,
        )
        va_loss, fp, ft, gp, gt = run_epoch(
            model, val_loader, device, fail_pw, good_pw,
        )
        # The thresholds are tuned here, on the validation split, and the ones belonging to the
        # epoch that is kept are stored with it.
        m, thr = evaluate(fp, ft, gp, gt)

        f1s = " ".join(f"{n.split('_')[0][:4].lower()} {m['f1_' + n]:.3f}"
                       for n in FAIL_NAMES)
        aucs = " ".join(f"{n.split('_')[0][:4].lower()} {m['auroc_' + n]:.3f}"
                        for n in FAIL_NAMES)
        print(
            f"Epoch {epoch + 1}/{args.epochs} | train {tr_loss:.4f} | "
            f"val {va_loss:.4f} | F1[{f1s}] AUROC[{aucs}] "
            f"good {m['f1_good']:.3f}/{m['auroc_good']:.3f} | "
            f"{time.time() - t0:.1f}s"
        )
        if run is not None:
            run.log({"epoch": epoch, "train_loss": tr_loss, "val_loss": va_loss, **m})

        if va_loss < best_val:
            best_val = va_loss
            best_thr = thr
            patience = 0
            torch.save(
                {
                    "model": model.state_dict(),
                    "node_dim": nd,
                    "latent_dim": latent_dim,
                    "edge_dim": config.gnn.edge_dim,
                    "hidden_dim": config.gnn.hidden_dim,
                    "heads": config.gnn.heads,
                    "dropout": config.gnn.dropout,
                    "head_hidden": config.gnn.head_hidden,
                    "num_fail": NUM_FAIL,
                    "fail_names": FAIL_NAMES,
                    # The code tables are recorded so that anything loading this checkpoint
                    # reproduces the exact node width and codes it was trained under.
                    "latents_path": [str(p) for p in args.latents],
                    "thresholds": best_thr,
                },
                ckpt_path,
            )
            print(f"  -> saved best (val {best_val:.4f}) to {ckpt_path.name}")
        else:
            patience += 1
            if patience >= args.patience:
                print(f"early stop at epoch {epoch + 1}")
                break

    # The test split is scored on the kept epoch and at the thresholds that epoch tuned on the
    # validation split, so nothing about the test split enters the decisions.
    if ckpt_path.exists():
        model.load_state_dict(torch.load(ckpt_path, map_location=device)["model"])
    te_loss, fp, ft, gp, gt = run_epoch(model, test_loader, device, fail_pw, good_pw)
    m, _ = evaluate(fp, ft, gp, gt, thresholds=best_thr)
    print(f"\n{'=' * 62}\nTest | loss {te_loss:.4f}  (F1 @ val-tuned thresholds)")
    print(f"  {'target':<22} {'F1':>6} {'AUROC':>7} {'AP':>7} {'thr':>6}")
    for i, name in enumerate(FAIL_NAMES):
        print(f"  {name:<22} {m['f1_' + name]:6.3f} {m['auroc_' + name]:7.3f} "
              f"{m['ap_' + name]:7.3f} {best_thr['fail'][i]:6.3f}")
    print(f"  {'Assembly_Good':<22} {m['f1_good']:6.3f} {m['auroc_good']:7.3f} "
          f"{m['ap_good']:7.3f} {best_thr['good']:6.3f}")
    print(f"  {'macro (fail)':<22} {m['f1_macro_fail']:6.3f}")
    print("=" * 62)
    if run is not None:
        run.log({"test_loss": te_loss, **{f"test_{k}": v for k, v in m.items()}})
        run.finish()


if __name__ == "__main__":
    main()
