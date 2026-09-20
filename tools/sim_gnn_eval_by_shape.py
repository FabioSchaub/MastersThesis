"""Score a trained surrogate again, broken down by the shape of the part under test.

Training reports one number per target over the whole split, which cannot say whether the
surrogate understands the vocabulary or has learned the few shapes that dominate the dataset.
This tool reproduces the same split from the same seed, so its overall figures match the ones
the training run printed, and then repeats them per shape.

The thresholds are the ones stored in the checkpoint and are not re-tuned here; the two ranking
measures do not depend on them at all. The verdict is the one that matters for the repair, since
that is the score the repair pushes.

Run:
    python -m tools.sim_gnn_eval_by_shape
    python -m tools.sim_gnn_eval_by_shape --split all --csv <output csv>

Writes:
    nothing by default; with ``--csv`` the full per-shape table, including the measures the
    terminal summary leaves out.
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from torch_geometric.loader import DataLoader

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.gnn import SimAssemblyGNN  # noqa: E402
from src.sim_gnn_dataset import load_shape_latents, node_dim  # noqa: E402

DEFAULT_CKPT = "gnn_models/sim_gnn_shape_node37_32.pth"
DEFAULT_TXT = "data/sim_shape_dataset_1613.txt"


def parse_with_names(txt_path, latents, target_flags, limit=None):
    """Read the dataset file into graphs, and also return the shape name of each part under test.

    A repetition of what the dataset module does, because that one discards the names and the
    grouping needs them. The construction of a graph has to stay identical to it.
    """
    from torch_geometric.data import Data
    from src.sim_gnn_dataset import _edge_features

    graphs, names = [], []
    with open(txt_path) as f:
        header = f.readline().split()
        ci = {n: i for i, n in enumerate(header)}
        for k, line in enumerate(f):
            if limit is not None and k >= limit:
                break
            r = line.split()
            n0, n1 = r[ci["Obj0_UsdName"]], r[ci["Obj1_UsdName"]]
            if n0 not in latents or n1 not in latents:
                continue
            s0 = torch.tensor([float(r[ci[f"Obj0_Size{a}"]]) for a in "XYZ"])
            s1 = torch.tensor([float(r[ci[f"Obj1_Size{a}"]]) for a in "XYZ"])
            p0 = [float(r[ci[f"Obj0_Pos{a}"]]) for a in "XYZ"]
            p1 = [float(r[ci[f"Obj1_Pos{a}"]]) for a in "XYZ"]
            x = torch.stack([
                torch.cat([latents[n0], s0, torch.tensor([1.0, 0.0])]),
                torch.cat([latents[n1], s1, torch.tensor([0.0, 1.0])]),
            ])
            edge_attr = torch.tensor([_edge_features(p0, p1), _edge_features(p1, p0)])
            edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
            y = torch.tensor([[float(r[ci[t]]) for t in target_flags]])
            y_good = torch.tensor([[float(r[ci["Assembly_Good?"]])]])
            graphs.append(Data(x=x, edge_index=edge_index, edge_attr=edge_attr,
                               y=y, y_good=y_good))
            names.append(n1)
    return graphs, names


def split_indices(n, seed=42):
    """Reproduce the split of the training run, so its numbers and these are comparable.

    It must stay identical to the one in the training module, including the seed and the order
    in which the graphs were built, or the test split here would contain training data.
    """
    idx = list(range(n))
    random.Random(seed).shuffle(idx)
    n_tr, n_va = int(0.8 * n), int(0.1 * n)
    return {"train": idx[:n_tr], "val": idx[n_tr:n_tr + n_va],
            "test": idx[n_tr + n_va:], "all": list(range(n))}


@torch.no_grad()
def predict(model, graphs, device, num_fail, batch_size=1024):
    """Run the surrogate over a set of graphs.

    Returns:
        The predicted probabilities and the targets for the failure modes, shape
        ``(N, num_fail)`` each, and for the verdict, shape ``(N,)`` each.
    """
    model.eval()
    loader = DataLoader(graphs, batch_size=batch_size)
    fp, ft, gp, gt = [], [], [], []
    for batch in loader:
        batch = batch.to(device)
        fl, gl = model(batch.x, batch.edge_index, batch.edge_attr)
        fp.append(torch.sigmoid(fl).cpu().numpy())
        ft.append(batch.y.view(-1, num_fail).cpu().numpy())
        gp.append(torch.sigmoid(gl).cpu().numpy())
        gt.append(batch.y_good.view(-1, 1).cpu().numpy())
    return (np.concatenate(fp), np.concatenate(ft),
            np.concatenate(gp).ravel(), np.concatenate(gt).ravel())


def auroc(p, t):
    """Area under the receiver operating characteristic, undefined for a single-class group."""
    return float("nan") if t.max() == t.min() else float(roc_auc_score(t, p))


def ap(p, t):
    """Average precision, undefined for a single-class group."""
    return float("nan") if t.max() == t.min() else float(average_precision_score(t, p))


def main():
    """Evaluate the checkpoint on the requested split and print the table grouped by shape."""
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--ckpt", default=DEFAULT_CKPT)
    ap_.add_argument("--txt", default=DEFAULT_TXT)
    ap_.add_argument("--split", default="test", choices=["train", "val", "test", "all"])
    ap_.add_argument("--limit", type=int, default=None)
    ap_.add_argument("--csv", default=None, help="write the per-shape table to CSV")
    args = ap_.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(ROOT / args.ckpt, map_location=device, weights_only=False)
    fail_names = ck["fail_names"]
    num_fail = ck.get("num_fail", len(fail_names))
    thr = ck["thresholds"]
    target_flags = [f"ObjNew_{n}" for n in fail_names]
    print(f"ckpt: {args.ckpt} | targets: {fail_names} | split: {args.split}")

    lp = ck["latents_path"]
    lp = [lp] if isinstance(lp, str) else lp
    latents = load_shape_latents([ROOT / p for p in lp])
    # The code tables recorded in the checkpoint must give the node width the model was built
    # for. A mismatch means the tables have changed since training and the codes would be wrong
    # without anything failing outright.
    assert node_dim(latents) == ck["node_dim"], "latent/ckpt node_dim mismatch"

    model = SimAssemblyGNN(
        node_dim=ck["node_dim"], edge_dim=ck["edge_dim"], hidden_dim=ck["hidden_dim"],
        heads=ck["heads"], dropout=ck["dropout"], head_hidden=ck["head_hidden"],
        num_fail=num_fail,
    ).to(device)
    model.load_state_dict(ck["model"])

    graphs, names = parse_with_names(ROOT / args.txt, latents, target_flags, args.limit)
    names = np.array(names)
    sel = split_indices(len(graphs))[args.split]
    graphs = [graphs[i] for i in sel]
    names = names[sel]
    print(f"{len(graphs)} graphs in split")

    fp, ft, gp, gt = predict(model, graphs, device, num_fail)

    cols = fail_names + ["Assembly_Good"]
    thr_by_col = {n: thr["fail"][i] for i, n in enumerate(fail_names)}
    thr_by_col["Assembly_Good"] = thr["good"]

    def probs_tgts(col):
        """Predictions and targets of one column, whether it is a failure mode or the verdict."""
        if col == "Assembly_Good":
            return gp, gt
        j = fail_names.index(col)
        return fp[:, j], ft[:, j]

    def row_metrics(mask):
        """Every measure for the subset the mask selects, plus how large that subset is."""
        out = {"n": int(mask.sum())}
        for c in cols:
            p, t = probs_tgts(c)
            p, t = p[mask], t[mask]
            out[f"{c}:AUROC"] = auroc(p, t)
            out[f"{c}:AP"] = ap(p, t)
            out[f"{c}:F1"] = f1_score(t, p > thr_by_col[c], zero_division=0) if t.max() != t.min() else float("nan")
            # The base rate is reported alongside, because a group in which a target almost
            # never occurs will show a poor precision whatever the surrogate does.
            out[f"{c}:rate"] = float(t.mean())
        return out

    shapes = sorted(set(names.tolist()))
    all_mask = np.ones(len(names), bool)
    # The whole split comes first, so the per-shape rows are read against it rather than alone.
    rows = [("ALL", row_metrics(all_mask))]
    for s in shapes:
        rows.append((s.replace(".usd", ""), row_metrics(names == s)))

    hdr = f"{'shape':<14} {'n':>6} " + " ".join(f"{c.split('_')[0][:4]:>6}" for c in fail_names) + f" {'GOOD':>6} {'G_F1':>6} {'G_rt':>5}"
    print("\n" + "=" * len(hdr))
    print("AUROC per target (+ Assembly_Good F1 and base rate)")
    print(hdr)
    print("-" * len(hdr))
    def f6(v):
        """Format a measure, showing a dash where it was undefined for the group."""
        return "     -" if v != v else f"{v:6.3f}"
    for name, m in rows:
        aurocs = " ".join(f6(m[f"{c}:AUROC"]) for c in fail_names)
        print(f"{name:<14} {m['n']:>6} {aurocs} {f6(m['Assembly_Good:AUROC'])} "
              f"{f6(m['Assembly_Good:F1'])} {m['Assembly_Good:rate']:5.2f}")
    print("=" * len(hdr))

    if args.csv:
        import csv
        keys = ["shape", "n"] + [f"{c}:{k}" for c in cols for k in ("AUROC", "AP", "F1", "rate")]
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for name, m in rows:
                w.writerow({"shape": name, **m})
        print(f"wrote {args.csv}")


if __name__ == "__main__":
    main()
