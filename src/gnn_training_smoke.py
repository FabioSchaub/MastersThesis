"""Short local run of the surrogate training, to check that the pipeline still works.

Builds a subset of the latent graph dataset and trains for a few epochs, reusing the loss, the
epoch loop and the metrics of ``src/gnn_training.py`` unchanged, so that what is exercised here
is the real training path and not a copy of it. Nothing is logged to Weights and Biases and no
checkpoint is written, which is what keeps a smoke run from being mistaken for a trained
surrogate.

Prints one line per epoch, then the stratified errors and the spread ratios of the last epoch.
Runs on a CPU with the defaults.

    python src/gnn_training_smoke.py
    python src/gnn_training_smoke.py --num-graphs 10000 --epochs 5
    python src/gnn_training_smoke.py --batch-size 256 --device cpu
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.config import config  # noqa: E402
from src.gnn import GNN  # noqa: E402
from src.gnn_dataset_preparation import (  # noqa: E402
    build_graph_dataset,
    calculate_class_weights,
    compute_target_stats,
    split_graphs,
)
from src.gnn_training import (  # noqa: E402
    calculate_mode_pos_weights,
    random_seed,
    train_one_epoch,
    validate,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-graphs", type=int, default=5000,
                        help="Subset size from the full dataset (default: 5000)")
    parser.add_argument("--epochs", type=int, default=3,
                        help="Number of epochs (default: 3)")
    parser.add_argument("--batch-size", type=int, default=256,
                        help="GNN batch size (default: 256, CPU-friendly)")
    parser.add_argument("--device", type=str, default=None,
                        help="cuda or cpu (default: auto)")
    parser.add_argument("--seed", type=int, default=config.general.random_seed or 42)
    args = parser.parse_args()

    random_seed(args.seed)
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Smoke run: {args.num_graphs} graphs, {args.epochs} epochs, "
          f"batch_size={args.batch_size}, device={device}")

    # Set before the dataset is built: the subset has to be taken while the dataframe is read,
    # otherwise the full file is encoded first and the run is no longer quick.
    os.environ["GNN_QUICK"] = "1"
    os.environ["GNN_QUICK_N"] = str(args.num_graphs)

    t0 = time.time()
    graphs: list[Data] = build_graph_dataset(device=str(device))
    if len(graphs) > args.num_graphs:
        graphs = graphs[: args.num_graphs]
    print(f"Built {len(graphs)} graphs in {time.time() - t0:.2f}s")

    train_graphs, val_graphs, _ = split_graphs(graphs)

    target_stats = compute_target_stats(train_graphs)
    pos_weight = calculate_class_weights(train_graphs)
    mode_pos_weight = calculate_mode_pos_weights(train_graphs)

    train_loader = DataLoader(train_graphs, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=args.batch_size)

    model = GNN(
        node_dim=config.gnn.node_dim,
        edge_dim=config.gnn.edge_dim,
        hidden_dim=config.gnn.hidden_dim,
        heads=config.gnn.heads,
        dropout=config.gnn.dropout,
    ).to(device)
    # Required by the shared loss, which reads them off the model to standardise its targets.
    model.register_buffer("target_mean", target_stats["mean"].to(device))
    model.register_buffer("target_std", target_stats["std"].to(device))
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=config.training.learning_rate)

    print(f"\n{'epoch':>5} {'train_loss':>10} {'val_loss':>10} "
          f"{'mae_ov':>8} {'mae_th':>8} {'F1_thr':>7} {'F1_bin':>7} {'time(s)':>7}")
    print("-" * 80)
    for epoch in range(args.epochs):
        t0 = time.time()
        train_loss, _ = train_one_epoch(
            model, train_loader, optimizer, device, pos_weight, mode_pos_weight,
        )
        val_loss, val_metrics = validate(
            model, val_loader, device, pos_weight, mode_pos_weight,
        )
        dt = time.time() - t0
        print(
            f"{epoch+1:>5} {train_loss:>10.5f} {val_loss:>10.5f} "
            f"{val_metrics['mae_overlap']:>8.5f} {val_metrics['mae_thickness']:>8.5f} "
            f"{val_metrics['thresh_f1']:>7.4f} {val_metrics['binary_f1']:>7.4f} "
            f"{dt:>7.2f}"
        )

    print("\nLast-epoch stratified MAE (mm) / slope:")
    for k, v in sorted(val_metrics.items()):
        if k.startswith("strat/") and k.endswith("/mae"):
            tag = k.replace("strat/", "").replace("/mae", "")
            slope = val_metrics.get(k.replace("/mae", "/slope"), float("nan"))
            n = val_metrics.get(k.replace("/mae", "/n"), 0)
            print(f"  {tag:<40s}  MAE={v*1000:6.2f} mm  slope={slope:+.3f}  n={int(n)}")

    print("\nstd_ratio (target ~1.0):")
    print(f"  overlap   = {val_metrics['std_ratio_overlap']:.3f}")
    print(f"  thickness = {val_metrics['std_ratio_thickness']:.3f}")

    print("\nSmoke finished cleanly.")


if __name__ == "__main__":
    main()
