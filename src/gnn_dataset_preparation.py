"""Turn the labelled configuration table into the graphs the surrogate is trained on.

In goes the text dataset assembled by ``tools/build_mixed_dataset.py``: one row per
configuration, with the position and the three edge lengths of both blocks, the two measured
quantities and the feasibility flag. Out come two-node
:class:`torch_geometric.data.Data` objects, node 0 the anchor and node 1 the part under test,
plus the target statistics and the class weights the training objective needs.

The module also owns the two conventions the rest of the code has to agree with: geometry is
passed to the model in raw metres, and the overlap target is standardised in logarithmic
space. Both are properties of the checkpoint in ``gnn_models/``, so a consumer that departs
from them queries the surrogate outside the regime it was fitted in.
"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

sys.path.insert(0, str(Path(__file__).parent.parent))
from config.config import config
from src.simulation_dataset import get_scale_factor, manipulate_data, read_txt_file

DATA_FOLDER = Path(__file__).parent.parent / config.data.data_folder

# An overlap of exactly zero is a common label, so the logarithm needs an offset to stay
# finite. ``graph.y`` keeps raw metres; the transform is applied inside the loss and the
# standardisation only, and inverted again at inference.
OVERLAP_LOG_EPS: float = 1e-5

# Configurations whose thickness is far past the limit are the easiest negatives and the most
# numerous ones. Dropping part of that bucket before the split rebalances the classes. The
# default fraction is zero, so the behaviour is opt-in through the environment.
DROP_THICK_FAR_THRESHOLD_M: float = float(
    os.environ.get("GNN_DROP_THICK_FAR_THRESHOLD_M", "0.025")
)
DROP_THICK_FAR_FRACTION: float = float(
    os.environ.get("GNN_DROP_THICK_FAR_FRACTION", "0.0")
)


def prepare_simulation_dataset(
    file_path: Path = DATA_FOLDER, file_name: str = config.data.data_file
) -> tuple[pd.DataFrame, float]:
    """Read the labelled configuration table and check that it has the expected schema.

    Args:
        file_path: Directory holding the dataset file.
        file_name: Name of the dataset file.

    Returns:
        The table with booleans mapped to zero and one, and the factor by which lengths are
        multiplied before they reach the model. That factor is one: geometry stays in raw
        metres.

    Raises:
        SystemExit: If the table lacks one of the required columns, or still carries the
            columns of the earlier four-criterion schema. Both cases would train a surrogate
            whose outputs no longer mean what the repair assumes they mean.
    """
    df: pd.DataFrame = read_txt_file(Path(file_path), file_name)
    df = manipulate_data(df)

    print(f" Using dataset: {file_name}")

    # Part I assesses two criteria only. A table that still carries table interference or
    # planned gap comes from an older generator and would not match the surrogate's heads.
    forbidden = [c for c in df.columns if "UnderSurface" in c or "PlannedGap" in c]
    if forbidden:
        raise SystemExit(
            f"Dataset {file_name} still contains legacy columns {forbidden}. "
            "Regenerate with the paper-v0 dataset_generation.py."
        )
    required = ["Block1_Overlap_m", "Block1_Thickness_m", "Assembly_Good?"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SystemExit(f"Dataset {file_name} missing required columns: {missing}")

    n_before = len(df)
    if DROP_THICK_FAR_FRACTION > 0.0:
        is_thick_far = df["Block1_Thickness_m"] > DROP_THICK_FAR_THRESHOLD_M
        n_thick_far = int(is_thick_far.sum())
        n_drop = int(n_thick_far * DROP_THICK_FAR_FRACTION)
        if n_drop > 0:
            thick_far_idx = df.index[is_thick_far].to_numpy()
            rng = np.random.default_rng(config.general.random_seed or 42)
            drop_idx = rng.choice(thick_far_idx, size=n_drop, replace=False)
            df = df.drop(index=drop_idx).reset_index(drop=True)
        print(
            f" Dropped {n_drop} / {n_thick_far} samples with thickness > "
            f"{DROP_THICK_FAR_THRESHOLD_M * 1000:.0f} mm  "
            f"(fraction={DROP_THICK_FAR_FRACTION:.2f}). "
            f"Dataset: {n_before} -> {len(df)} ({len(df) / n_before:.0%})"
        )

    # Lengths are handed to the model unscaled. The checkpoint in gnn_models/ was fitted this
    # way, so every inference path -- training, analysis, repair -- has to feed the same
    # scale, which is why the factor is returned explicitly rather than left implicit.
    scale_factor: float = 1.0
    print(" Using scale_factor = 1.0 (raw metres, matches 164136 checkpoint)")

    return df, scale_factor


def _per_axis_overlap_array(df: pd.DataFrame) -> np.ndarray:
    """Per-axis interval overlap between the two blocks, shape ``(N, 3)`` in metres."""
    arr = np.zeros((len(df), 3), dtype=np.float32)
    for i, ax in enumerate(("X", "Y", "Z")):
        p0 = df[f"Block0_Pos{ax}"].values
        s0 = df[f"Block0_Size{ax}"].values
        p1 = df[f"Block1_Pos{ax}"].values
        s1 = df[f"Block1_Size{ax}"].values
        bot0, top0 = p0 - s0 / 2.0, p0 + s0 / 2.0
        bot1, top1 = p1 - s1 / 2.0, p1 + s1 / 2.0
        ov = np.maximum(0.0, np.minimum(top0, top1) - np.maximum(bot0, bot1))
        arr[:, i] = ov.astype(np.float32)
    return arr


def build_graph_from_row(
    row: pd.Series,
    scale_factor: float,
    device: str = config.general.device,
) -> Data:
    """Build the configuration graph of one row.

    Node 0 is the anchor and node 1 the part under test; the two directed edges carry the
    offset with opposite sign but the same axis-pair distances.

    Args:
        row: One row of the labelled table.
        scale_factor: Multiplier applied to every length before it enters the graph.
        device: Device the feature tensors are created on.

    Returns:
        A graph with ``x`` of shape ``(2, 5)``, ``edge_attr`` of shape ``(2, 6)``, the
        regression target ``y`` of shape ``(1, 2)`` in metres and the feasibility label
        ``y_good`` of shape ``(1,)``.
    """
    sf = float(scale_factor)

    size_0 = torch.tensor(
        [row["Block0_SizeX"] * sf, row["Block0_SizeY"] * sf, row["Block0_SizeZ"] * sf],
        dtype=torch.float, device=device,
    )
    size_1 = torch.tensor(
        [row["Block1_SizeX"] * sf, row["Block1_SizeY"] * sf, row["Block1_SizeZ"] * sf],
        dtype=torch.float, device=device,
    )

    type_0 = torch.tensor([1.0, 0.0], dtype=torch.float, device=device)
    type_1 = torch.tensor([0.0, 1.0], dtype=torch.float, device=device)

    x = torch.stack([
        torch.cat([size_0, type_0]),
        torch.cat([size_1, type_1]),
    ])

    dx = (row["Block1_PosX"] - row["Block0_PosX"]) * sf
    dy = (row["Block1_PosY"] - row["Block0_PosY"]) * sf
    dz = (row["Block1_PosZ"] - row["Block0_PosZ"]) * sf
    dist_xy = float(np.sqrt(dx**2 + dy**2))
    dist_xz = float(np.sqrt(dx**2 + dz**2))
    dist_yz = float(np.sqrt(dy**2 + dz**2))
    edge_attr = torch.stack([
        torch.tensor([dx, dy, dz, dist_xy, dist_xz, dist_yz], dtype=torch.float, device=device),
        torch.tensor([-dx, -dy, -dz, dist_xy, dist_xz, dist_yz], dtype=torch.float, device=device),
    ])
    edge_index: torch.Tensor = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)

    y: torch.Tensor = torch.tensor(
        [row["Block1_Overlap_m"], row["Block1_Thickness_m"]],
        dtype=torch.float,
    ).unsqueeze(0)

    y_good: torch.Tensor = torch.tensor(
        [float(row["Assembly_Good?"])], dtype=torch.float,
    )

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, y_good=y_good)


def build_graph_dataset(
    device: str = "cpu",
    data_file: str | None = None,
) -> list[Data]:
    """Build one configuration graph per row of the dataset.

    Args:
        device: Ignored. Graphs are built on the host and moved by the data loader; the
            argument is kept so existing callers still work.
        data_file: Dataset file to read instead of the one named in the configuration, used
            when continuing training on a narrower set.

    Returns:
        One graph per row, in file order.
    """
    df: pd.DataFrame
    scale_factor: float
    if data_file is not None:
        df, scale_factor = prepare_simulation_dataset(file_name=data_file)
    else:
        df, scale_factor = prepare_simulation_dataset()

    # Truncating here rather than after the loop keeps a local sanity check cheap: the graphs
    # that are never used are not built in the first place.
    quick_n_env = os.environ.get("GNN_QUICK_N")
    if os.environ.get("GNN_QUICK", "0") == "1" and quick_n_env:
        n_keep = int(quick_n_env)
        df = df.iloc[:n_keep].reset_index(drop=True)
        print(f"GNN_QUICK: dataframe truncated to {len(df)} rows.")

    n: int = len(df)
    sf: float = scale_factor
    print(
        f"Building graphs from {n} assemblies "
        f"(node_dim=5, edge_dim=6, 2-target reg + mode aux)..."
    )

    s0x_np = (df["Block0_SizeX"].values * sf).astype(np.float32)
    s0y_np = (df["Block0_SizeY"].values * sf).astype(np.float32)
    s0z_np = (df["Block0_SizeZ"].values * sf).astype(np.float32)
    s1x_np = (df["Block1_SizeX"].values * sf).astype(np.float32)
    s1y_np = (df["Block1_SizeY"].values * sf).astype(np.float32)
    s1z_np = (df["Block1_SizeZ"].values * sf).astype(np.float32)

    dx_np = ((df["Block1_PosX"] - df["Block0_PosX"]).values * sf).astype(np.float32)
    dy_np = ((df["Block1_PosY"] - df["Block0_PosY"]).values * sf).astype(np.float32)
    dz_np = ((df["Block1_PosZ"] - df["Block0_PosZ"]).values * sf).astype(np.float32)
    dist_xy_np = np.sqrt(dx_np**2 + dy_np**2).astype(np.float32)
    dist_xz_np = np.sqrt(dx_np**2 + dz_np**2).astype(np.float32)
    dist_yz_np = np.sqrt(dy_np**2 + dz_np**2).astype(np.float32)

    y_reg_np = df[["Block1_Overlap_m", "Block1_Thickness_m"]].values.astype(np.float32)
    y_good_np = df["Assembly_Good?"].values.astype(np.float32)

    s0x_t = torch.from_numpy(s0x_np); s0y_t = torch.from_numpy(s0y_np); s0z_t = torch.from_numpy(s0z_np)
    s1x_t = torch.from_numpy(s1x_np); s1y_t = torch.from_numpy(s1y_np); s1z_t = torch.from_numpy(s1z_np)
    dx_t = torch.from_numpy(dx_np); dy_t = torch.from_numpy(dy_np); dz_t = torch.from_numpy(dz_np)
    dist_xy_t = torch.from_numpy(dist_xy_np)
    dist_xz_t = torch.from_numpy(dist_xz_np)
    dist_yz_t = torch.from_numpy(dist_yz_np)
    y_t = torch.from_numpy(y_reg_np)
    y_good_t = torch.from_numpy(y_good_np)

    type_0: torch.Tensor = torch.tensor([1.0, 0.0], dtype=torch.float32)
    type_1: torch.Tensor = torch.tensor([0.0, 1.0], dtype=torch.float32)
    edge_index: torch.Tensor = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)

    graphs: list[Data] = []
    for i in range(n):
        size_0 = torch.stack([s0x_t[i], s0y_t[i], s0z_t[i]])
        size_1 = torch.stack([s1x_t[i], s1y_t[i], s1z_t[i]])
        x: torch.Tensor = torch.stack([
            torch.cat([size_0, type_0]),
            torch.cat([size_1, type_1]),
        ])

        dxi = dx_t[i]; dyi = dy_t[i]; dzi = dz_t[i]
        dxyi = dist_xy_t[i]; dxzi = dist_xz_t[i]; dyzi = dist_yz_t[i]
        edge_attr: torch.Tensor = torch.stack([
            torch.stack([dxi, dyi, dzi, dxyi, dxzi, dyzi]),
            torch.stack([-dxi, -dyi, -dzi, dxyi, dxzi, dyzi]),
        ])

        graph: Data = Data(
            x=x, edge_index=edge_index, edge_attr=edge_attr,
            y=y_t[i].unsqueeze(0),
            y_good=y_good_t[i].view(1),
        )
        graphs.append(graph)

        if (i + 1) % 30000 == 0:
            print(f"  Built {i + 1}/{n} graphs")

    print(f"Built {len(graphs)} graphs total")
    return graphs


def compute_target_stats(train_graphs: list[Data]) -> dict[str, torch.Tensor]:
    """Mean and standard deviation of the two regression targets, over the training split.

    Only the training split is used, so the standardisation carries no information about the
    held-out data.

    The overlap column is standardised in logarithmic space, the thickness column directly.
    Consumers detect which is which from the sign of the first mean, which is negative for a
    logarithm of a length in metres; :func:`src.repair_optimizer.destandardize` relies on that.

    Args:
        train_graphs: Graphs of the training split.

    Returns:
        A mapping with ``mean`` and ``std``, both of shape ``(2,)``.
    """
    Y = torch.stack([g.y.squeeze(0) for g in train_graphs])
    Y_for_stats = Y.clone()
    Y_for_stats[:, 0] = torch.log(Y[:, 0] + OVERLAP_LOG_EPS)
    mean = Y_for_stats.mean(dim=0)
    std = Y_for_stats.std(dim=0).clamp_min(1e-6)
    print("Target stats (TRAIN, overlap in log-space, thickness raw):")
    names = ["log_overlap", "thickness"]
    for i, n in enumerate(names):
        print(f"  {n:14s} mean={mean[i].item():+.5f}  std={std[i].item():.5f}")
    return {"mean": mean, "std": std}


def calculate_class_weights(graphs: list[Data]) -> torch.Tensor:
    """Weight of the positive class for the feasibility head.

    Returns:
        A one-element tensor holding the ratio of infeasible to feasible configurations, to be
        passed as ``pos_weight`` to the binary cross entropy.
    """
    labels: torch.Tensor = torch.stack([g.y_good for g in graphs])
    num_neg: float = (labels == 0).sum().float().item()
    num_pos: float = (labels == 1).sum().float().item()
    pos_weight: torch.Tensor = torch.tensor([num_neg / max(num_pos, 1)])
    print(
        f"  assembly_good: pos={num_pos:.0f}, neg={num_neg:.0f}, "
        f"pos_weight={pos_weight.item():.3f}"
    )
    return pos_weight


def split_graphs(
    graphs: list[Data],
    train_split: float = config.training.train_split,
    val_split: float = config.training.val_split,
) -> tuple[list[Data], list[Data], list[Data]]:
    """Split the graphs into training, validation and test set.

    The permutation is drawn from the global torch generator, so seeding it reproduces the
    same split. Every offline evaluation on this branch re-creates the split that way instead
    of storing indices.

    Returns:
        The three splits, in the order training, validation, test.
    """
    n: int = len(graphs)
    perm: list[int] = torch.randperm(n).tolist()

    train_end: int = int(train_split * n)
    val_end: int = int((train_split + val_split) * n)

    train_graphs: list[Data] = [graphs[i] for i in perm[:train_end]]
    val_graphs: list[Data] = [graphs[i] for i in perm[train_end:val_end]]
    test_graphs: list[Data] = [graphs[i] for i in perm[val_end:]]

    print(
        f"Split: {len(train_graphs)} train, {len(val_graphs)} val, {len(test_graphs)} test"
    )
    return train_graphs, val_graphs, test_graphs
