"""Turn the labelled pair dataset into the latent graphs the surrogate is trained on.

Each row of the dataset is one configuration of two blocks and becomes one two-node graph: node
0 is the frozen parent, node 1 the part under test, joined by the two directed edges that carry
their offset. What distinguishes this branch is what a node holds. The three edge lengths are
sent through the frozen box encoder and the resulting code takes their place, so the surrogate
never sees a size. Everything else, the edge features, the targets and the split, is the same
as in the parameter branch.

This module also owns the encoder cache the whole pipeline shares, the standardisation of the
targets, and the class weight of the feasibility label. The repair rebuilds the same graph
layout by hand in ``src/repair_optimizer.py``; the two must not drift apart.
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
from src.enc_box import BoxEncoder
from src.simulation_dataset import get_scale_factor, manipulate_data, read_txt_file

DATA_FOLDER = Path(__file__).parent.parent / config.data.data_folder
ENCODER_DIR = Path(__file__).parent.parent / config.autoencoder.autoencoder_folder
LATENT_DIM = int(config.autoencoder.latent_dim)
ENCODER_CKPT = ENCODER_DIR / f"best_encoder_decoder_latentdim{LATENT_DIM}.pth"

# Cached frozen encoder, loaded lazily on first call.
_BOX_ENCODER: BoxEncoder | None = None


def get_box_encoder(device: str = "cpu") -> BoxEncoder:
    """Return the frozen box encoder, loading and caching it on first call.

    Every part of the pipeline that needs a code goes through this function, so that dataset
    construction, training and repair are all working in the same latent space. A surrogate
    trained on one encoder's codes is meaningless under another's.

    Args:
        device: Device the encoder is moved to on the first call. Later calls return the cached
            instance and ignore this argument.

    Raises:
        SystemExit: If the checkpoint is missing or carries no ``encoder`` state dict.
    """
    global _BOX_ENCODER
    if _BOX_ENCODER is None:
        if not ENCODER_CKPT.exists():
            raise SystemExit(
                f"BoxEncoder checkpoint not found: {ENCODER_CKPT}\n"
                "Train the auto-encoder first (src/enc_dec_training.py)."
            )
        ckpt = torch.load(ENCODER_CKPT, map_location="cpu", weights_only=False)
        if "encoder" not in ckpt:
            raise SystemExit(
                f"Encoder checkpoint {ENCODER_CKPT.name} has no 'encoder' key; "
                f"available keys: {list(ckpt.keys())}"
            )
        encoder = BoxEncoder(latent_dim=LATENT_DIM)
        encoder.load_state_dict(ckpt["encoder"])
        encoder.eval()
        for p in encoder.parameters():
            p.requires_grad_(False)
        _BOX_ENCODER = encoder.to(device)
        print(f" Loaded frozen BoxEncoder: {ENCODER_CKPT.name} (latent_dim={LATENT_DIM})")
    return _BOX_ENCODER


@torch.no_grad()
def encode_sizes_to_z(sizes_m: torch.Tensor, device: str = "cpu") -> torch.Tensor:
    """Encode a batch of blocks into codes.

    Args:
        sizes_m: Full edge lengths, shape ``(B, 3)`` in metres. The encoder is defined on
            half-extents, so the halving happens here.
        device: Device the encoding runs on.

    Returns:
        Codes of shape ``(B, latent_dim)``.
    """
    encoder = get_box_encoder(device=device)
    half = (sizes_m / 2.0).to(device=device, dtype=torch.float)
    return encoder(half)

# Added before the logarithm so that an overlap of exactly zero stays finite. Zero occurs, for
# blocks that do not touch tangentially at all. The targets stored on a graph remain in metres;
# the logarithm is applied only where the targets are standardised, and inverted in
# src/repair_optimizer.py, which reads the same constant from there.
OVERLAP_LOG_EPS: float = 1e-5

# Optional rebalancing before the split: blocks far above the thickness limit dominate the
# dataset and carry little information, since they fail for an obvious reason. Dropping a
# fraction of them is off by default, and both values can be overridden per run without
# touching the file.
DROP_THICK_FAR_THRESHOLD_M: float = float(
    os.environ.get("GNN_DROP_THICK_FAR_THRESHOLD_M", "0.025")
)
DROP_THICK_FAR_FRACTION: float = float(
    os.environ.get("GNN_DROP_THICK_FAR_FRACTION", "0.0")
)


def prepare_simulation_dataset(
    file_path: Path = DATA_FOLDER, file_name: str = config.data.data_file
) -> tuple[pd.DataFrame, float]:
    """Read the labelled dataset, check its schema and optionally rebalance it.

    Returns:
        The prepared table and the factor between the frame the pipeline works in and metres.
        The factor is one: the surrogate is trained on metres directly, and the attention
        layers absorb the magnitude of the inputs in their weights.

    Raises:
        SystemExit: If the table carries the gap and table-clearance columns of an older
            schema, or is missing one of the columns the targets are built from. Both are fatal
            rather than warnings, because either would produce a surrogate trained on something
            other than what the repair later queries.
    """
    df: pd.DataFrame = read_txt_file(Path(file_path), file_name)
    df = manipulate_data(df)

    print(f" Using dataset: {file_name}")

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
            # Seeded from the configuration so that two runs of the same setting drop the same
            # rows and the split downstream stays reproducible.
            rng = np.random.default_rng(config.general.random_seed or 42)
            drop_idx = rng.choice(thick_far_idx, size=n_drop, replace=False)
            df = df.drop(index=drop_idx).reset_index(drop=True)
        print(
            f" Dropped {n_drop} / {n_thick_far} samples with thickness > "
            f"{DROP_THICK_FAR_THRESHOLD_M * 1000:.0f} mm  "
            f"(fraction={DROP_THICK_FAR_FRACTION:.2f}). "
            f"Dataset: {n_before} -> {len(df)} ({len(df) / n_before:.0%})"
        )

    # The factor is threaded through the pipeline instead of being assumed, because every
    # inference path has to feed the surrogate at the same scale it was trained on.
    scale_factor: float = 1.0
    print(" Using scale_factor = 1.0 (raw metres, matches 164136 checkpoint)")

    return df, scale_factor


def _per_axis_overlap_array(df: pd.DataFrame) -> np.ndarray:
    """Per-axis overlap between the two blocks of every row, ``(N, 3)`` in metres."""
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
    """Build the graph of one configuration.

    Returns:
        A graph whose node features have shape ``(2, latent_dim + 2)``, the code of each block
        followed by the two-entry one-hot that marks parent against part under test; edge
        features of shape ``(2, 6)``, the offset and the three axis-pair distances in metres,
        with the offset negated on the reverse edge; ``y`` of shape ``(1, 2)``, the measured
        overlap and thickness in metres; and ``y_good`` of shape ``(1,)``, the feasibility
        label.
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

    z_pair = encode_sizes_to_z(torch.stack([size_0, size_1]), device=device)
    z_0, z_1 = z_pair[0], z_pair[1]

    type_0 = torch.tensor([1.0, 0.0], dtype=torch.float, device=device)
    type_1 = torch.tensor([0.0, 1.0], dtype=torch.float, device=device)

    x = torch.stack([
        torch.cat([z_0, type_0]),
        torch.cat([z_1, type_1]),
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
    """Build the graph of every configuration in the dataset, in one pass.

    Same result as calling :func:`build_graph_from_row` on every row, but the columns are read
    as arrays and every block is encoded in a single call, which matters because the encoder is
    invoked once per block and there are two per row.

    Args:
        device: Accepted but ignored. Graphs are built on the host and moved to the training
            device by the data loader.
        data_file: Dataset to read instead of ``config.data.data_file``.
    """
    df: pd.DataFrame
    scale_factor: float
    if data_file is not None:
        df, scale_factor = prepare_simulation_dataset(file_name=data_file)
    else:
        df, scale_factor = prepare_simulation_dataset()

    # Truncating before the graphs are built, not after, so that a quick check costs a fraction
    # of the construction time rather than all of it.
    quick_n_env = os.environ.get("GNN_QUICK_N")
    if os.environ.get("GNN_QUICK", "0") == "1" and quick_n_env:
        n_keep = int(quick_n_env)
        df = df.iloc[:n_keep].reset_index(drop=True)
        print(f"GNN_QUICK: dataframe truncated to {len(df)} rows.")

    n: int = len(df)
    sf: float = scale_factor
    print(
        f"Building graphs from {n} assemblies "
        f"(node_dim={LATENT_DIM + 2}, edge_dim=6, 2-target reg + mode aux)..."
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

    size_0_all = torch.from_numpy(np.stack([s0x_np, s0y_np, s0z_np], axis=1))
    size_1_all = torch.from_numpy(np.stack([s1x_np, s1y_np, s1z_np], axis=1))

    # Two calls for the whole dataset rather than two per row. The encoder is small enough that
    # doing this on the host costs less than moving the data to an accelerator would.
    enc_device = "cpu"
    z_0_all = encode_sizes_to_z(size_0_all, device=enc_device).cpu()
    z_1_all = encode_sizes_to_z(size_1_all, device=enc_device).cpu()

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
        z_0 = z_0_all[i]
        z_1 = z_1_all[i]
        x: torch.Tensor = torch.stack([
            torch.cat([z_0, type_0]),
            torch.cat([z_1, type_1]),
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
    """Standardise the two regression targets, from the training split alone.

    Overlap is standardised after a logarithm and thickness is not. Overlap spans orders of
    magnitude and the criterion sits near its lower end, where a millimetre matters and an
    absolute error does not distinguish a near miss from a gross one; thickness varies over a
    much narrower band and needs no such transform.

    Nothing records which column was transformed, so the sign of the mean is what identifies it
    downstream: a logarithm of a length in metres is negative, an untransformed one is not.
    ``destandardize`` in ``src/repair_optimizer.py`` relies on exactly that.

    Args:
        train_graphs: The training split only. Using any other split here would leak the
            evaluation data into the scale the model is fitted on.

    Returns:
        A dictionary with ``mean`` and ``std``, both ``(2,)``, overlap then thickness.
    """
    Y = torch.stack([g.y.squeeze(0) for g in train_graphs])
    Y_for_stats = Y.clone()
    Y_for_stats[:, 0] = torch.log(Y[:, 0] + OVERLAP_LOG_EPS)
    mean = Y_for_stats.mean(dim=0)
    # Clamped so that a degenerate target, one that happens to be constant over the split,
    # cannot produce a division by zero that would silently poison every later prediction.
    std = Y_for_stats.std(dim=0).clamp_min(1e-6)
    print("Target stats (TRAIN, overlap in log-space, thickness raw):")
    names = ["log_overlap", "thickness"]
    for i, n in enumerate(names):
        print(f"  {n:14s} mean={mean[i].item():+.5f}  std={std[i].item():.5f}")
    return {"mean": mean, "std": std}


def calculate_class_weights(graphs: list[Data]) -> torch.Tensor:
    """Return the weight the feasible class gets in the feasibility loss, as a ``(1,)`` tensor.

    The ratio of infeasible to feasible configurations, so that the smaller class is not simply
    ignored. Feasible pairs are the minority in this dataset, and they are the ones the repair
    has to be able to recognise.
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
    """Split the graphs into training, validation and test sets, in that order.

    The permutation comes from the global torch generator, which the caller seeds from
    ``config.general.random_seed``. Other parts of the pipeline reproduce the same split by
    seeding identically, which is what lets the repair be evaluated on assemblies the surrogate
    never saw.
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
