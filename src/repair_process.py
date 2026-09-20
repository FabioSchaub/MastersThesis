"""Turn an assembly of several blocks into a sequence of pair repairs, and run them in order.

The surrogate only ever sees two blocks. An assembly of more is handled by freezing the first
block and walking along the chain: for every later block, the one before it is the frozen
parent and the block itself is the part under test. A repaired block is committed and becomes
the parent of the next step, so the chain either reaches the end or stops at the first pair
that cannot be repaired. This is a decomposition, not a solution to the general problem: a
block that has to satisfy two parents at once cannot be expressed in it.

Each pair is handled in the same order. The contact face is inferred from the raw geometry, the
surrogate is asked once, and a pair it already accepts is committed untouched. Otherwise the
latent repair of ``src/repair_optimizer.py`` runs, the child is put back into face-to-face
contact, and the closed-form metrics of ``src/analytical_metrics.py`` are recorded beside the
prediction. Those metrics never steer the flow: the surrogate alone decides what is committed,
and the comparison of the two verdicts is what identifies a bluff, where the network accepts a
geometry the formulas reject, and an over-cautious refusal, where it rejects one they accept.

This module also pins the surrogate checkpoint the repair runs against, in
``PLACEHOLDER_GNN_NAME``. Running it directly evaluates the chain on infeasible assemblies of
the test split and writes ``results_paramter_optimization/repair_results_chain.csv``.
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
from config.config import config  # noqa: E402
from src.analytical_metrics import (  # noqa: E402
    analytical_overlap,
    analytical_thickness,
    infer_face,
    is_repaired_analytical,
    snap_to_contact_face,
)
from src.gnn import GNN  # noqa: E402
from src.repair_optimizer import (  # noqa: E402
    EVAL_TOLERANCE,
    THRESH_OVERLAP,
    THRESH_THICKNESS,
    adam_repair_batched,
    build_graph,
    destandardize,
    encode_size,
    is_repaired_batch,
)
from src.gnn_dataset_preparation import get_box_encoder  # noqa: E402
from src.simulation_dataset import (  # noqa: E402
    get_block_prefixes,
    get_scale_factor,
    manipulate_data,
    read_txt_file,
)

DATA_FOLDER: Path = Path(__file__).parent.parent / config.data.data_folder
os.makedirs(DATA_FOLDER, exist_ok=True)
GNN_MODEL_FOLDER: Path = Path(__file__).parent.parent / config.gnn.gnn_folder
os.makedirs(GNN_MODEL_FOLDER, exist_ok=True)

# The surrogate the repair of Part II is run against. It is named here rather than in the
# configuration so that the checkpoint behind a reported result is pinned in code and cannot be
# changed by editing a comment. A checkpoint has to match the encoder whose codes it was trained
# on, so a surrogate and an encoder cannot be recombined freely. The GNN_MODEL environment
# variable overrides it; if neither exists on disk, _find_latest_gnn takes over.
PLACEHOLDER_GNN_NAME: str = (
    "gnn_latentspace_node10_batchsize2048_20260608-124934.pth"
)


def _find_latest_gnn() -> str:
    """Fall back to the most recently written surrogate checkpoint, preferring the latent ones.

    The filenames follow the convention of ``src/gnn_training.py``:
    ``gnn_<tag>_node<node_dim>_batchsize<batch_size>_<timestamp>.pth``.

    Raises:
        SystemExit: If no checkpoint exists at all.
    """
    candidates = sorted(
        GNN_MODEL_FOLDER.glob("gnn_latentspace_*.pth"),
        key=lambda p: p.stat().st_mtime,
    )
    if not candidates:
        candidates = sorted(
            GNN_MODEL_FOLDER.glob("gnn_*node10*.pth"),
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


MODEL_NAME: str = os.environ.get("GNN_MODEL", PLACEHOLDER_GNN_NAME)

REG_NAMES: list[str] = ["overlap", "thickness"]


def load_models(device: torch.device) -> tuple[GNN, float]:
    """Load the surrogate and the two frozen networks the repair needs, and return the scale.

    Three networks are loaded here: the surrogate, the box encoder, and the box decoder that
    maps a code back to half-extents. The auxiliary signed-distance decoder of stage 1 is not
    among them; it plays no part in the repair. The encoder and the decoder are only touched to
    warm their caches, so that the first pair of a run does not pay for reading them from disk.

    Returns:
        The surrogate in evaluation mode, and the factor between the frame the pipeline works
        in and metres.

    Raises:
        AssertionError: If the checkpoint carried no target standardisation. Without it every
            prediction would be interpreted on the wrong scale, silently.
    """
    # Imported here rather than at the top to keep the import graph free of a cycle.
    from src.dec_box import get_box_decoder
    _ = get_box_encoder(device=str(device))
    _ = get_box_decoder(device=str(device))

    gnn_name = MODEL_NAME or _find_latest_gnn()
    if not (GNN_MODEL_FOLDER / gnn_name).exists():
        fallback = _find_latest_gnn()
        print(
            f"GNN checkpoint '{gnn_name}' not found in {GNN_MODEL_FOLDER}; "
            f"falling back to most recent: {fallback}"
        )
        gnn_name = fallback
    gnn_ckpt: dict = torch.load(
        GNN_MODEL_FOLDER / gnn_name, map_location="cpu", weights_only=False
    )
    print(f"Loading GNN checkpoint: {gnn_name}")

    gnn_model: GNN = GNN(
        node_dim=config.gnn.node_dim,
        edge_dim=config.gnn.edge_dim,
        hidden_dim=config.gnn.hidden_dim,
        heads=config.gnn.heads,
        dropout=config.gnn.dropout,
        head_hidden=config.gnn.head_hidden,
    )

    ckpt_state = gnn_ckpt.get("model", gnn_ckpt)
    # The standardisation of the targets is stored as a buffer inside the checkpoint. The
    # buffers have to exist before the state dict is loaded, and their width is taken from the
    # checkpoint so that a model trained on a different number of targets still loads.
    tm_ckpt = ckpt_state.get("target_mean")
    n_targets = int(tm_ckpt.shape[0]) if tm_ckpt is not None else 2
    gnn_model.register_buffer("target_mean", torch.zeros(n_targets))
    gnn_model.register_buffer("target_std", torch.ones(n_targets))

    # Entries are matched by name and shape and mismatches are reported rather than raised, so
    # that a checkpoint from a slightly different configuration can still be inspected. The
    # assertion below is what stops a genuinely wrong checkpoint from being used silently.
    model_state = gnn_model.state_dict()
    compatible_state = {}
    incompatible: list[tuple] = []
    for k, v in ckpt_state.items():
        if k in model_state:
            if v.size() == model_state[k].size():
                compatible_state[k] = v
            else:
                incompatible.append(
                    (k, tuple(v.size()), tuple(model_state[k].size()))
                )
        else:
            incompatible.append(
                (k, tuple(v.size()) if hasattr(v, "size") else None, None)
            )

    gnn_model.load_state_dict(compatible_state, strict=False)
    if incompatible:
        print("GNN checkpoint compatibility warnings:")
        for k, ckpt_shape, model_shape in incompatible:
            print(f"  - {k}: ckpt={ckpt_shape} model={model_shape}")

    gnn_model.eval().to(device)
    print(f"target_mean: {gnn_model.target_mean.cpu().numpy()}")
    print(f"target_std:  {gnn_model.target_std.cpu().numpy()}")
    assert (gnn_model.target_std > 1e-6).all(), "target_std not loaded properly"

    df: pd.DataFrame = read_txt_file(DATA_FOLDER, config.data.data_file)
    df = manipulate_data(df)
    # The pinned checkpoint was trained without rescaling, so the scaled frame and metres
    # coincide. The factor is threaded through the pipeline anyway, because a checkpoint
    # trained on rescaled data would need it and the conversions must not be scattered.
    scale_factor: float = 1.0
    print(f"Dataset: {config.data.data_file}  scale_factor=1.0 (raw metres)")

    return gnn_model, scale_factor


@dataclass
class BlockData:
    """One block of an assembly: its column prefix, its centre and its full edge lengths.

    Both tensors are ``(3,)`` in the scaled frame. Sizes are held rather than codes, because
    the design that goes in and the design that comes out are both stated in metres; the code
    exists only inside the repair.
    """
    prefix: str
    pos: torch.Tensor
    size: torch.Tensor

    @property
    def size_z(self) -> float:
        return float(self.size[2])

    @property
    def he(self) -> np.ndarray:
        """Half-extents as a ``(3,)`` array, the form the closed-form metrics take."""
        return (self.size / 2.0).cpu().numpy()


def _get_pos(
    row: pd.Series,
    prefix: str,
    scale_factor: float,
    device: torch.device,
) -> torch.Tensor:
    return torch.tensor(
        [
            row[f"{prefix}_PosX"] * scale_factor,
            row[f"{prefix}_PosY"] * scale_factor,
            row[f"{prefix}_PosZ"] * scale_factor,
        ],
        dtype=torch.float,
        device=device,
    )


def _get_size(
    row: pd.Series,
    prefix: str,
    scale_factor: float,
    device: torch.device,
) -> torch.Tensor:
    return torch.tensor(
        [
            row[f"{prefix}_SizeX"] * scale_factor,
            row[f"{prefix}_SizeY"] * scale_factor,
            row[f"{prefix}_SizeZ"] * scale_factor,
        ],
        dtype=torch.float,
        device=device,
    )


def get_all_blocks_data(
    row: pd.Series,
    scale_factor: float,
    device: torch.device,
) -> list[BlockData]:
    """Read every block of one design row, in the order the column prefixes appear.

    That order is the chain order: block 0 becomes the first frozen parent and each later block
    is repaired against the one before it.
    """
    prefixes = get_block_prefixes(row.index)
    blocks: list[BlockData] = []
    for prefix in prefixes:
        pos = _get_pos(row, prefix, scale_factor, device)
        size = _get_size(row, prefix, scale_factor, device)
        blocks.append(BlockData(prefix=prefix, pos=pos, size=size))
    return blocks


def predict_feasibility(
    size_anchor: torch.Tensor,
    size_active: torch.Tensor,
    pos_anchor: torch.Tensor,
    pos_active: torch.Tensor,
    gnn_model: GNN,
) -> tuple[float, np.ndarray]:
    """Ask the surrogate about one pair, without optimising anything.

    Callers pass sizes; the encoding into codes happens here, because the surrogate of this
    branch reads codes as node features and nothing outside the repair needs to know that.

    Args:
        size_anchor, size_active: Full edge lengths of parent and part under test, ``(3,)``
            each in the scaled frame.
        pos_anchor, pos_active: Their centres, same convention.

    Returns:
        A pair ``(p_binary, reg_out_m)``: the feasibility probability, and the two regressed
        criteria in metres as a ``(2,)`` array, overlap then thickness.
    """
    encoder = get_box_encoder(device=str(size_anchor.device))
    with torch.no_grad():
        z_anchor = encode_size(encoder, size_anchor.unsqueeze(0)).squeeze(0)
        z_active = encode_size(encoder, size_active.unsqueeze(0)).squeeze(0)
    x, edge_index, edge_attr, _ = build_graph(
        z_anchor, z_active, pos_anchor, pos_active,
    )
    with torch.no_grad():
        reg_std, binary_logit, *_ = gnn_model(x, edge_index, edge_attr)
        reg = destandardize(reg_std, gnn_model.target_mean, gnn_model.target_std)
        p = float(torch.sigmoid(binary_logit.squeeze()).item())
        reg_m = reg.squeeze(0).cpu().numpy()
    return p, reg_m


def is_repaired(
    reg_out_m: np.ndarray, p_binary: float, tol: float = EVAL_TOLERANCE,
) -> tuple[bool, list[str]]:
    """Decide whether a pair may be committed, from the surrogate's outputs alone.

    This is the only judge the chain acts on. The closed-form check runs alongside and is
    recorded, but never changes what is committed.

    Returns:
        A pair ``(ok, violations)``, the second naming each failed criterion with its value.
    """
    violated: list[str] = []
    if reg_out_m[0] < THRESH_OVERLAP - tol:
        violated.append(f"overlap={reg_out_m[0]:.4f} < {THRESH_OVERLAP}")
    if reg_out_m[1] > THRESH_THICKNESS + tol:
        violated.append(f"thickness={reg_out_m[1]:.4f} > {THRESH_THICKNESS}")
    if p_binary < 0.5:
        violated.append(f"p_binary={p_binary:.3f} < 0.5")
    return len(violated) == 0, violated


def analytical_diagnose(
    anchor: BlockData,
    pos_active_after_scaled: torch.Tensor,
    size_active_after_scaled: torch.Tensor,
    face: int,
    scale_factor: float,
) -> dict:
    """Measure the two criteria on a repaired pair in closed form, after the fact.

    Because the parts are axis-aligned boxes these formulas are the same ones the labels were
    produced with, so they settle the pair without another simulation run. They are computed
    after the repair and after the contact correction, on the geometry that is actually
    committed.

    Args:
        pos_active_after_scaled, size_active_after_scaled: The child as it stands after the
            repair, ``(3,)`` each in the scaled frame.
        face: Contact face code from 0 to 5.
        scale_factor: Factor between the scaled frame and metres.

    Returns:
        A dictionary with ``ana_overlap`` and ``ana_thickness`` in metres, and ``he_after_m``,
        the child's half-extents in metres.
    """
    pos_p_m = anchor.pos.cpu().numpy() / scale_factor
    he_p_m = anchor.he / scale_factor
    pos_c_m = pos_active_after_scaled.cpu().numpy() / scale_factor
    he_c_m = (size_active_after_scaled / 2.0).cpu().numpy() / scale_factor
    ana_ov = analytical_overlap(pos_p_m, he_p_m, pos_c_m, he_c_m, face)
    ana_th = analytical_thickness(he_c_m, face)
    return {
        "ana_overlap": ana_ov,
        "ana_thickness": ana_th,
        "he_after_m": he_c_m,
    }


def _apply_contact_snap(
    anchor: BlockData,
    pos_active_scaled: torch.Tensor,
    size_active_scaled: torch.Tensor,
    face: int,
    scale_factor: float,
    verbose: bool = False,
) -> tuple[torch.Tensor, dict]:
    """Put the repaired child back in exact contact with its parent, returning the new centre
    and a record of how far it had to move."""
    contact_axis = 0 if face <= 1 else (1 if face <= 3 else 2)

    pos_p_m = anchor.pos.cpu().numpy() / scale_factor
    he_p_m = anchor.he / scale_factor
    pos_c_m = pos_active_scaled.cpu().numpy() / scale_factor
    he_c_m = (size_active_scaled / 2.0).cpu().numpy() / scale_factor

    pos_c_snapped_m, shift_m, _, gap_before_m = snap_to_contact_face(
        pos_p_m, he_p_m, pos_c_m, he_c_m, contact_axis=contact_axis,
    )

    pos_c_snapped_scaled = torch.from_numpy(
        (pos_c_snapped_m * scale_factor).astype(np.float32)
    ).to(pos_active_scaled.device)

    if verbose and abs(shift_m) > 1e-5:
        sign_str = "gap closed" if gap_before_m > 0 else "penetration resolved"
        print(
            f"    [snap] {sign_str}: contact_axis={contact_axis}  "
            f"shift={shift_m*1000:+.2f} mm  gap_before={gap_before_m*1000:+.2f} mm"
        )

    return pos_c_snapped_scaled, {
        "snap_shift_mm": float(shift_m * 1000),
        "snap_gap_before_mm": float(gap_before_m * 1000),
        "snap_contact_axis": contact_axis,
    }


@dataclass
class PairRepairResult:
    """Everything one pair of the chain produced, including both verdicts.

    The two flags at the end are the point of the record. ``is_bluff`` marks a pair the
    surrogate accepted and the formulas rejected, which is a repair that only exists in the
    network; ``is_overcautious`` marks the opposite, a sound geometry the network refused. Both
    are counted in the head-to-head comparison of Parts I and II.

    Geometry is in the scaled frame, predicted and measured criteria are in metres, and
    ``pair_idx`` is the index of the block under test, so pair ``k`` joins block ``k - 1`` to
    block ``k``. ``method`` is one of ``already_feasible``, ``adam`` or ``failed``.
    """

    pair_idx: int
    prefix_fixed: str
    prefix_active: str
    face: int
    needed_repair: bool
    method: str
    repaired: bool = False
    violated: list[str] = field(default_factory=list)

    size_before: torch.Tensor | None = None
    size_after: torch.Tensor | None = None
    pos_before: np.ndarray | None = None
    pos_after: np.ndarray | None = None

    reg_before: np.ndarray | None = None
    reg_after: np.ndarray | None = None
    p_before: float = 0.0
    p_after: float = 0.0

    ana_overlap_after: float = float("nan")
    ana_thickness_after: float = float("nan")

    gnn_ok: bool = False
    analytical_ok: bool = False
    is_bluff: bool = False
    is_overcautious: bool = False


@dataclass
class ChainResult:
    """The whole chain for one assembly: every step, the committed blocks, and where it stopped.

    ``failed_pair_idx`` is the index of the first pair that could not be repaired, or -1 when
    the chain reached the end. ``committed_blocks`` holds the repaired assembly and is shorter
    than the input whenever the chain stopped early.
    """

    assembly_idx: int
    n_blocks: int
    steps: list[PairRepairResult]
    committed_blocks: list[BlockData]
    success: bool
    failed_pair_idx: int = -1


def repair_chain_single(
    blocks: list[BlockData],
    gnn_model: GNN,
    scale_factor: float,
    device: torch.device,
    num_steps: int | None = None,
    lr: float | None = None,
    freeze_pos_if_overlap_ok: bool = True,
    verbose: bool = False,
) -> ChainResult:
    """Repair one assembly by walking the chain of pairs and committing as it goes.

    Block 0 is never modified. Each later block is repaired against the committed version of
    its predecessor, so a change made early propagates forward. The chain stops at the first
    pair the surrogate will not accept; the failed step is recorded with its final state and no
    later pair is attempted, because its parent would not exist.

    A pair the surrogate already accepts is committed without running the optimiser, which is
    why the reported repair rate has to be read against how many pairs needed repairing at all.

    Args:
        blocks: The assembly in chain order, from :func:`get_all_blocks_data`.
        gnn_model: The surrogate, as loaded by :func:`load_models`.
        scale_factor: Factor between the scaled frame and metres.
        device: Device the tensors live on.
        num_steps: Step budget per pair, or ``None`` for the default of the optimiser.
        lr: Learning rate per pair, or ``None`` for the default of the optimiser.
        freeze_pos_if_overlap_ok: Passed through to the optimiser; keeps the position fixed for
            a pair whose overlap already passes.
        verbose: Whether the optimiser prints its progress.

    Returns:
        The chain result, with ``assembly_idx`` left at -1 for the caller to fill in.
    """
    n_blocks = len(blocks)
    committed: list[BlockData] = [blocks[0]]
    steps: list[PairRepairResult] = []

    adam_kwargs: dict = {
        "verbose": verbose,
        "freeze_pos_if_overlap_ok": freeze_pos_if_overlap_ok,
    }
    if num_steps is not None:
        adam_kwargs["num_steps"] = num_steps
    if lr is not None:
        adam_kwargs["lr"] = lr

    for k in range(1, n_blocks):
        # The parent is the committed block, not the designed one: a block repaired in an
        # earlier step has moved, and the next pair has to be judged against where it now is.
        prev = committed[k - 1]
        curr = blocks[k]

        pos_p_m = prev.pos.cpu().numpy() / scale_factor
        pos_c_m = curr.pos.cpu().numpy() / scale_factor
        he_p_m = prev.he / scale_factor
        he_c_m = curr.he / scale_factor
        face = infer_face(pos_p_m, he_p_m, pos_c_m, he_c_m)

        step = PairRepairResult(
            pair_idx=k,
            prefix_fixed=prev.prefix,
            prefix_active=curr.prefix,
            face=face,
            needed_repair=False,
            method="already_feasible",
            size_before=curr.size.clone(),
            pos_before=curr.pos.cpu().numpy(),
        )

        p_before, reg_before = predict_feasibility(
            prev.size, curr.size, prev.pos, curr.pos, gnn_model,
        )
        step.reg_before = reg_before
        step.p_before = p_before

        if is_repaired(reg_before, p_before)[0]:
            step.repaired = True
            step.gnn_ok = True
            step.size_after = curr.size.clone()
            step.pos_after = curr.pos.cpu().numpy()
            step.reg_after = reg_before
            step.p_after = p_before
            diag = analytical_diagnose(
                prev, curr.pos, curr.size, face, scale_factor,
            )
            _attach_diag(step, diag)
            committed.append(curr)
            steps.append(step)
            continue

        step.needed_repair = True

        out = adam_repair_batched(
            [prev.size], [curr.size], [prev.pos], [curr.pos],
            gnn_model, scale_factor, **adam_kwargs,
        )
        size_after = out["size_active_out"][0]
        pos_after = out["pos_active_out"][0]
        reg_after = out["reg_out_m"][0]
        p_after = float(out["p_binary"][0])
        success_adam = bool(out["success_mask"][0])

        if success_adam:
            pos_snapped, _snap_info = _apply_contact_snap(
                prev, pos_after, size_after, face, scale_factor, verbose=verbose,
            )

            step.method = "adam"
            step.repaired = True
            step.gnn_ok = True
            step.size_after = size_after.clone()
            step.pos_after = pos_snapped.cpu().numpy()
            step.reg_after = reg_after
            step.p_after = p_after

            diag = analytical_diagnose(
                prev, pos_snapped, size_after, face, scale_factor,
            )
            _attach_diag(step, diag)

            committed.append(
                BlockData(
                    prefix=curr.prefix,
                    pos=pos_snapped.detach(),
                    size=size_after.detach(),
                )
            )
            steps.append(step)
            continue

        # The failed pair is recorded in full, including the closed-form metrics of the state
        # the optimiser ended in. Whether the geometry was in fact sound is what distinguishes
        # a genuine failure from an over-cautious refusal.
        step.method = "failed"
        step.repaired = False
        step.gnn_ok = False
        _, viol = is_repaired(reg_after, p_after)
        step.violated = viol
        step.size_after = size_after.clone()
        step.pos_after = pos_after.cpu().numpy()
        step.reg_after = reg_after
        step.p_after = p_after
        diag_failed = analytical_diagnose(
            prev, pos_after, size_after, face, scale_factor,
        )
        _attach_diag(step, diag_failed)
        steps.append(step)
        break

    success = all(s.repaired for s in steps)
    failed_pair = (
        -1 if success else next((s.pair_idx for s in steps if not s.repaired), -1)
    )
    return ChainResult(
        assembly_idx=-1, n_blocks=n_blocks, steps=steps,
        committed_blocks=committed, success=success, failed_pair_idx=failed_pair,
    )


def _attach_diag(step: PairRepairResult, diag: dict | None) -> None:
    """Record the closed-form verdict on a step and set the bluff and over-caution flags."""
    if diag is None:
        return
    step.ana_overlap_after = float(diag.get("ana_overlap", float("nan")))
    step.ana_thickness_after = float(diag.get("ana_thickness", float("nan")))
    if not np.isnan(step.ana_overlap_after):
        ana_ok, _ = is_repaired_analytical(
            step.ana_overlap_after, step.ana_thickness_after
        )
        step.analytical_ok = ana_ok
        step.is_bluff = step.gnn_ok and not ana_ok
        step.is_overcautious = (not step.gnn_ok) and ana_ok


def _mean_across_step_columns(results_df: pd.DataFrame, suffix: str) -> float:
    """Average a per-pair column over the pairs of each assembly and then over assemblies."""
    cols = [c for c in results_df.columns if c.startswith("pair") and c.endswith(suffix)]
    if not cols:
        raise KeyError(f"No pair columns found for suffix '{suffix}'")
    return float(results_df[cols].mean(axis=1).mean())


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    gnn_model, scale_factor = load_models(device)

    df: pd.DataFrame = read_txt_file(DATA_FOLDER, config.data.data_file)
    df = manipulate_data(df)
    print(f"Loaded {len(df)} assemblies")

    all_prefixes = get_block_prefixes(df.columns)
    n_block_types = len(all_prefixes)
    print(f"Detected {n_block_types} block type(s): {all_prefixes}")

    # The split is reproduced from the same seed and the same proportions the surrogate was
    # trained with, so that the assemblies evaluated here are ones it never saw.
    torch.manual_seed(config.general.random_seed)
    n = len(df)
    perm = torch.randperm(n).tolist()
    val_end = int((config.training.train_split + config.training.val_split) * n)
    test_indices = perm[val_end:]

    test_df = df.iloc[test_indices]
    infeasible = test_df.loc[~test_df["Assembly_Good?"].astype(bool)]
    print(f"Found {len(infeasible)} infeasible assemblies in test set")

    for prefix in all_prefixes:
        col = f"{prefix}_FailureReason"
        if col in infeasible.columns:
            counts = infeasible[col].value_counts()
            print(f"  {prefix} failure reasons: {counts.to_dict()}")

    n_eval_request = int(os.environ.get("N_EVAL", "100"))
    n_eval = min(n_eval_request, len(infeasible))
    sample_df = infeasible.sample(n=n_eval, random_state=config.general.random_seed)

    print(f"\nEvaluating chain repair on {n_eval} assemblies (pure-Adam, no stages)")

    results: list[dict] = []
    chain_results: list[ChainResult] = []
    start_time = time.time()

    for i, (assembly_idx, row) in enumerate(sample_df.iterrows()):
        blocks = get_all_blocks_data(row, scale_factor, device)
        n_blk = len(blocks)
        if n_blk < 2:
            continue

        # A two-block assembly the surrogate already accepts has nothing to repair and is left
        # out of the statistics entirely, rather than being counted as a success it did not
        # earn. A longer assembly is still run, because a later pair may need work.
        p_check, reg_check = predict_feasibility(
            blocks[0].size, blocks[1].size, blocks[0].pos, blocks[1].pos, gnn_model,
        )
        if is_repaired(reg_check, p_check)[0] and n_blk == 2:
            continue

        chain = repair_chain_single(blocks, gnn_model, scale_factor, device)
        chain.assembly_idx = int(row.name)
        chain_results.append(chain)

        row_out: dict = {
            "assembly_idx": chain.assembly_idx,
            "n_blocks": chain.n_blocks,
            "failure_reason": row.get("Block1_FailureReason", "UNKNOWN"),
            "chain_success": chain.success,
            "failed_pair_idx": chain.failed_pair_idx,
            "n_steps": len(chain.steps),
        }
        for step in chain.steps:
            k = step.pair_idx
            row_out[f"pair{k}_method"] = step.method
            row_out[f"pair{k}_repaired"] = step.repaired
            row_out[f"pair{k}_face"] = step.face
            row_out[f"pair{k}_gnn_ok"] = step.gnn_ok
            row_out[f"pair{k}_analytical_ok"] = step.analytical_ok
            row_out[f"pair{k}_is_bluff"] = step.is_bluff
            row_out[f"pair{k}_is_overcautious"] = step.is_overcautious
            row_out[f"pair{k}_ana_overlap_after"] = step.ana_overlap_after
            row_out[f"pair{k}_ana_thickness_after"] = step.ana_thickness_after
            if step.reg_before is not None:
                for j, rn in enumerate(REG_NAMES):
                    row_out[f"pair{k}_before_{rn}"] = float(step.reg_before[j])
            if step.reg_after is not None:
                for j, rn in enumerate(REG_NAMES):
                    row_out[f"pair{k}_after_{rn}"] = float(step.reg_after[j])
            row_out[f"pair{k}_p_binary_after"] = step.p_after
            if step.violated:
                row_out[f"pair{k}_violated"] = "; ".join(step.violated)
        results.append(row_out)

        methods = [s.method for s in chain.steps]
        bluffs = sum(1 for s in chain.steps if s.is_bluff)
        bluff_tag = f" (bluffs={bluffs})" if bluffs else ""
        status = "OK" if chain.success else f"FAIL(p{chain.failed_pair_idx})"
        print(
            f"  [{i+1}/{n_eval}] idx={chain.assembly_idx} "
            f"{status} methods={methods}{bluff_tag}"
        )

    total_time = time.time() - start_time
    results_df = pd.DataFrame(results)
    if len(results_df) == 0:
        print("Nothing to repair.")
        return

    n_total = len(results_df)
    n_success = int(results_df["chain_success"].sum())
    print(f"\n{'='*70}")
    print(f"Chain Repair Results — {n_total} assemblies, {total_time:.1f}s total")
    print(f"{'='*70}")
    print(
        f"Overall chain success rate: "
        f"{n_success}/{n_total} ({100*n_success/n_total:.1f}%)"
    )

    n_steps_total = 0
    n_bluff = 0
    n_overcautious = 0
    for chain in chain_results:
        for s in chain.steps:
            n_steps_total += 1
            n_bluff += int(s.is_bluff)
            n_overcautious += int(s.is_overcautious)
    if n_steps_total > 0:
        print(f"\nDiagnostic side-channel ({n_steps_total} steps):")
        print(
            f"  Bluffs (GNN ok, analytical fail):       "
            f"{n_bluff} ({100*n_bluff/n_steps_total:.1f}%)"
        )
        print(
            f"  Overcautious (GNN fail, analytical ok): "
            f"{n_overcautious} ({100*n_overcautious/n_steps_total:.1f}%)"
        )

    print("\nRegression (before -> after, mean over committed pairs):")
    for name in REG_NAMES:
        try:
            before = _mean_across_step_columns(results_df, f"before_{name}")
            after = _mean_across_step_columns(results_df, f"after_{name}")
            print(f"  {name}: {before:.4f} -> {after:.4f}")
        except KeyError:
            pass

    max_steps = int(results_df["n_steps"].max()) if "n_steps" in results_df else 1
    if max_steps > 0:
        print("\nPer-pair repair methods:")
        for k in range(1, max_steps + 1):
            col = f"pair{k}_method"
            if col in results_df.columns:
                counts = results_df[col].value_counts().to_dict()
                print(f"  Pair {k}: {counts}")

    failures = results_df[~results_df["chain_success"]]
    if len(failures) > 0:
        print(f"\nFailed assemblies ({len(failures)}):")
        show_cols = ["assembly_idx", "failure_reason", "failed_pair_idx"]
        for k in range(1, max_steps + 1):
            vcol = f"pair{k}_violated"
            if vcol in failures.columns:
                show_cols.append(vcol)
        avail = [c for c in show_cols if c in failures.columns]
        print(failures[avail].to_string(index=False))

    output_dir: Path = (
        Path(__file__).parent.parent / "results_paramter_optimization"
    )
    output_dir.mkdir(exist_ok=True)
    out_path = output_dir / "repair_results_chain.csv"
    results_df.to_csv(out_path, index=False)
    print(f"\nResults saved to {out_path}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
