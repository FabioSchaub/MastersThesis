"""Repair an assembly of several blocks one joint at a time, and score the result.

The surrogate only ever sees two blocks. An assembly of more is handled by walking along it:
the first block is frozen, then each following block is repaired against the one before it,
which by then has already been fixed and is itself frozen. That keeps every query to the
surrogate a two-block query and needs no second model.

Per joint the sequence is: infer the contact face from the geometry, ask the surrogate whether
the joint is already feasible, and if it is not, run the gradient repair on the child alone.
A joint that is repaired is put back into contact before it is committed, so the block the
next joint is measured against is the repaired one.

Alongside the flow, and never affecting it, the closed-form quantities are recomputed on the
repaired geometry. The surrogate decides what happens; the closed form records what is true.
A joint the surrogate passes and the closed form rejects is recorded as a bluff, the reverse
as overcaution, and those two counts are what make a repair result assessable rather than
merely reported.

Running this file evaluates the repair over a sample of infeasible assemblies from the test
split and writes ``results_paramter_optimization/repair_results_chain.csv``:

    python src/repair_process.py

The environment variable ``N_EVAL`` sets how many assemblies are drawn, and ``GNN_MODEL``
selects the checkpoint.
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
    is_repaired_batch,
)
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

# The checkpoint the results of Part I were produced with. It is named rather than discovered
# so that a rerun uses the same weights; ``GNN_MODEL`` overrides it.
PLACEHOLDER_GNN_NAME: str = (
    "gnn_small-mixed-10mm_node5_batchsize2048_20260519-092003.pth"
)


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


MODEL_NAME: str = os.environ.get("GNN_MODEL", PLACEHOLDER_GNN_NAME)

REG_NAMES: list[str] = ["overlap", "thickness"]


def load_models(device: torch.device) -> tuple[GNN, float]:
    """Restore the surrogate and the scale its inputs are expected in.

    The width of the standardisation buffers is read off the checkpoint before the buffers are
    registered, and weights whose shape does not match the model are reported and skipped
    rather than silently dropped, so a checkpoint from a differently sized run does not load
    as a partially initialised model without anyone noticing.

    Returns:
        The surrogate in evaluation mode on ``device``, and the factor by which lengths are
        multiplied before they reach it, which is one.

    Raises:
        AssertionError: If the standardisation did not come through. Without it every
            de-standardised output would be meaningless, and the repair would optimise noise.
    """
    gnn_name = MODEL_NAME or _find_latest_gnn()
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
    tm_ckpt = ckpt_state.get("target_mean")
    n_targets = int(tm_ckpt.shape[0]) if tm_ckpt is not None else 2
    gnn_model.register_buffer("target_mean", torch.zeros(n_targets))
    gnn_model.register_buffer("target_std", torch.ones(n_targets))

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
    # Must match what the checkpoint was trained on, see prepare_simulation_dataset.
    scale_factor: float = 1.0
    print(f"Dataset: {config.data.data_file}  scale_factor=1.0 (raw metres)")

    return gnn_model, scale_factor


@dataclass
class BlockData:
    """One block: its centre and its three edge lengths, in the scaled domain.

    Full edge lengths, not half-extents, because that is what the surrogate consumes; the
    closed-form functions want halves, which is what :attr:`he` is for.

    Attributes:
        prefix: Column prefix the block came from, kept so results can be written back.
        pos: Centre, shape ``(3,)``.
        size: Edge lengths, shape ``(3,)``.
    """
    prefix: str
    pos: torch.Tensor
    size: torch.Tensor

    @property
    def size_z(self) -> float:
        return float(self.size[2])

    @property
    def he(self) -> np.ndarray:
        """Half-extents, shape ``(3,)``, in the scaled domain."""
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
    """Every block of one row, in the order the column prefixes are numbered."""
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
    """Ask the surrogate about one joint.

    Returns:
        The feasibility probability, and the overlap and thickness in metres as a ``(2,)``
        array.
    """
    x, edge_index, edge_attr, _ = build_graph(
        size_anchor, size_active, pos_anchor, pos_active,
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
    """Commit gate for one joint, with a readable string per failed condition.

    Same three conditions as :func:`src.repair_optimizer.is_repaired_single`; it is restated
    here so that the chain does not depend on the optimiser module for its flow control.
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
    """Recompute the two quantities in closed form on the repaired joint.

    For axis-aligned boxes these are the formulas the labels come from, so the joint can be
    judged without going back to the simulator. Nothing here influences the flow; the values
    are recorded so that a repair can be checked against something other than the model that
    produced it.

    Returns:
        The overlap and thickness in metres, and the child's half-extents in metres.
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
    """Put the repaired child back onto its parent's face, and report how far it had to move.

    Neither criterion changes, since only the contact-axis coordinate is written.
    """
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
    """What happened at one joint, including both verdicts on the outcome.

    The geometry is recorded before and after, so a repair can be reconstructed and drawn
    without rerunning it, and both the surrogate's answer and the closed-form answer are kept
    side by side. The two flags at the end are the comparison the evaluation rests on.

    Attributes:
        pair_idx: Index of the child, so the joint between block ``pair_idx - 1`` and
            ``pair_idx``. Counting starts at one; the anchor is never a child.
        face: Contact face, inferred from the geometry.
        method: How the joint was resolved: already feasible, repaired, or failed.
        size_before, size_after: Edge lengths in the scaled domain.
        pos_before, pos_after: Centres in the scaled domain, the latter after the snap.
        reg_before, reg_after: The surrogate's overlap and thickness in metres.
        ana_overlap_after, ana_thickness_after: The same two quantities in closed form.
        is_bluff: The surrogate passed the joint and the closed form rejects it.
        is_overcautious: The surrogate rejected a joint the closed form accepts.
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
    """The repair of one assembly, joint by joint.

    Attributes:
        committed_blocks: The blocks as they stand after the walk, the anchor first. Shorter
            than the assembly when a joint failed, because the walk stops there.
        success: Every joint of the assembly was resolved.
        failed_pair_idx: Index of the joint that stopped the walk, or minus one.
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
    """Walk along one assembly and repair each joint against the block before it.

    A joint the surrogate already accepts is committed untouched, which matters for the
    accounting: a design counted as repaired should not include joints that were never broken.
    A joint that fails after the gradient repair stops the walk, because every later joint
    would be measured against geometry that is not going to be built.

    Args:
        blocks: The assembly, first block first. It becomes the anchor.
        gnn_model: The surrogate.
        scale_factor: Factor between the scaled domain and metres.
        device: Device the repair runs on.
        num_steps: Overrides the number of Adam steps.
        lr: Overrides the Adam step size.
        freeze_pos_if_overlap_ok: Passed through to the optimiser; a joint whose overlap
            already holds is refined in size only.
        verbose: Whether the optimiser prints progress.

    Returns:
        One :class:`PairRepairResult` per joint reached, and the blocks as committed.
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
        prev = committed[k - 1]
        curr = blocks[k]

        # The parent is taken from the committed list, not from the input: by this point it
        # may itself have been repaired, and the child has to be fitted to what will be built.
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

        # The failed joint is recorded in full before the walk stops, so the assembly can be
        # inspected at the point where the repair gave up.
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
    """Record the closed-form verdict on a step and set the bluff and overcaution flags."""
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
    """Average one per-joint column over all joints and all assemblies."""
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

    # Reproduces the split of ``split_graphs`` from the same seed and the same ratios, so the
    # repair is evaluated on data the surrogate was not fitted on.
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

        p_check, reg_check = predict_feasibility(
            blocks[0].size, blocks[1].size, blocks[0].pos, blocks[1].pos, gnn_model,
        )
        # A two-block assembly the surrogate already accepts has nothing to repair and is left
        # out of the count entirely, rather than entering it as a free success.
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
