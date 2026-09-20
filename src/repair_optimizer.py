"""The repair itself: descend the surrogate's gradient until a pair of blocks is buildable.

The optimiser holds the latent code of the part under test and its position, and moves them so
that the surrogate reports at least ``config.gnn.thresh_overlap_min`` of horizontal contact and
at most ``config.gnn.thresh_thickness_max`` of joint thickness. The parent is frozen. This is
the one substitution that separates Part II from Part I, where the same loop holds the three
edge lengths instead of a code.

Size does not disappear, it only stops being the variable. Whenever the loop needs a geometry
rather than a code, the frozen box decoder of ``src/dec_box.py`` supplies it, never the
auxiliary signed-distance decoder of stage 1: the drift penalty compares decoded sizes, the
hard caps are stated in metres and enforced by decoding, clamping and re-encoding, and the
sizes returned to the caller are decoded. Formulating the drift and the caps in metres rather
than in the code is deliberate, since a penalty on the code would mean something different
from the parameter branch and the two runs would no longer be comparable.

The caller passes and receives sizes throughout, so nothing upstream needs to know that the
optimisation happens in a latent space.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch, Data

sys.path.insert(0, str(Path(__file__).parent.parent))
from config.config import config  # noqa: E402
from src.dec_box import BoxDecoder, get_box_decoder  # noqa: E402
from src.enc_box import BoxEncoder  # noqa: E402
from src.gnn import GNN  # noqa: E402
from src.gnn_dataset_preparation import get_box_encoder  # noqa: E402

# The two screwdriving criteria, in metres, read from the configuration the surrogate was
# trained under.
THRESH_OVERLAP: float = float(config.gnn.thresh_overlap_min)
THRESH_THICKNESS: float = float(config.gnn.thresh_thickness_max)
LATENT_DIM: int = int(config.autoencoder.latent_dim)

# Both slacks are zero on purpose. Anything positive would let the loop stop, or a repair be
# accepted, on a pair that only just misses a criterion, and the reported repair rate would
# then depend on the size of the slack rather than on the repair.
EARLY_STOP_MARGIN: float = 0.0
EVAL_TOLERANCE: float = 0.0
# The offset added before the logarithm when overlap was standardised for training. It has to
# match src/gnn_dataset_preparation.py exactly, otherwise the inverse below is biased.
OVERLAP_LOG_EPS: float = 1e-5


# The encoder and the decoder are both defined on half-extents, while the pipeline speaks in
# full edge lengths. These four helpers keep the factor of two in one place.
def _size_to_half(size: torch.Tensor) -> torch.Tensor:
    return size / 2.0


def _half_to_size(half: torch.Tensor) -> torch.Tensor:
    return half * 2.0


def encode_size(
    encoder: BoxEncoder, size: torch.Tensor
) -> torch.Tensor:
    """Encode full edge lengths ``(..., 3)`` into codes ``(..., latent_dim)``."""
    return encoder(_size_to_half(size))


def decode_z(
    decoder: BoxDecoder, z: torch.Tensor
) -> torch.Tensor:
    """Decode codes ``(..., latent_dim)`` back into full edge lengths ``(..., 3)``.

    Uses the box decoder, the one that returns half-extents, not the auxiliary
    signed-distance decoder of stage 1.
    """
    return _half_to_size(decoder(z))


def destandardize(
    reg_pred_std: torch.Tensor,
    target_mean: torch.Tensor,
    target_std: torch.Tensor,
) -> torch.Tensor:
    """Turn the surrogate's standardised regression outputs back into metres.

    The loop compares against thresholds stated in metres, so this inverse sits between every
    forward pass and every decision.

    Args:
        reg_pred_std: Standardised predictions, shape ``(..., 2)``, overlap then thickness.
        target_mean: Mean the targets were standardised with, shape ``(2,)``.
        target_std: Standard deviation the targets were standardised with, shape ``(2,)``.

    Returns:
        The same shape in metres.
    """
    out = reg_pred_std * target_std + target_mean
    # Overlap was standardised after a logarithm and thickness was not, and nothing in the
    # checkpoint records which. A mean below zero can only come from the log branch, since an
    # overlap in metres is positive, so its sign identifies the convention.
    if target_mean[0].item() < 0.0:
        out = out.clone()
        out[..., 0] = torch.exp(out[..., 0]) - OVERLAP_LOG_EPS
    return out


def is_repaired_batch(
    reg_out_m: np.ndarray,
    p_binary: np.ndarray,
    tol: float = EVAL_TOLERANCE,
) -> np.ndarray:
    """Ask the surrogate whether each repaired pair passes, over a whole batch.

    Both regressed criteria and the feasibility logit must agree. This is the network's own
    verdict and is not evidence that the geometry is sound; the independent check is
    :func:`src.analytical_metrics.is_repaired_analytical`, and a pair that passes here and
    fails there is a bluff.

    Args:
        reg_out_m: Predictions in metres, shape ``(N, 2)``, overlap then thickness.
        p_binary: Feasibility probabilities after the sigmoid, shape ``(N,)``.
        tol: Slack on the two threshold comparisons.

    Returns:
        A boolean array of shape ``(N,)``.
    """
    ov_ok = reg_out_m[:, 0] >= THRESH_OVERLAP - tol
    th_ok = reg_out_m[:, 1] <= THRESH_THICKNESS + tol
    bin_ok = p_binary >= 0.5
    return ov_ok & th_ok & bin_ok


def is_repaired_single(
    reg_out_m: np.ndarray,
    p_binary: float,
    tol: float = EVAL_TOLERANCE,
) -> tuple[bool, list[str]]:
    """Same verdict as :func:`is_repaired_batch` for one pair, with the reasons.

    Returns:
        A pair ``(ok, violations)``, where ``violations`` names each failed criterion with its
        value and is empty when ``ok`` is true.
    """
    violated: list[str] = []
    if reg_out_m[0] < THRESH_OVERLAP - tol:
        violated.append(f"overlap={reg_out_m[0]:.4f} < {THRESH_OVERLAP}")
    if reg_out_m[1] > THRESH_THICKNESS + tol:
        violated.append(f"thickness={reg_out_m[1]:.4f} > {THRESH_THICKNESS}")
    if p_binary < 0.5:
        violated.append(f"p_binary={p_binary:.3f} < 0.5")
    return len(violated) == 0, violated


# The graphs below are assembled by hand rather than through the dataset module, because the
# repair has to rebuild them from the current code and position at every step. Their layout
# must therefore stay identical to the one src/gnn_dataset_preparation.py produces: node 0 is
# the frozen parent, node 1 the part under test, and the two-entry one-hot at the end of a node
# feature marks which is which.
def _node_features(
    z_anchor: torch.Tensor,  # (latent_dim,)
    z_active: torch.Tensor,  # (latent_dim,)
) -> torch.Tensor:
    """Build the ``(2, latent_dim + 2)`` node features of one pair, parent first."""
    device = z_anchor.device
    type_0 = torch.tensor([1.0, 0.0], device=device, dtype=torch.float)
    type_1 = torch.tensor([0.0, 1.0], device=device, dtype=torch.float)
    return torch.stack(
        [
            torch.cat([z_anchor, type_0]),
            torch.cat([z_active, type_1]),
        ]
    )


def _edge_features(
    pos_anchor_scaled: torch.Tensor,  # (3,)
    pos_active_scaled: torch.Tensor,  # (3,)
) -> torch.Tensor:
    """Build the ``(2, 6)`` edge features of one pair, forward edge first.

    The offset changes sign between the two directions while the three axis-pair distances do
    not, which is what makes the edge feature direction-aware without being asymmetric in
    magnitude.
    """
    delta = pos_active_scaled - pos_anchor_scaled
    dx, dy, dz = delta[0], delta[1], delta[2]
    # Added inside the square root, not outside: two coincident centres would otherwise give a
    # zero-length vector whose gradient is undefined, and the repair differentiates through
    # this expression at every step.
    eps = 1e-8
    dist_xy = torch.sqrt(dx**2 + dy**2 + eps)
    dist_xz = torch.sqrt(dx**2 + dz**2 + eps)
    dist_yz = torch.sqrt(dy**2 + dz**2 + eps)
    fwd = torch.stack([dx, dy, dz, dist_xy, dist_xz, dist_yz])
    rev = torch.stack([-dx, -dy, -dz, dist_xy, dist_xz, dist_yz])
    return torch.stack([fwd, rev])


def build_graph(
    z_anchor: torch.Tensor,
    z_active: torch.Tensor,
    pos_anchor_scaled: torch.Tensor,
    pos_active_scaled: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Assemble the four tensors the surrogate takes for a single pair.

    Args:
        z_anchor, z_active: Codes of parent and part under test, each ``(latent_dim,)``.
        pos_anchor_scaled, pos_active_scaled: Centres in the scaled frame, each ``(3,)``.

    Returns:
        A tuple ``(x, edge_index, edge_attr, batch)`` with node features ``(2, node_dim)``, the
        two directed edges ``(2, 2)``, edge features ``(2, edge_dim)`` and a batch vector that
        assigns both nodes to graph zero.
    """
    device = z_anchor.device
    x = _node_features(z_anchor, z_active)
    edge_attr = _edge_features(pos_anchor_scaled, pos_active_scaled)
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long, device=device)
    batch = torch.zeros(2, dtype=torch.long, device=device)
    return x, edge_index, edge_attr, batch


def _build_batch(
    z_anchor_all: torch.Tensor,  # (N, latent_dim)
    z_active_all: torch.Tensor,  # (N, latent_dim)
    pos_anchor_all: torch.Tensor,  # (N, 3) scaled
    pos_active_all: torch.Tensor,  # (N, 3) scaled
) -> Batch:
    """Pack ``N`` two-node graphs into one batch, so a step costs one forward pass.

    Called once per optimisation step and rebuilt from scratch each time, because the codes and
    positions it reads are the parameters being optimised and the batch has to stay part of the
    graph the gradient flows through.
    """
    device = z_anchor_all.device
    N = z_anchor_all.shape[0]
    type_0 = torch.tensor([1.0, 0.0], device=device, dtype=torch.float)
    type_1 = torch.tensor([0.0, 1.0], device=device, dtype=torch.float)
    edge_index_template = torch.tensor(
        [[0, 1], [1, 0]], dtype=torch.long, device=device
    )

    graphs = []
    for i in range(N):
        x_i = torch.stack(
            [
                torch.cat([z_anchor_all[i], type_0]),
                torch.cat([z_active_all[i], type_1]),
            ]
        )
        ea_i = _edge_features(pos_anchor_all[i], pos_active_all[i])
        graphs.append(Data(x=x_i, edge_index=edge_index_template, edge_attr=ea_i))
    return Batch.from_data_list(graphs)


def _apply_analytical_snap(
    pos_anchor_all: torch.Tensor,  # (N, 3) scaled
    size_anchor_all: torch.Tensor,  # (N, 3) scaled
    pos_active: nn.Parameter,  # (N, 3) scaled — modified in-place
    size_active_decoded: torch.Tensor,  # (N, 3) scaled — read-only here
    repaired_mask: torch.Tensor,  # (N,) bool — these are skipped
    contact_axes: torch.Tensor,  # (N,) long {0,1,2}
    scale_factor: float,
    floor_correction: bool = True,
) -> int:
    """Put every pair back in contact and above the table, modifying ``pos_active`` in place.

    Shrinking a block moves its faces, so a gradient step leaves the child floating above its
    parent or sunk into it. The surrogate has no notion of that: it reads positions and codes
    and will happily report a feasible pair that no longer touches. Restoring contact
    periodically keeps the configuration the network is asked about a physically meaningful
    one. Only positions move, so neither criterion is altered by the correction itself.

    Pairs already marked repaired are skipped, so a committed solution is not disturbed.

    Args:
        repaired_mask: ``(N,)``, true where the pair is already accepted and left alone.
        contact_axes: ``(N,)`` axis indices, passed in rather than re-inferred so the axis
            cannot change mid-run as the geometry moves.
        scale_factor: Factor between the scaled frame the tensors are in and metres.
        floor_correction: Whether to also lift a block whose underside went below the table.

    Returns:
        How many pairs moved by more than a millimetre.
    """
    # Imported here rather than at the top to keep the import graph of this module free of a
    # cycle through the analytical checks.
    from src.analytical_metrics import snap_to_contact_face

    sf = float(scale_factor)
    pos_p_m = (pos_anchor_all / sf).detach().cpu().numpy()
    he_p_m = (size_anchor_all / (2.0 * sf)).detach().cpu().numpy()
    pos_c_m = (pos_active.data / sf).detach().cpu().numpy()
    he_c_m = (size_active_decoded / (2.0 * sf)).detach().cpu().numpy()
    mask_np = repaired_mask.detach().cpu().numpy()
    axes_np = contact_axes.detach().cpu().numpy()

    n_adjusted = 0
    for i in range(pos_p_m.shape[0]):
        if mask_np[i]:
            continue
        snapped, shift_m, _, _ = snap_to_contact_face(
            pos_p_m[i],
            he_p_m[i],
            pos_c_m[i],
            he_c_m[i],
            contact_axis=int(axes_np[i]),
        )
        pos_c_m[i] = snapped
        # Count only shifts above a millimetre. Every snap moves something, and reporting
        # sub-millimetre corrections as adjustments would hide the ones that matter.
        if abs(shift_m) > 1e-3:
            n_adjusted += 1

    if floor_correction:
        bottom = pos_c_m[:, 2] - he_c_m[:, 2]
        below = bottom < 0.0
        if below.any():
            pos_c_m[below, 2] = he_c_m[below, 2]

    pos_active.data = torch.from_numpy(pos_c_m * sf).to(
        device=pos_active.device,
        dtype=pos_active.dtype,
    )
    return n_adjusted


def _project_z_via_size_clamp(
    z_active: nn.Parameter,
    encoder: BoxEncoder,
    decoder: BoxDecoder,
    size_lo: torch.Tensor | None,   # (N, 3) scaled, lower bound
    size_hi: torch.Tensor | None,   # (N, 3) scaled, upper bound
    size_min_abs: float,            # scalar abs floor
    size_max_abs: float,            # scalar abs ceiling
) -> None:
    """Project the codes back into the region of legal sizes, in place.

    This is the step that keeps the optimisation honest. Bounds on a block are statements about
    metres, not about a latent code, so they are enforced where they are meaningful: decode the
    code, clamp the size, encode the clamped size again. Running this after every gradient step
    makes the loop a projected gradient descent, and it is the only thing preventing the
    optimiser from wandering into a region of the latent space that no box occupies.

    Args:
        size_lo, size_hi: Per-pair bounds in the scaled frame, ``(N, 3)`` each, or ``None`` to
            apply only the absolute limits.
        size_min_abs, size_max_abs: Absolute limits every block must respect, in the scaled
            frame.
    """
    with torch.no_grad():
        size_dec = decode_z(decoder, z_active)
        size_clamped = size_dec.clamp(min=size_min_abs, max=size_max_abs)
        if size_lo is not None and size_hi is not None:
            size_clamped = torch.minimum(
                torch.maximum(size_clamped, size_lo), size_hi
            )
        # Re-encode only the pairs the clamp actually moved. The encoder and the decoder are
        # not exact inverses, so a roundtrip applied to a pair already inside the bounds would
        # displace its code a little at every step for no reason.
        moved = (size_clamped - size_dec).abs().sum(dim=1) > 1e-9
        if moved.any():
            z_new = encode_size(encoder, size_clamped[moved])
            z_active.data[moved] = z_new


def adam_repair_batched(
    size_anchor_list: list[torch.Tensor],
    size_active_list: list[torch.Tensor],
    pos_anchor_list: list[torch.Tensor],
    pos_active_list: list[torch.Tensor],
    gnn_model: GNN,
    scale_factor: float,
    num_steps: int = config.design_repair.num_steps,
    lr: float = config.design_repair.learning_rate,
    hinge_scale: float = 1e6,
    hinge_weights: tuple[float, float] = (1.0, 1.0),
    lambda_size: float = config.design_repair.lambda_shape,
    lambda_pos: float = config.design_repair.lambda_position,
    size_min_m: float = 0.008,
    size_max_m: float = 0.250,
    early_stop_interval: int = 20,
    pos_drift_box_m: float | None = 0.020,
    size_alpha: float | None = 0.3,
    asymmetric_tangential: bool = True,
    contact_shrink_alpha: float = 0.9,
    contact_axes_list: (
        list[int] | None
    ) = None,
    analytical_snap_interval: int = 50,
    analytical_floor_correction: bool = True,
    reset_adam_state_on_snap: bool = True,
    freeze_pos_if_overlap_ok: bool = False,
    early_stop_p_binary: float = 0.9,
    verbose: bool = True,
    trace: list | None = None,
) -> dict:
    """Repair a batch of pairs by descending the surrogate's gradient in latent space.

    All pairs are optimised together in one graph, so a step costs one forward and one backward
    pass regardless of how many pairs there are. Each step evaluates the surrogate, forms the
    loss, takes an Adam step on the codes and positions, and then projects the result back into
    the region of legal sizes and positions. Every ``analytical_snap_interval`` steps the
    children are put back in contact with their parents, and once more at the end before the
    outputs are produced.

    The loss has three parts: a squared hinge that pushes the feasibility logit past a margin,
    squared hinges on the two criteria, and a drift penalty on how far size and position have
    moved from where they started. The criterion thresholds are relaxed at the beginning of the
    run and tightened to their true values by the end, which lets a badly infeasible pair make
    progress before it has to meet the real requirement.

    Note that both the drift and the caps are expressed in metres, obtained by decoding, and
    not in the latent code. This is what keeps the run comparable with the parameter branch.

    Args:
        size_anchor_list, size_active_list: Full edge lengths of parent and part under test,
            one ``(3,)`` tensor per pair, in the scaled frame.
        pos_anchor_list, pos_active_list: Their centres, same convention.
        gnn_model: The trained surrogate. It must carry ``target_mean`` and ``target_std``,
            since the loop compares its outputs against thresholds in metres.
        scale_factor: Factor between the scaled frame and metres.
        num_steps: Maximum number of Adam steps.
        lr: Adam learning rate.
        hinge_scale: Factor on the two criterion hinges. They are squared metres, so of order
            1e-6, and without a factor of this magnitude they would be invisible next to the
            feasibility term.
        hinge_weights: Relative weight of the overlap and the thickness hinge.
        lambda_size: Weight of the drift in size, the term that keeps the repaired block close
            to the one that was designed.
        lambda_pos: Weight of the drift in position.
        size_min_m, size_max_m: Absolute limits on any edge length, in metres.
        early_stop_interval: How often the pairs are tested and the satisfied ones frozen.
        pos_drift_box_m: Half-width of the box around the starting position a centre may move
            in, in metres, or ``None`` for no box.
        size_alpha: Fraction by which an edge length may grow, or ``None`` to disable the
            per-pair box and keep only the absolute limits.
        asymmetric_tangential: When true, the two tangential axes may only grow and only the
            contact axis may shrink. Shrinking a tangential axis would reduce overlap, which is
            never what the repair wants.
        contact_shrink_alpha: Fraction by which the contact axis may shrink, so the thickness
            can be brought under its threshold.
        contact_axes_list: Contact axis per pair. Inferred from the initial geometry when
            omitted, and held fixed for the whole run either way.
        analytical_snap_interval: Steps between contact corrections, or 0 to disable them.
        analytical_floor_correction: Whether the correction also lifts a block above the table.
        reset_adam_state_on_snap: Whether to clear the Adam state after a correction.
        freeze_pos_if_overlap_ok: When true, pairs whose overlap already passes at the start
            keep their position fixed and only their code is optimised.
        early_stop_p_binary: Feasibility probability a pair must reach to be frozen early. It
            is stricter than the 0.5 the final verdict uses, so that a pair is only set aside
            once the surrogate is confident rather than merely past the decision boundary.
        verbose: Whether to print progress.
        trace: If a list is passed, one record per pair per step is appended to it. Purely
            diagnostic; it does not change the optimisation.

    Returns:
        A dictionary with ``size_active_out`` and ``pos_active_out``, both ``(N, 3)`` tensors in
        the scaled frame, the first decoded from the final codes; ``z_active_out`` of shape
        ``(N, latent_dim)``; ``reg_out_m`` of shape ``(N, 2)`` in metres; ``p_binary`` of shape
        ``(N,)``; and ``success_mask`` of shape ``(N,)``, the surrogate's own verdict.
    """
    device = size_anchor_list[0].device
    N = len(size_anchor_list)

    size_min_scaled = float(size_min_m * scale_factor)
    size_max_scaled = float(size_max_m * scale_factor)

    if verbose:
        print(
            f"[adam_repair-latent] N={N} num_steps={num_steps} lr={lr} "
            f"hinge_scale={hinge_scale} hinge_weights={hinge_weights}  "
            f"lambda_size={lambda_size} lambda_pos={lambda_pos}"
        )

    # Both are frozen. The repair differentiates through the decoder with respect to the code,
    # never with respect to the weights of either network.
    encoder = get_box_encoder(device=str(device))
    decoder = get_box_decoder(device=str(device))

    size_anchor_all = torch.stack([s.clone().detach() for s in size_anchor_list])
    pos_anchor_all = torch.stack([p.clone().detach() for p in pos_anchor_list])
    size_active_start = torch.stack([s.clone().detach() for s in size_active_list])
    pos_active_start = torch.stack([p.clone().detach() for p in pos_active_list])

    # The drift is measured against the sizes as designed, not against the reconstruction of
    # the starting code. Anything the autoencoder loses on the first roundtrip would otherwise
    # be charged to the repair.
    size_active_orig = size_active_start.clone()
    pos_active_orig = pos_active_start.clone()

    with torch.no_grad():
        z_anchor_all = encode_size(encoder, size_anchor_all)
        z_active_start = encode_size(encoder, size_active_start)

    z_active = nn.Parameter(z_active_start.clone())
    pos_active = nn.Parameter(pos_active_start.clone())

    optimizer = torch.optim.Adam([z_active, pos_active], lr=lr)
    gnn_model.eval()

    target_mean = gnn_model.target_mean
    target_std = gnn_model.target_std

    repaired_mask = torch.zeros(N, dtype=torch.bool, device=device)
    w_ov, w_th = float(hinge_weights[0]), float(hinge_weights[1])

    pos_box_scaled = (
        float(pos_drift_box_m) * scale_factor
        if pos_drift_box_m is not None and pos_drift_box_m > 0
        else None
    )

    # The contact axis is fixed once, from the initial geometry, and used both for the
    # asymmetric size box and for the contact correction. Re-inferring it during the run would
    # let a pair change which face it is joined on halfway through, and the two would then be
    # working against each other.
    if contact_axes_list is not None:
        contact_axes = torch.tensor(
            contact_axes_list,
            dtype=torch.long,
            device=device,
        )
    else:
        from src.analytical_metrics import infer_face

        contact_axes = torch.zeros(N, dtype=torch.long, device=device)
        sf = float(scale_factor)
        for i in range(N):
            pos_p_m = pos_anchor_all[i].cpu().numpy() / sf
            he_p_m = (size_anchor_all[i] / 2.0).cpu().numpy() / sf
            pos_c_m = pos_active_orig[i].cpu().numpy() / sf
            he_c_m = (size_active_orig[i] / 2.0).cpu().numpy() / sf
            face = infer_face(pos_p_m, he_p_m, pos_c_m, he_c_m)
            # The face code is 2 * axis + side, so integer division recovers the axis.
            contact_axes[i] = face // 2

    if size_alpha is not None and size_alpha > 0:
        size_box_hi = size_active_orig * (1.0 + size_alpha)
        if asymmetric_tangential:
            # Only the contact axis is allowed below its designed value, because that is the
            # one thickness is measured on. A tangential axis that shrank would lose overlap,
            # so its lower bound is the original size.
            contact_mask = F.one_hot(contact_axes, num_classes=3).bool()
            contact_lo = size_active_orig * (1.0 - float(contact_shrink_alpha))
            size_box_lo = torch.where(contact_mask, contact_lo, size_active_orig)
        else:
            size_box_lo = size_active_orig * (1.0 - size_alpha)
        size_lo_final = torch.clamp(size_box_lo, min=size_min_scaled)
        size_hi_final = torch.clamp(size_box_hi, max=size_max_scaled)
        if verbose:
            if asymmetric_tangential:
                print(
                    f"[adam_repair-latent] hard caps: pos_box=±{pos_drift_box_m*1000:.0f}mm, "
                    f"tangential grow=+{size_alpha*100:.0f}% (no shrink), "
                    f"contact-axis shrink up to {contact_shrink_alpha*100:.0f}% "
                    f"(absolute min={size_min_m*1000:.0f}mm)"
                )
            else:
                print(
                    f"[adam_repair-latent] hard caps: pos_box=±{pos_drift_box_m*1000:.0f}mm, "
                    f"size_alpha=±{size_alpha*100:.0f}% (symmetric)"
                )
    else:
        size_lo_final = None
        size_hi_final = None

    if verbose and analytical_snap_interval > 0:
        print(
            f"[adam_repair-latent] analytical snap every {analytical_snap_interval} steps "
            f"(floor_correction={analytical_floor_correction}, "
            f"reset_adam={reset_adam_state_on_snap})"
        )

    # The freeze is decided once, from the surrogate's prediction before any step. A pair whose
    # overlap already passes has nothing to gain from moving, and moving it could only lose the
    # contact it has.
    if freeze_pos_if_overlap_ok:
        with torch.no_grad():
            batch_probe = _build_batch(
                z_anchor_all, z_active, pos_anchor_all, pos_active,
            )
            reg_probe_std, _, _ = gnn_model(
                batch_probe.x, batch_probe.edge_index, batch_probe.edge_attr,
            )
            reg_probe = destandardize(reg_probe_std, target_mean, target_std)
            pos_freeze_mask = (reg_probe[:, 0] >= THRESH_OVERLAP).to(device)
        if verbose:
            print(
                f"[adam_repair-latent] freeze_pos_if_overlap_ok: "
                f"{int(pos_freeze_mask.sum())}/{N} samples have pos frozen "
                f"(z-only repair)"
            )
    else:
        pos_freeze_mask = None

    for step in range(num_steps):
        # The hinges start at half the required overlap and twice the allowed thickness and
        # reach the true values at the last step. A pair that is far from feasible would
        # otherwise sit in a region where the hinge is large and flat from the start; easing
        # the requirement in gives it a gradient to follow. The early-stop test and the final
        # verdict always use the true thresholds, so this relaxation never enters a result.
        progress = step / max(num_steps - 1, 1)
        relax = 1.0 + (1.0 - progress)
        thresh_ov = THRESH_OVERLAP / relax
        thresh_th = THRESH_THICKNESS * relax

        batch = _build_batch(
            z_anchor_all, z_active, pos_anchor_all, pos_active,
        )
        reg_std, bin_logit, *_ = gnn_model(batch.x, batch.edge_index, batch.edge_attr)
        reg_m = destandardize(reg_std, target_mean, target_std)
        ov_m = reg_m[:, 0]
        th_m = reg_m[:, 1]

        # Pairs already accepted contribute nothing further. Continuing to push them would
        # keep driving the logit up long after the pair is feasible, at the cost of drift.
        active_mask = (~repaired_mask).float()
        # Guard against a division by zero on the step in which the last pair is accepted.
        n_active = active_mask.sum().clamp(min=1.0)

        # The push acts on the logit, not on the probability: a sigmoid saturates and its
        # gradient vanishes exactly where the pair is still clearly infeasible. The margin of
        # two stops the term once the pair is comfortably on the feasible side, so the drift
        # penalty is not fighting an unbounded push.
        binary_loss = (
            F.relu(2.0 - bin_logit.squeeze(1)) ** 2 * active_mask
        ).sum() / n_active

        hinge_ov = F.relu(thresh_ov - ov_m) ** 2
        hinge_th = F.relu(th_m - thresh_th) ** 2
        reg_loss = (
            hinge_scale
            * ((w_ov * hinge_ov + w_th * hinge_th) * active_mask).sum()
            / n_active
        )

        # The drift is taken on the decoded size, so it penalises a change in the block as
        # built. A penalty on the code would be a distance in an arbitrary space and would not
        # correspond to anything the parameter branch charges for.
        size_active_decoded = decode_z(decoder, z_active)
        size_drift = (
            (size_active_decoded - size_active_orig).pow(2).sum(dim=1) * active_mask
        ).sum() / n_active
        pos_drift = (
            (pos_active - pos_active_orig).pow(2).sum(dim=1) * active_mask
        ).sum() / n_active

        loss = (
            binary_loss + reg_loss + lambda_size * size_drift + lambda_pos * pos_drift
        )

        optimizer.zero_grad()
        loss.backward()
        # The loss passes through the surrogate and, for the drift term, through the decoder as
        # well, so a single badly conditioned pair can produce a very large gradient. Clipping
        # keeps one such pair from displacing the whole batch in a single step.
        torch.nn.utils.clip_grad_norm_([z_active, pos_active], max_norm=1.0)
        # Zeroing the gradient rather than excluding the parameter, so that Adam still sees a
        # tensor of the same shape and the frozen entries simply never move.
        if pos_freeze_mask is not None and pos_active.grad is not None:
            pos_active.grad[pos_freeze_mask] = 0
        optimizer.step()

        # The projection runs after every step, not once at the end. Bounds applied only at the
        # end would let the optimiser spend the whole run in a region it is not allowed to
        # reach and then be pulled back to something it never evaluated.
        with torch.no_grad():
            _project_z_via_size_clamp(
                z_active,
                encoder,
                decoder,
                size_lo_final,
                size_hi_final,
                size_min_scaled,
                size_max_scaled,
            )

            if pos_box_scaled is not None:
                pos_diff = pos_active.data - pos_active_orig
                pos_active.data = pos_active_orig + pos_diff.clamp(
                    -pos_box_scaled, pos_box_scaled,
                )

            if (
                analytical_snap_interval > 0
                and (step + 1) % analytical_snap_interval == 0
                and not bool(repaired_mask.all())
            ):
                size_active_now = decode_z(decoder, z_active)
                n_adj = _apply_analytical_snap(
                    pos_anchor_all,
                    size_anchor_all,
                    pos_active,
                    size_active_now,
                    repaired_mask,
                    contact_axes,
                    scale_factor,
                    floor_correction=analytical_floor_correction,
                )
                # The correction can push a centre outside its box, so the box is reapplied
                # after it.
                if pos_box_scaled is not None:
                    pos_diff = pos_active.data - pos_active_orig
                    pos_active.data = pos_active_orig + pos_diff.clamp(
                        -pos_box_scaled, pos_box_scaled,
                    )
                # The correction is a discontinuous jump. Adam's accumulated moments describe
                # the trajectory before it and would carry the optimiser straight back off the
                # face, so they are discarded.
                if reset_adam_state_on_snap:
                    optimizer.state.clear()
                if verbose and n_adj > 0:
                    print(
                        f"[adam_repair-latent] step {step+1}: analytical snap adjusted "
                        f"{n_adj}/{int((~repaired_mask).sum().item())} active samples"
                    )

        # The trace records the decoded thickness beside the predicted one at the same step,
        # which is what makes a bluff visible: the block is only genuinely thinner if the two
        # fall together. Recorded after the step and after the projection, so it describes the
        # state the next step starts from.
        if trace is not None:
            with torch.no_grad():
                size_dec_tr = decode_z(decoder, z_active)
                b_tr = _build_batch(z_anchor_all, z_active, pos_anchor_all, pos_active)
                reg_tr_std, bin_tr, *_ = gnn_model(b_tr.x, b_tr.edge_index, b_tr.edge_attr)
                reg_tr = destandardize(reg_tr_std, target_mean, target_std)
                p_tr = torch.sigmoid(bin_tr).squeeze(1)
                for i in range(N):
                    ca = int(contact_axes[i].item())
                    trace.append(
                        {
                            "step": step + 1,
                            "sample": i,
                            "contact_axis": ca,
                            "thickness_decoded_mm": float(size_dec_tr[i, ca]) / scale_factor * 1000.0,
                            "thickness_pred_mm": float(reg_tr[i, 1]) * 1000.0,
                            "overlap_pred_mm": float(reg_tr[i, 0]) * 1000.0,
                            "p_binary": float(p_tr[i]),
                            "repaired": bool(repaired_mask[i]),
                        }
                    )

        # Pairs that already satisfy the true thresholds are frozen rather than optimised
        # further, and the loop exits once every pair is. Checking every step would double the
        # number of forward passes for no gain.
        if (step + 1) % early_stop_interval == 0:
            with torch.no_grad():
                batch_es = _build_batch(
                    z_anchor_all, z_active, pos_anchor_all, pos_active,
                )
                reg_es_std, bin_es, *_ = gnn_model(
                    batch_es.x, batch_es.edge_index, batch_es.edge_attr
                )
                reg_es = destandardize(reg_es_std, target_mean, target_std)
                p_es = torch.sigmoid(bin_es).squeeze(1).cpu().numpy()
                r_es = reg_es.cpu().numpy()

                ov_ok = r_es[:, 0] > THRESH_OVERLAP + EARLY_STOP_MARGIN
                th_ok = r_es[:, 1] < THRESH_THICKNESS - EARLY_STOP_MARGIN
                bn_ok = p_es > early_stop_p_binary
                all_ok = ov_ok & th_ok & bn_ok
                repaired_mask = torch.from_numpy(all_ok).to(device)

                if all_ok.all():
                    if verbose:
                        print(
                            f"[adam_repair-latent] early stop @ step {step+1} — "
                            f"all {N} repaired"
                        )
                    break

    # The final correction is applied to every pair, including the ones frozen early, and
    # without the position box. It is what guarantees that the geometry reported below is one
    # in which every child actually touches its parent.
    if analytical_snap_interval > 0:
        with torch.no_grad():
            size_active_now = decode_z(decoder, z_active)
            n_adj = _apply_analytical_snap(
                pos_anchor_all,
                size_anchor_all,
                pos_active,
                size_active_now,
                repaired_mask=torch.zeros(N, dtype=torch.bool, device=device),
                contact_axes=contact_axes,
                scale_factor=scale_factor,
                floor_correction=analytical_floor_correction,
            )
            if verbose:
                print(
                    f"[adam_repair-latent] final analytical snap adjusted "
                    f"{n_adj}/{N} samples"
                )

    # The reported prediction comes from a forward pass on the corrected geometry, not from the
    # last pass inside the loop. The correction moved the blocks, so anything measured before
    # it would describe a configuration that is not the one being returned.
    with torch.no_grad():
        batch_f = _build_batch(
            z_anchor_all, z_active, pos_anchor_all, pos_active,
        )
        reg_f_std, bin_f, *_ = gnn_model(
            batch_f.x, batch_f.edge_index, batch_f.edge_attr
        )
        reg_f = destandardize(reg_f_std, target_mean, target_std)
        p_binary = torch.sigmoid(bin_f).squeeze(1).cpu().numpy()
        reg_out_m = reg_f.cpu().numpy()
        size_active_out = decode_z(decoder, z_active).detach()

    success_mask = is_repaired_batch(reg_out_m, p_binary)

    if verbose:
        print(
            f"[adam_repair-latent] finished: "
            f"{int(success_mask.sum())}/{N} pass GNN gate"
        )

    return {
        "size_active_out": size_active_out,
        "pos_active_out": pos_active.detach(),
        "z_active_out": z_active.detach(),
        "reg_out_m": reg_out_m,
        "p_binary": p_binary,
        "success_mask": success_mask,
    }


def adam_repair_single(
    size_anchor: torch.Tensor,
    size_active: torch.Tensor,
    pos_anchor: torch.Tensor,
    pos_active: torch.Tensor,
    gnn_model: GNN,
    scale_factor: float,
    **kwargs,
) -> dict:
    """Repair a single pair, unpacking the batched result into scalars and ``(3,)`` tensors.

    A convenience wrapper only; the optimisation is exactly the one
    :func:`adam_repair_batched` performs on a batch of one.
    """
    out = adam_repair_batched(
        [size_anchor],
        [size_active],
        [pos_anchor],
        [pos_active],
        gnn_model,
        scale_factor,
        **kwargs,
    )
    return {
        "size_active_out": out["size_active_out"][0],
        "pos_active_out": out["pos_active_out"][0],
        "z_active_out": out["z_active_out"][0],
        "reg_out_m": out["reg_out_m"][0],
        "p_binary": float(out["p_binary"][0]),
        "success": bool(out["success_mask"][0]),
    }
