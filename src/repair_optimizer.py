"""Repair a joint by descending the surrogate's gradient with respect to the child's geometry.

The free variables are the three edge lengths and the three coordinates of the part under
test; the parent is frozen. Adam moves those six numbers until the surrogate reports a joint
that satisfies both criteria, and a projection after every step keeps the design inside a
region where the surrogate's answer still means something.

This is the repair of Part I, and it is deliberately plain: one optimiser, no stages, no
fallback, and no representation between the design and the model. What the optimiser writes
back are the same quantities the design was specified in, so a repaired design can be handed
straight to the cell.

Three mechanisms keep the descent honest, and they are the reason the repair does not simply
learn to fool the surrogate:

* the hinges act on the two physical quantities in metres rather than on the feasibility
  probability alone, so an improvement has to show up as a change of the geometry;
* the projection after every step caps how far a block may move and how far it may grow,
  which bounds how far the design can travel from the data the surrogate was fitted on;
* every few steps the child is snapped back onto its parent's face, so the geometry the
  surrogate is asked about is one that could actually be assembled.

Graph inputs are built here rather than reused from the dataset module, and they have to match
it exactly: node ``[size_x, size_y, size_z, role_0, role_1]``, edge
``[dx, dy, dz, dist_xy, dist_xz, dist_yz]``, both in the scaled domain, which for the
checkpoint of this branch is raw metres.
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
from src.gnn import GNN  # noqa: E402

# The two screwdriving limits in metres, read from the configuration so that the repair and
# the training objective cannot drift apart. Note that ``pipeline/repair_strategies.py``
# rebinds ``THRESH_THICKNESS`` on this module at import time.
THRESH_OVERLAP: float = float(config.gnn.thresh_overlap_min)
THRESH_THICKNESS: float = float(config.gnn.thresh_thickness_max)

# Both are zero, so the early stop and the commit gate use the limits as they stand. A margin
# here would mean reporting as repaired a design that only passes because the check was
# loosened, which is precisely what the evaluation of the repair has to be able to rule out.
EARLY_STOP_MARGIN: float = 0.0
EVAL_TOLERANCE: float = 0.0

# Mirrors ``gnn_dataset_preparation.OVERLAP_LOG_EPS``; the two have to agree or the inverse of
# the standardisation is not the inverse.
OVERLAP_LOG_EPS: float = 1e-5


def destandardize(
    reg_pred_std: torch.Tensor,
    target_mean: torch.Tensor,
    target_std: torch.Tensor,
) -> torch.Tensor:
    """Turn the surrogate's regression outputs back into metres.

    The overlap column is standardised in logarithmic space and the thickness column directly.
    Which convention applies is read off the sign of the first mean, negative for a logarithm
    of a length in metres, so a checkpoint standardised the other way is handled correctly
    without a flag having to be passed around.

    Args:
        reg_pred_std: Standardised outputs, shape ``(..., 2)``.
        target_mean: Means saved with the checkpoint, shape ``(2,)``.
        target_std: Standard deviations saved with the checkpoint, shape ``(2,)``.

    Returns:
        Overlap and thickness in metres, same shape as the input. The operation is
        differentiable, which is what lets the hinges act on physical quantities.
    """
    out = reg_pred_std * target_std + target_mean
    if target_mean[0].item() < 0.0:
        out = out.clone()
        out[..., 0] = torch.exp(out[..., 0]) - OVERLAP_LOG_EPS
    return out


def is_repaired_batch(
    reg_out_m: np.ndarray,
    p_binary: np.ndarray,
    tol: float = EVAL_TOLERANCE,
) -> np.ndarray:
    """Commit gate over a batch: both quantities within their limits and the verdict positive.

    All three conditions are required. The regression outputs alone would accept a design the
    surrogate itself does not call buildable, and the verdict alone would accept one whose
    quantities sit on the wrong side of a limit.

    Args:
        reg_out_m: Overlap and thickness in metres, shape ``(N, 2)``.
        p_binary: Feasibility probability per pair, shape ``(N,)``.
        tol: Slack on the two limits.

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
    """Commit gate for one pair, with a readable string per failed condition."""
    violated: list[str] = []
    if reg_out_m[0] < THRESH_OVERLAP - tol:
        violated.append(f"overlap={reg_out_m[0]:.4f} < {THRESH_OVERLAP}")
    if reg_out_m[1] > THRESH_THICKNESS + tol:
        violated.append(f"thickness={reg_out_m[1]:.4f} > {THRESH_THICKNESS}")
    if p_binary < 0.5:
        violated.append(f"p_binary={p_binary:.3f} < 0.5")
    return len(violated) == 0, violated


def _node_features(
    size_anchor_scaled: torch.Tensor,  # (3,)
    size_active_scaled: torch.Tensor,  # (3,)
) -> torch.Tensor:
    """Node features of one pair, shape ``(2, 5)``, anchor first."""
    device = size_anchor_scaled.device
    type_0 = torch.tensor([1.0, 0.0], device=device, dtype=torch.float)
    type_1 = torch.tensor([0.0, 1.0], device=device, dtype=torch.float)
    return torch.stack(
        [
            torch.cat([size_anchor_scaled, type_0]),
            torch.cat([size_active_scaled, type_1]),
        ]
    )


def _edge_features(
    pos_anchor_scaled: torch.Tensor,  # (3,)
    pos_active_scaled: torch.Tensor,  # (3,)
) -> torch.Tensor:
    """Edge features of one pair, shape ``(2, 6)``, forward edge then reverse."""
    delta = pos_active_scaled - pos_anchor_scaled
    dx, dy, dz = delta[0], delta[1], delta[2]
    # The offset is a free variable here, so it can pass through zero during the descent. The
    # square root would have no gradient there; the offset inside it keeps the step finite.
    eps = 1e-8
    dist_xy = torch.sqrt(dx**2 + dy**2 + eps)
    dist_xz = torch.sqrt(dx**2 + dz**2 + eps)
    dist_yz = torch.sqrt(dy**2 + dz**2 + eps)
    fwd = torch.stack([dx, dy, dz, dist_xy, dist_xz, dist_yz])
    rev = torch.stack([-dx, -dy, -dz, dist_xy, dist_xz, dist_yz])
    return torch.stack([fwd, rev])


def build_graph(
    size_anchor_scaled: torch.Tensor,
    size_active_scaled: torch.Tensor,
    pos_anchor_scaled: torch.Tensor,
    pos_active_scaled: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Assemble the surrogate's inputs for one pair.

    Args:
        size_anchor_scaled, size_active_scaled: Edge lengths of parent and child, shape
            ``(3,)`` each, in the scaled domain.
        pos_anchor_scaled, pos_active_scaled: Their centres, shape ``(3,)`` each, same domain.

    Returns:
        Node features, edge list, edge features and the graph assignment vector, ready to be
        passed to :meth:`src.gnn.GNN.forward`.
    """
    device = size_anchor_scaled.device
    x = _node_features(size_anchor_scaled, size_active_scaled)
    edge_attr = _edge_features(pos_anchor_scaled, pos_active_scaled)
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long, device=device)
    batch = torch.zeros(2, dtype=torch.long, device=device)
    return x, edge_index, edge_attr, batch


def _build_batch(
    size_anchor_all: torch.Tensor,  # (N, 3) scaled
    size_active_all: torch.Tensor,  # (N, 3) scaled
    pos_anchor_all: torch.Tensor,  # (N, 3) scaled
    pos_active_all: torch.Tensor,  # (N, 3) scaled
) -> Batch:
    """Pack ``N`` pairs into one batched graph, in the order the caller passed them."""
    device = size_anchor_all.device
    N = size_anchor_all.shape[0]
    type_0 = torch.tensor([1.0, 0.0], device=device, dtype=torch.float)
    type_1 = torch.tensor([0.0, 1.0], device=device, dtype=torch.float)
    edge_index_template = torch.tensor(
        [[0, 1], [1, 0]], dtype=torch.long, device=device
    )

    graphs = []
    for i in range(N):
        x_i = torch.stack(
            [
                torch.cat([size_anchor_all[i], type_0]),
                torch.cat([size_active_all[i], type_1]),
            ]
        )
        ea_i = _edge_features(pos_anchor_all[i], pos_active_all[i])
        graphs.append(Data(x=x_i, edge_index=edge_index_template, edge_attr=ea_i))
    return Batch.from_data_list(graphs)


def _apply_analytical_snap(
    pos_anchor_all: torch.Tensor,  # (N, 3) scaled
    size_anchor_all: torch.Tensor,  # (N, 3) scaled
    pos_active: nn.Parameter,  # (N, 3) scaled — modified in-place
    size_active: nn.Parameter,  # (N, 3) scaled — read-only here
    repaired_mask: torch.Tensor,  # (N,) bool — these are skipped
    contact_axes: torch.Tensor,  # (N,) long {0,1,2}
    scale_factor: float,
    floor_correction: bool = True,
) -> int:
    """Put every pair back into contact, and back above the table, writing into ``pos_active``.

    Pairs already marked repaired are skipped so that a design that has passed the gate is not
    disturbed again.

    Returns:
        How many pairs moved by more than a millimetre, which is the number worth reporting;
        smaller corrections happen on nearly every call.
    """
    from src.analytical_metrics import snap_to_contact_face

    sf = float(scale_factor)
    pos_p_m = (pos_anchor_all / sf).detach().cpu().numpy()
    he_p_m = (size_anchor_all / (2.0 * sf)).detach().cpu().numpy()
    pos_c_m = (pos_active.data / sf).detach().cpu().numpy()
    he_c_m = (size_active.data / (2.0 * sf)).detach().cpu().numpy()
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
        if abs(shift_m) > 1e-3:
            n_adjusted += 1

    if floor_correction:
        # The table is at z = 0 and the optimiser knows nothing about it, so a block it has
        # pushed downwards is lifted until its underside rests on the surface.
        bottom = pos_c_m[:, 2] - he_c_m[:, 2]
        below = bottom < 0.0
        if below.any():
            pos_c_m[below, 2] = he_c_m[below, 2]

    pos_active.data = torch.from_numpy(pos_c_m * sf).to(
        device=pos_active.device,
        dtype=pos_active.dtype,
    )
    return n_adjusted


def adam_repair_batched(
    size_anchor_list: list[torch.Tensor],  # each (3,) scaled
    size_active_list: list[torch.Tensor],  # each (3,) scaled
    pos_anchor_list: list[torch.Tensor],  # each (3,) scaled
    pos_active_list: list[torch.Tensor],  # each (3,) scaled
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
) -> dict:
    """Repair a batch of joints by gradient descent on the children's sizes and positions.

    The objective has four parts, averaged over the pairs that are not yet repaired: a term
    that pushes the feasibility logit up until it saturates, one squared hinge per criterion,
    and two quadratic penalties on how far size and position have moved from where they
    started. The hinges are stated on the quantities in metres, not on the probability, so the
    optimiser cannot satisfy the objective by making the surrogate confident about a geometry
    that has not changed.

    The hinge scale converts squared metres to squared millimetres. That is what makes the two
    kinds of term comparable: with the default, a joint that misses the overlap limit by four
    millimetres contributes about sixteen, against a saturating feasibility term of at most
    about twenty-five, so neither one silently dominates.

    After every step the design is projected back into a box around where it started. The
    thresholds are also relaxed at the beginning and tightened to their true values by the
    end, so the descent is not asked to satisfy the full criterion from a design that violates
    it badly.

    Args:
        size_anchor_list, size_active_list: Edge lengths of parent and child, one ``(3,)``
            tensor per pair, in the scaled domain.
        pos_anchor_list, pos_active_list: Their centres, one ``(3,)`` tensor per pair.
        gnn_model: The surrogate, carrying its ``target_mean`` and ``target_std`` buffers.
        scale_factor: Factor between the scaled domain and metres; one for the checkpoint of
            this branch.
        num_steps: Number of Adam steps.
        lr: Adam step size.
        hinge_scale: Multiplier on both hinges, see above.
        hinge_weights: Relative weight of the overlap and the thickness hinge.
        lambda_size: Weight of the penalty on changing the edge lengths.
        lambda_pos: Weight of the penalty on moving the block.
        size_min_m: Smallest edge length in metres a block may be reduced to, regardless of
            the relative budget.
        size_max_m: Largest edge length in metres.
        early_stop_interval: How often the gate is evaluated; pairs that pass are frozen.
        pos_drift_box_m: Half-width in metres of the box the centre may move in, per axis.
            ``None`` removes the cap.
        size_alpha: Relative growth budget per axis; an edge may not exceed its original
            length by more than this fraction.
        asymmetric_tangential: When set, the two tangential axes may only grow. Letting them
            shrink would raise the surrogate's overlap for the wrong reason, since a smaller
            child can be centred more favourably on its parent.
        contact_shrink_alpha: Fraction by which the contact axis may shrink. It is large
            because thickness is the criterion that is most often violated and the only way to
            satisfy it is to remove material along that axis.
        contact_axes_list: Contact axis per pair. Inferred from the initial geometry when not
            given.
        analytical_snap_interval: How often the child is put back onto its parent's face.
            Zero disables the correction.
        analytical_floor_correction: Whether the same correction also lifts a block that has
            been pushed below the table.
        reset_adam_state_on_snap: Whether to clear the optimiser's momentum after a snap. The
            snap is a jump, and momentum accumulated before it points somewhere that no longer
            exists.
        freeze_pos_if_overlap_ok: When set, pairs whose overlap already passes are probed once
            at the start and then have their position gradient zeroed, so only the edge
            lengths are refined. The periodic snap still moves them, but only along the
            contact axis, which leaves the overlap alone.
        early_stop_p_binary: Confidence a pair must reach before it is frozen. Higher than the
            0.5 of the commit gate, so a pair is not released while it is still borderline.
        verbose: Whether to print progress.

    Returns:
        A mapping with the repaired edge lengths and centres, both ``(N, 3)`` in the scaled
        domain; ``reg_out_m`` of shape ``(N, 2)``, the surrogate's overlap and thickness in
        metres after the final snap; ``p_binary`` of shape ``(N,)``; and ``success_mask`` of
        shape ``(N,)``, the commit gate.

        The geometry is returned whether or not the gate passed, and one last snap is applied
        to every pair beforehand, so what comes out is always a joint that is in contact and
        above the table. Whether it is also feasible is what ``success_mask`` says.

        The caps mean that a design needing more shrinkage than the budget allows comes back
        as a failure rather than as a repair. That is deliberate: the alternative is a design
        the surrogate accepts because it has been pushed away from anything it was trained on.
    """
    device = size_anchor_list[0].device
    N = len(size_anchor_list)

    size_min_scaled = float(size_min_m * scale_factor)
    size_max_scaled = float(size_max_m * scale_factor)

    if verbose:
        print(
            f"[adam_repair] N={N} num_steps={num_steps} lr={lr} "
            f"hinge_scale={hinge_scale} hinge_weights={hinge_weights}  "
            f"lambda_size={lambda_size} lambda_pos={lambda_pos}"
        )

    size_anchor_all = torch.stack([s.clone().detach() for s in size_anchor_list])
    pos_anchor_all = torch.stack([p.clone().detach() for p in pos_anchor_list])
    size_active_start = torch.stack([s.clone().detach() for s in size_active_list])
    pos_active_start = torch.stack([p.clone().detach() for p in pos_active_list])

    size_active_orig = size_active_start.clone()
    pos_active_orig = pos_active_start.clone()

    size_active = nn.Parameter(size_active_start.clone())
    pos_active = nn.Parameter(pos_active_start.clone())

    optimizer = torch.optim.Adam([size_active, pos_active], lr=lr)
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

    # The contact axis is needed whether or not the caller supplied it: both the asymmetric
    # size box and the periodic snap are stated per axis.
    if contact_axes_list is not None:
        contact_axes = torch.tensor(
            contact_axes_list,
            dtype=torch.long,
            device=device,
        )
    else:
        # Inferred from the geometry as it is before the descent; the axis is a property of
        # the joint and must not follow the block around while it moves.
        from src.analytical_metrics import infer_face

        contact_axes = torch.zeros(N, dtype=torch.long, device=device)
        sf = float(scale_factor)
        for i in range(N):
            pos_p_m = pos_anchor_all[i].cpu().numpy() / sf
            he_p_m = (size_anchor_all[i] / 2.0).cpu().numpy() / sf
            pos_c_m = pos_active_orig[i].cpu().numpy() / sf
            he_c_m = (size_active_orig[i] / 2.0).cpu().numpy() / sf
            face = infer_face(pos_p_m, he_p_m, pos_c_m, he_c_m)
            contact_axes[i] = face // 2

    if size_alpha is not None and size_alpha > 0:
        # The budget is asymmetric on purpose. Growth is capped equally on all three axes,
        # but shrinking is allowed only along the contact axis, where it is the honest way to
        # satisfy the thickness limit. A tangential axis that were free to shrink would let
        # the surrogate report a better overlap without the joint improving.
        size_box_hi = size_active_orig * (1.0 + size_alpha)
        if asymmetric_tangential:
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
                    f"[adam_repair] hard caps: pos_box=±{pos_drift_box_m*1000:.0f}mm, "
                    f"tangential grow=+{size_alpha*100:.0f}% (no shrink), "
                    f"contact-axis shrink up to {contact_shrink_alpha*100:.0f}% "
                    f"(absolute min={size_min_m*1000:.0f}mm)"
                )
            else:
                print(
                    f"[adam_repair] hard caps: pos_box=±{pos_drift_box_m*1000:.0f}mm, "
                    f"size_alpha=±{size_alpha*100:.0f}% (symmetric)"
                )
    else:
        size_lo_final = None
        size_hi_final = None

    if verbose and analytical_snap_interval > 0:
        print(
            f"[adam_repair] analytical snap every {analytical_snap_interval} steps "
            f"(floor_correction={analytical_floor_correction}, "
            f"reset_adam={reset_adam_state_on_snap})"
        )

    # Decided once, from the geometry as handed in, and then held for the whole run. A pair
    # whose overlap is already satisfied should not have it traded away for thickness.
    if freeze_pos_if_overlap_ok:
        with torch.no_grad():
            batch_probe = _build_batch(
                size_anchor_all,
                size_active,
                pos_anchor_all,
                pos_active,
            )
            reg_probe_std, _, _ = gnn_model(
                batch_probe.x,
                batch_probe.edge_index,
                batch_probe.edge_attr,
            )
            reg_probe = destandardize(reg_probe_std, target_mean, target_std)
            pos_freeze_mask = (reg_probe[:, 0] >= THRESH_OVERLAP).to(device)
        if verbose:
            print(
                f"[adam_repair] freeze_pos_if_overlap_ok: "
                f"{int(pos_freeze_mask.sum())}/{N} samples have pos frozen "
                f"(size-only repair)"
            )
    else:
        pos_freeze_mask = None

    for step in range(num_steps):
        # The hinges start at half the overlap limit and twice the thickness limit and reach
        # the true values at the last step. A design that is far outside the feasible region
        # otherwise sees a hinge that is saturated from the first step and is pushed hard
        # enough to leave the caps behind before the geometry has improved.
        progress = step / max(num_steps - 1, 1)
        relax = 1.0 + (1.0 - progress)
        thresh_ov = THRESH_OVERLAP / relax
        thresh_th = THRESH_THICKNESS * relax

        batch = _build_batch(
            size_anchor_all,
            size_active,
            pos_anchor_all,
            pos_active,
        )
        reg_std, bin_logit, *_ = gnn_model(batch.x, batch.edge_index, batch.edge_attr)
        reg_m = destandardize(reg_std, target_mean, target_std)
        ov_m = reg_m[:, 0]
        th_m = reg_m[:, 1]

        active_mask = (~repaired_mask).float()
        n_active = active_mask.sum().clamp(min=1.0)

        # Saturating rather than unbounded: once the logit passes two the term is flat, so the
        # optimiser stops buying confidence it does not need and the hinges take over.
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

        size_drift = (
            (size_active - size_active_orig).pow(2).sum(dim=1) * active_mask
        ).sum() / n_active
        pos_drift = (
            (pos_active - pos_active_orig).pow(2).sum(dim=1) * active_mask
        ).sum() / n_active

        loss = (
            binary_loss + reg_loss + lambda_size * size_drift + lambda_pos * pos_drift
        )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_([size_active, pos_active], max_norm=1.0)
        if pos_freeze_mask is not None and pos_active.grad is not None:
            pos_active.grad[pos_freeze_mask] = 0
        optimizer.step()

        # Projection back into the feasible set of the free variables. Applied after the step
        # rather than built into the objective, so a cap cannot be traded against the hinges.
        with torch.no_grad():
            # Absolute bounds first: they hold whatever the relative budget allows.
            size_active.data.clamp_(min=size_min_scaled, max=size_max_scaled)

            if pos_box_scaled is not None:
                pos_diff = pos_active.data - pos_active_orig
                pos_active.data = pos_active_orig + pos_diff.clamp(
                    -pos_box_scaled,
                    pos_box_scaled,
                )

            if size_lo_final is not None and size_hi_final is not None:
                size_active.data = torch.minimum(
                    torch.maximum(size_active.data, size_lo_final),
                    size_hi_final,
                )

            # Contact is restored periodically, not only at the end: the surrogate was fitted
            # on joints whose faces touch, so a design that has drifted out of contact is
            # being asked about outside the regime the answer is meaningful in.
            if (
                analytical_snap_interval > 0
                and (step + 1) % analytical_snap_interval == 0
                and not bool(repaired_mask.all())
            ):
                n_adj = _apply_analytical_snap(
                    pos_anchor_all,
                    size_anchor_all,
                    pos_active,
                    size_active,
                    repaired_mask,
                    contact_axes,
                    scale_factor,
                    floor_correction=analytical_floor_correction,
                )
                # The snap ignores the caps, so the box is applied again behind it.
                if pos_box_scaled is not None:
                    pos_diff = pos_active.data - pos_active_orig
                    pos_active.data = pos_active_orig + pos_diff.clamp(
                        -pos_box_scaled,
                        pos_box_scaled,
                    )
                # Momentum built up before the jump points at a design that is gone; keeping
                # it would simply undo the snap over the next few steps.
                if reset_adam_state_on_snap:
                    optimizer.state.clear()
                if verbose and n_adj > 0:
                    print(
                        f"[adam_repair] step {step+1}: analytical snap adjusted "
                        f"{n_adj}/{int((~repaired_mask).sum().item())} active samples"
                    )

        # Pairs are released individually rather than the whole batch at once, so a joint that
        # is done is not pushed further while its neighbours in the batch are still moving.
        if (step + 1) % early_stop_interval == 0:
            with torch.no_grad():
                batch_es = _build_batch(
                    size_anchor_all,
                    size_active,
                    pos_anchor_all,
                    pos_active,
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
                            f"[adam_repair] early stop @ step {step+1} — "
                            f"all {N} repaired"
                        )
                    break

    # One last snap, over every pair including the ones that stopped early, so that what is
    # returned is always in contact and above the table. The position box is not re-applied
    # afterwards: at this point the geometric constraint takes precedence over the budget.
    if analytical_snap_interval > 0:
        with torch.no_grad():
            n_adj = _apply_analytical_snap(
                pos_anchor_all,
                size_anchor_all,
                pos_active,
                size_active,
                repaired_mask=torch.zeros(N, dtype=torch.bool, device=device),
                contact_axes=contact_axes,
                scale_factor=scale_factor,
                floor_correction=analytical_floor_correction,
            )
            if verbose:
                print(
                    f"[adam_repair] final analytical snap adjusted "
                    f"{n_adj}/{N} samples"
                )

    # The reported numbers come from the geometry that is actually returned, after the snap,
    # not from the last step of the descent.
    with torch.no_grad():
        batch_f = _build_batch(
            size_anchor_all,
            size_active,
            pos_anchor_all,
            pos_active,
        )
        reg_f_std, bin_f, *_ = gnn_model(
            batch_f.x, batch_f.edge_index, batch_f.edge_attr
        )
        reg_f = destandardize(reg_f_std, target_mean, target_std)
        p_binary = torch.sigmoid(bin_f).squeeze(1).cpu().numpy()
        reg_out_m = reg_f.cpu().numpy()

    success_mask = is_repaired_batch(reg_out_m, p_binary)

    if verbose:
        print(f"[adam_repair] finished: {int(success_mask.sum())}/{N} pass GNN gate")

    return {
        "size_active_out": size_active.detach(),
        "pos_active_out": pos_active.detach(),
        "reg_out_m": reg_out_m,
        "p_binary": p_binary,
        "success_mask": success_mask,
    }


def adam_repair_single(
    size_anchor: torch.Tensor,  # (3,) scaled
    size_active: torch.Tensor,  # (3,) scaled
    pos_anchor: torch.Tensor,  # (3,) scaled
    pos_active: torch.Tensor,  # (3,) scaled
    gnn_model: GNN,
    scale_factor: float,
    **kwargs,
) -> dict:
    """Repair a single joint.

    Returns:
        The same entries as :func:`adam_repair_batched` with the leading dimension removed:
        two ``(3,)`` tensors, a ``(2,)`` array in metres, a probability and, under the name
        ``success``, the commit gate for this one pair.
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
        "reg_out_m": out["reg_out_m"][0],
        "p_binary": float(out["p_binary"][0]),
        "success": bool(out["success_mask"][0]),
    }
