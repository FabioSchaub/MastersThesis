"""Repair of an infeasible configuration by descending the gradient of the surrogate.

A configuration of two parts is taken as given and the part under test is moved until the
surrogate calls the configuration feasible. What may be changed is the point of the experiment:

    size          one scalar that scales the whole bounding box. The simulator sized every part
                  by a single factor, so scaling one axis alone would take the pair out of the
                  set of configurations the surrogate was ever shown.
    placement     the lateral offset. The height is not free; see below.
    shape code    the latent code of the part. This is the branch under examination, and it is
                  the one that turns out to persuade the surrogate without changing the object.

Three properties of the loop carry the result and are easy to damage.

The height of the part under test is recomputed inside the forward expression, from the current
sizes, so that it rests on the top face of the base at every step. It is therefore a function of
the scale and not a variable, the contact is exact throughout, and there is no need to snap the
configuration back onto the base periodically, as a formulation with a free height would need.

The objective acts on the scores before the sigmoid, as squared hinges with margins. Past the
sigmoid a confident score has almost no gradient left, so an optimiser working on probabilities
stalls exactly where it should still be improving. The margins stop the descent once a score is
comfortably on the right side, rather than letting it push a single score to infinity.

The bounding box handed to the surrogate is the measured one, scaled. When the shape code is
free this is a real inconsistency: the code is changed but the box is not recomputed from the
shape it now describes, so the surrogate is told about a form and a size that need not belong
together. It is part of why the shape branch persuades the surrogate and not the physics.

What can be checked here is only partial. The overlap and the thickness follow from the
bounding boxes and are computed exactly. Tipping and the overall verdict do not; they exist
only inside the simulator, and the value used for them here is the surrogate's own opinion.
Any statement about real feasibility therefore has to come from replaying the result under
physics, not from this module.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from config.config import config  # noqa: E402
from src.gnn import SimAssemblyGNN  # noqa: E402

# The two criteria that can be checked without the simulator, in metres. They are the same
# values the simulator labelled with, taken from the configuration rather than restated here.
THRESH_OV = float(config.gnn.thresh_overlap_min)
THRESH_TH = float(config.gnn.thresh_thickness_max)


def load_sim_gnn(ckpt_path, device: str = "cpu"):
    """Load the surrogate together with the thresholds and target names it was saved with.

    The model is put in evaluation mode and its parameters are detached from the graph: the
    repair differentiates through the model with respect to the design, never with respect to
    the weights.

    Returns:
        The model, the names of the failure modes in the order the model predicts them, and the
        decision thresholds tuned on the validation split.
    """
    ck = torch.load(ROOT / ckpt_path if not Path(ckpt_path).is_absolute() else ckpt_path,
                    map_location=device, weights_only=False)
    num_fail = ck.get("num_fail", len(ck["fail_names"]))
    model = SimAssemblyGNN(
        node_dim=ck["node_dim"], edge_dim=ck["edge_dim"], hidden_dim=ck["hidden_dim"],
        heads=ck["heads"], dropout=ck["dropout"], head_hidden=ck["head_hidden"],
        num_fail=num_fail,
    )
    model.load_state_dict(ck["model"])
    model.eval().to(device)
    for p in model.parameters():
        p.requires_grad_(False)
    return model, ck["fail_names"], ck["thresholds"]


def active_latent_dims(latents: dict, frac: float = 0.05) -> np.ndarray:
    """Indices of the code entries that still vary across the shapes of the table.

    The compression leaves most entries at essentially one value. An entry that no shape ever
    moves is one the decoder has seen only at that value, so changing it during the repair
    would leave the set of shapes the decoder actually represents. Restricting the search to
    the varying entries is the attempt to keep it inside that set.

    Args:
        latents: The code table, mapping mesh name to code.
        frac: Fraction of the largest spread an entry must reach to count as varying.
    """
    Z = np.stack([np.asarray(v, dtype=np.float32).ravel() for v in latents.values()])
    std = Z.std(axis=0)
    return np.where(std > frac * std.max())[0]


def _edge_attr(pos_src: torch.Tensor, pos_dst: torch.Tensor) -> torch.Tensor:
    """Edge features of both directions, shape ``(2, 6)``, differentiable in the positions.

    A separate implementation of what the dataset module computes, because that one works on
    arrays and the repair needs the gradient to reach the position through it. The two have to
    agree exactly, including the sign convention of the offset.
    """
    d = pos_dst - pos_src
    # The distances are square roots and their derivative is unbounded at zero, which is
    # reached whenever two centres share a coordinate plane.
    eps = 1e-12
    dxy = torch.sqrt(d[0] ** 2 + d[1] ** 2 + eps)
    dxz = torch.sqrt(d[0] ** 2 + d[2] ** 2 + eps)
    dyz = torch.sqrt(d[1] ** 2 + d[2] ** 2 + eps)
    fwd = torch.stack([d[0], d[1], d[2], dxy, dxz, dyz])
    rev = torch.stack([-d[0], -d[1], -d[2], dxy, dxz, dyz])
    return torch.stack([fwd, rev])


def analytic_overlap_thickness(bbox_a, pos_a, bbox_b, pos_b) -> tuple[float, float]:
    """The two criteria that can be checked without the simulator, from the bounding boxes.

    The overlap is the smaller of the two horizontal intersection extents, which is the width
    of the region in which the two parts actually meet; the thickness is the vertical extent of
    the part under test. Both are read off axis-aligned boxes, so a rotated part is treated as
    its box and the overlap is an over-estimate for it.

    Args:
        bbox_a, pos_a: Full extents and centre of the base, in metres.
        bbox_b, pos_b: Full extents and centre of the part under test, in metres.

    Returns:
        The overlap and the thickness, both in metres.
    """
    bbox_a, pos_a, bbox_b, pos_b = (np.asarray(v, float) for v in (bbox_a, pos_a, bbox_b, pos_b))
    ov = []
    for k in (0, 1):
        lo = max(pos_a[k] - bbox_a[k] / 2, pos_b[k] - bbox_b[k] / 2)
        hi = min(pos_a[k] + bbox_a[k] / 2, pos_b[k] + bbox_b[k] / 2)
        ov.append(max(0.0, hi - lo))
    return float(min(ov)), float(bbox_b[2])


def repair_pair(
    z_anchor, bbox_anchor, pos_anchor,     # base (frozen), np arrays
    z_active0, bbox_active0, pos_active0,   # child start
    model, thresholds, *,
    opt_z: bool = True,
    active_dims=None,          # if set (indices), restrict z-repair to these LV-active dims
    opt_scale: bool = True,
    num_steps: int = 300,
    lr: float = 0.02,
    m_good: float = 3.0,
    m_fail: float = 2.0,
    w_fail: float = 0.3,
    lambda_z: float = 1.0,
    lambda_s: float = 1.0,
    lambda_pos: float = 10.0,
    s_lo: float = 0.4,
    s_hi: float = 2.0,
    device: str = "cpu",
    verbose: bool = False,
) -> dict:
    """Repair one configuration and report what it looked like before and after.

    The base is frozen throughout. Which of the three groups of variables is free is the only
    difference between the branches that are compared in the thesis, so the loss, the projection
    and the number of steps must stay the same when one of them is changed.

    Args:
        z_anchor, bbox_anchor, pos_anchor: The frozen base: its code, its full extents in metres
            and its centre in metres.
        z_active0, bbox_active0, pos_active0: The same for the part under test, as it starts.
        model: The surrogate. It standardises its own inputs, so raw values are passed to it.
        thresholds: The decision thresholds from the checkpoint; only the one for the overall
            verdict is used here.
        opt_z: Free the shape code. This is the branch that persuades the surrogate without
            changing the object.
        active_dims: Restrict the shape code to these entries; see :func:`active_latent_dims`.
        opt_scale: Free the size.
        m_good: Margin on the score of the verdict. The loss stops rewarding it beyond this.
        m_fail: Margin on the scores of the failure modes, in the other direction.
        w_fail: Weight of the failure modes against the verdict.
        lambda_z, lambda_s, lambda_pos: Weights of the three drift terms, which keep the repair
            close to the design it started from. Position is weighted most: a lateral move is
            the cheapest change to make and would otherwise absorb the whole repair.
        s_lo, s_hi: Bounds the size is clipped to after every step.

    Returns:
        The predicted probabilities and the two measured criteria, before and after; the
        repaired code, size, box and position; and three verdicts. ``geo_ok_after`` is measured
        and true, ``gnn_good_after`` is only the surrogate's opinion, and ``bluff`` is the
        disagreement between them, that is a configuration the surrogate accepts and the
        geometry rejects.
    """
    z_anchor = torch.as_tensor(z_anchor, dtype=torch.float32, device=device)
    bbox_anchor = torch.as_tensor(bbox_anchor, dtype=torch.float32, device=device)
    pos_anchor = torch.as_tensor(pos_anchor, dtype=torch.float32, device=device)
    z0 = torch.as_tensor(z_active0, dtype=torch.float32, device=device)
    bbox0 = torch.as_tensor(bbox_active0, dtype=torch.float32, device=device)
    pos0 = torch.as_tensor(pos_active0, dtype=torch.float32, device=device)

    type0 = torch.tensor([1.0, 0.0], device=device)
    type1 = torch.tensor([0.0, 1.0], device=device)
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long, device=device)
    good_idx = thresholds["good"]

    z_active = nn.Parameter(z0.clone(), requires_grad=opt_z)
    # A mask on the gradient rather than a smaller parameter, so that the node handed to the
    # surrogate keeps its full width whichever entries are free.
    z_mask = None
    if opt_z and active_dims is not None:
        z_mask = torch.zeros_like(z0)
        z_mask[torch.as_tensor(active_dims, dtype=torch.long, device=device)] = 1.0
    s = nn.Parameter(torch.ones(1, device=device), requires_grad=opt_scale)
    pos_xy = nn.Parameter(pos0[:2].clone())
    params = [pos_xy] + ([z_active] if opt_z else []) + ([s] if opt_scale else [])
    opt = torch.optim.Adam(params, lr=lr)

    def _forward():
        """Build the graph from the current variables and query the surrogate."""
        # The box is the measured one scaled, not the box of the shape the code now describes.
        # When the code is free the two need not agree; that mismatch is deliberate and is part
        # of what the comparison of the branches is about.
        bbox_active = bbox0 * s.reshape(())
        # The height follows from the sizes at every step, so the part rests exactly on the top
        # face of the base throughout and nothing has to be snapped back into contact later. It
        # carries the gradient of the scale, which is why the scale can be repaired at all.
        pos_z = pos_anchor[2] + 0.5 * bbox_anchor[2] + 0.5 * bbox_active[2]
        pos_active = torch.cat([pos_xy, pos_z.reshape(1)])
        # Raw values: the surrogate standardises its own inputs from the statistics it was
        # trained under, and pre-normalising here would apply them twice.
        x = torch.stack([
            torch.cat([z_anchor, bbox_anchor, type0]),
            torch.cat([z_active, bbox_active, type1]),
        ])
        ea = _edge_attr(pos_anchor, pos_active)
        fail_logits, good_logit = model(x, edge_index, ea)
        return fail_logits.reshape(-1), good_logit.reshape(()), bbox_active, pos_active

    def _probs():
        """The same forward pass, but as probabilities and detached, for reporting."""
        with torch.no_grad():
            fl, gl, bbox_a, pos_a = _forward()
            return (torch.sigmoid(fl).cpu().numpy(), float(torch.sigmoid(gl)),
                    bbox_a.cpu().numpy(), pos_a.cpu().numpy())

    fail0, good0, bbox_b0, pos_b0 = _probs()
    ov0, th0 = analytic_overlap_thickness(bbox_anchor.cpu().numpy(), pos_anchor.cpu().numpy(),
                                          bbox_b0, pos_b0)

    # The part is kept over the footprint of the base. The height is derived from contact, so
    # nothing in the formulation would otherwise stop the optimiser from sliding the part off
    # the side while the surrogate still reports it resting on something.
    lat_half = 0.5 * bbox_anchor[:2]

    for step in range(num_steps):
        fail_logits, good_logit, _, _ = _forward()
        # Both terms act on the scores before the sigmoid. Past the sigmoid a confident score
        # has almost no gradient left, so an objective on probabilities stops moving exactly
        # where it should still be improving. Squared hinges with margins: the pressure grows
        # with the shortfall, and stops once a score is far enough on the right side, rather
        # than rewarding one score being driven ever further.
        good_loss = F.relu(m_good - good_logit) ** 2
        fail_loss = (F.relu(fail_logits + m_fail) ** 2).sum()
        # Without these the optimiser would return whatever design the surrogate likes best,
        # which need have nothing to do with the one that was asked about.
        drift = (lambda_z * ((z_active - z0) ** 2).sum()
                 + lambda_s * (s.reshape(()) - 1.0) ** 2
                 + lambda_pos * ((pos_xy - pos0[:2]) ** 2).sum())
        loss = good_loss + w_fail * fail_loss + drift

        opt.zero_grad()
        loss.backward()
        if z_mask is not None and z_active.grad is not None:
            z_active.grad.mul_(z_mask)
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        # Projected after the step rather than enforced through the parameterisation, so that
        # every branch sees the same unconstrained gradient and only the feasible set differs.
        with torch.no_grad():
            if opt_scale:
                s.clamp_(s_lo, s_hi)
            pos_xy.data = torch.clamp(pos_xy.data, pos_anchor[:2] - lat_half,
                                      pos_anchor[:2] + lat_half)
        if verbose and (step + 1) % 50 == 0:
            print(f"    step {step+1}: loss {float(loss):.3f} p_good {float(torch.sigmoid(good_logit)):.3f}")

    fail1, good1, bbox_b1, pos_b1 = _probs()
    ov1, th1 = analytic_overlap_thickness(bbox_anchor.cpu().numpy(), pos_anchor.cpu().numpy(),
                                          bbox_b1, pos_b1)
    geo_ok = ov1 >= THRESH_OV and th1 <= THRESH_TH
    gnn_good = good1 > good_idx
    return {
        "p_good_before": good0, "p_good_after": good1,
        "fail_before": fail0, "fail_after": fail1,
        "overlap_before_m": ov0, "overlap_after_m": ov1,
        "thickness_before_m": th0, "thickness_after_m": th1,
        # Measured on the geometry, and therefore true.
        "geo_ok_after": bool(geo_ok),
        # The surrogate's own opinion, covering the two criteria that cannot be measured here.
        "gnn_good_after": bool(gnn_good),
        # The two disagreeing: the surrogate accepts a configuration the geometry rejects.
        "bluff": bool(gnn_good and not geo_ok),
        "z_out": z_active.detach().cpu().numpy(),
        "z_drift": float(torch.linalg.norm(z_active.detach() - z0)),
        "n_active_dims": (None if z_mask is None else int(z_mask.sum())),
        "scale_out": float(s.detach()),
        "bbox_out": bbox_b1, "pos_out": pos_b1,
    }


def _accept(out) -> bool:
    """Accept a repair only if the measured geometry and the surrogate both agree it is good."""
    return out["geo_ok_after"] and out["gnn_good_after"]


def repair_with_shape_search(
    pair: dict, orig_shape: str, model, thresholds,
    shape_latents: dict, shape_bboxes: dict, candidates: list[str], **kw,
) -> dict:
    """Repair by size and placement, changing the form only by choosing another shape outright.

    This is the honest counterpart to freeing the shape code. The original shape is repaired
    first with its form frozen; only if that fails is every candidate shape tried in turn, each
    repaired the same way, and the accepted one closest to the original design is returned. The
    form is thus decided by an enumeration over shapes that exist, never by a gradient through
    a code, so the result is by construction a shape the decoder actually represents.

    Args:
        pair: The arguments describing the configuration, as :func:`repair_pair` takes them.
        orig_shape: Name of the shape the part under test starts as; it is skipped in the
            enumeration, having already been tried.
        shape_latents: Code by shape name, the code a candidate is started from.
        shape_bboxes: A representative bounding box per shape name, the size a candidate is
            started from, since a candidate has no size of its own in this configuration.
        candidates: The shapes that may be substituted in.
        **kw: Passed through to :func:`repair_pair`, so both stages run identically.

    Returns:
        The shape that was settled on, whether it differs from the original, whether the result
        was accepted, the underlying repair, how many shapes were tried, and for a substitution
        the deviation from the original design that was minimised.
    """
    orig = repair_pair(**pair, model=model, thresholds=thresholds,
                       opt_z=False, opt_scale=True, **kw)
    if _accept(orig):
        return {"shape": orig_shape, "changed": False, "feasible": True,
                "result": orig, "n_tried": 1}

    best, best_dev, tried = None, float("inf"), 1
    for name in candidates:
        if name == orig_shape:
            continue
        cand = dict(pair)
        cand["z_active0"] = shape_latents[name].numpy()
        cand["bbox_active0"] = shape_bboxes[name]
        out = repair_pair(**cand, model=model, thresholds=thresholds,
                          opt_z=False, opt_scale=True, **kw)
        tried += 1
        if _accept(out):
            # Deviation adds a dimensionless size ratio to a lateral distance in metres. It
            # only orders the accepted candidates against each other and is not a length.
            dev = abs(out["scale_out"] - 1.0) + float(
                np.linalg.norm(out["pos_out"][:2] - pair["pos_active0"][:2]))
            if dev < best_dev:
                best, best_dev = (name, out), dev
    if best is not None:
        return {"shape": best[0], "changed": True, "feasible": True,
                "result": best[1], "n_tried": tried, "deviation": best_dev}
    return {"shape": None, "changed": False, "feasible": False,
            "result": orig, "n_tried": tried}
