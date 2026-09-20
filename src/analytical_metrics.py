"""Closed-form feasibility quantities for a pair of boxes, and the contact snap.

The formulas are the ones the dataset generator uses to label a configuration, extracted so
that they can be evaluated on any pair of positions and half-extents rather than only on rows
of the training table. That is what makes them useful twice over: as the labeller for designs
that never went through the simulator, and as the independent verdict against which the
surrogate's answer is checked after a repair. A repair the surrogate calls feasible and these
functions call infeasible is the bluff the evaluation counts.

Everything is in metres, in the frame the dataset uses: the table surface lies at z equal to
zero, and a box is given by its centre and its three half-extents.

Because a pair is labelled per contact face, the face code appears throughout. It runs 0 to 5
as plus and minus x, plus and minus y, plus and minus z, so the contact axis is the code
halved and the sign follows from its parity. The two axes that are not the contact axis are
the tangential ones, and the overlap is measured on those.

Gap and table interference are not part of the criteria of Part I -- the generators construct
configurations in which both are zero -- but the two functions are kept because they are the
quantities a chain of more than two blocks would have to check.
"""

from __future__ import annotations

import numpy as np
import torch


# The two screwdriving limits, in metres. They repeat the values of ``config.gnn`` on purpose:
# this module is meant to stay usable without the model configuration, so the numbers are
# duplicated and have to be kept in step by hand.
THRESH_OVERLAP: float = 0.010
THRESH_THICKNESS: float = 0.020
THRESH_UNDER_SURFACE: float = 0.001
THRESH_GAP: float = 0.001

# Slack on the comparisons. Positions and sizes travel through single precision and through a
# millimetre-rounded CSV, so a design that is exactly on a limit must not be called a failure.
EVAL_TOLERANCE: float = 2e-4


def _contact_axis(face: int) -> int:
    if face in (0, 1):
        return 0
    if face in (2, 3):
        return 1
    return 2


def _tangential_axes(face: int) -> tuple[int, int]:
    if face in (0, 1):
        return (1, 2)
    if face in (2, 3):
        return (0, 2)
    return (0, 1)


def _face_sign(face: int) -> float:
    return 1.0 if face % 2 == 0 else -1.0


def infer_face(
    pos_p: np.ndarray, he_p: np.ndarray,
    pos_c: np.ndarray, he_c: np.ndarray,
) -> int:
    """Recover the contact face of a pair from its geometry alone.

    Needed because a design that comes from the generative stage carries positions and sizes
    but no face annotation, whereas both the labelling and the repair are stated per face.

    The contact axis is taken to be the one on which the two boxes are furthest apart relative
    to their own extent, and the sign of the offset on that axis picks the face. Choosing
    instead the axis on which the boxes are closest to touching would be wrong for a pair that
    already overlaps fully on one axis and has a gap on the intended contact axis: a part
    floating above its parent would be read as a side contact.

    Args:
        pos_p, he_p: Centre and half-extents of the parent, shape ``(3,)``.
        pos_c, he_c: Centre and half-extents of the child, shape ``(3,)``.

    Returns:
        The face code 0 to 5. All four inputs must share one frame; a common scale factor
        cancels.
    """
    pos_p, he_p = np.asarray(pos_p), np.asarray(he_p)
    pos_c, he_c = np.asarray(pos_c), np.asarray(he_c)
    delta = pos_c - pos_p
    ratio = np.abs(delta) / (he_p + he_c + 1e-12)
    ax = int(np.argmax(ratio))
    return 2 * ax + (0 if delta[ax] >= 0 else 1)


def analytical_overlap(
    pos_p: np.ndarray, he_p: np.ndarray,
    pos_c: np.ndarray, he_c: np.ndarray,
    face: int,
) -> float:
    """Size of the contact patch, in metres.

    The two boxes are intersected on each of the two tangential axes and the smaller of the
    two lengths is returned, because the screwdriver needs room in both directions at once.
    Larger is better; the criterion is ``overlap >= THRESH_OVERLAP``.
    """
    pos_p, he_p = np.asarray(pos_p), np.asarray(he_p)
    pos_c, he_c = np.asarray(pos_c), np.asarray(he_c)
    ov = np.zeros(3)
    for d in range(3):
        min_p, max_p = pos_p[d] - he_p[d], pos_p[d] + he_p[d]
        min_c, max_c = pos_c[d] - he_c[d], pos_c[d] + he_c[d]
        ov[d] = max(0.0, min(max_p, max_c) - max(min_p, min_c))
    t1, t2 = _tangential_axes(face)
    return float(min(ov[t1], ov[t2]))


def analytical_thickness(he_c: np.ndarray, face: int) -> float:
    """Material the screw has to pass through, in metres.

    That is the child's own edge length along the contact axis, so the parent does not enter
    the expression. Smaller is better; the criterion is ``thickness < THRESH_THICKNESS``.
    """
    ax = _contact_axis(face)
    return float(2.0 * he_c[ax])


def analytical_planned_gap(
    pos_p: np.ndarray, he_p: np.ndarray,
    pos_c: np.ndarray, he_c: np.ndarray,
    face: int,
) -> float:
    """Signed distance between the two contact faces, in metres.

    Negative means the boxes interpenetrate, positive that they float apart. Zero for every
    configuration the generators of Part I produce, which is why it is not one of the two
    criteria; it becomes informative again once a chain is repaired joint by joint.
    """
    delta = np.asarray(pos_c) - np.asarray(pos_p)
    ax = _contact_axis(face)
    sign = _face_sign(face)
    return float(sign * delta[ax] - (he_p[ax] + he_c[ax]))


def analytical_under_surface(pos_c: np.ndarray, he_c: np.ndarray) -> float:
    """How far the child reaches below the table surface, in metres, clamped at zero.

    Zero for every configuration the generators of Part I produce.
    """
    bottom = float(pos_c[2] - he_c[2])
    return max(0.0, -bottom)


def is_repaired_analytical(
    overlap: float,
    thickness: float,
    tol: float = EVAL_TOLERANCE,
) -> tuple[bool, list[str]]:
    """Decide feasibility of a joint from the two measured quantities.

    This is the independent verdict the repair is scored against, so it deliberately checks
    only what the criteria of Part I state. Gap and table interference are left out because
    they are zero by construction; :func:`is_repaired_analytical_full` checks all four.

    Args:
        overlap: Contact patch in metres.
        thickness: Material along the contact axis in metres.
        tol: Slack on both comparisons.

    Returns:
        Whether the joint passes, together with one readable string per violated criterion.
    """
    violated: list[str] = []
    if overlap < THRESH_OVERLAP - tol:
        violated.append(f"overlap={overlap:.6f} < {THRESH_OVERLAP}")
    if thickness >= THRESH_THICKNESS + tol:
        violated.append(f"thickness={thickness:.6f} >= {THRESH_THICKNESS}")
    return len(violated) == 0, violated


def is_repaired_analytical_full(
    under: float, overlap: float, thickness: float, gap: float,
    tol: float = EVAL_TOLERANCE,
) -> tuple[bool, list[str]]:
    """Feasibility against all four limits, including gap and table interference."""
    violated: list[str] = []
    if under >= THRESH_UNDER_SURFACE + tol:
        violated.append(f"under_surface={under:.6f} >= {THRESH_UNDER_SURFACE}")
    if overlap < THRESH_OVERLAP - tol:
        violated.append(f"overlap={overlap:.6f} < {THRESH_OVERLAP}")
    if thickness >= THRESH_THICKNESS + tol:
        violated.append(f"thickness={thickness:.6f} >= {THRESH_THICKNESS}")
    if abs(gap) >= THRESH_GAP + tol:
        violated.append(f"|gap|={abs(gap):.6f} >= {THRESH_GAP}")
    return len(violated) == 0, violated


def compute_all_analytical(
    pos_p: np.ndarray, he_p: np.ndarray,
    pos_c: np.ndarray, he_c: np.ndarray,
    face: int,
) -> dict[str, float]:
    """All four quantities in one call, under the names the surrogate's outputs use."""
    return {
        "overlap": analytical_overlap(pos_p, he_p, pos_c, he_c, face),
        "thickness": analytical_thickness(he_c, face),
        "planned_gap": analytical_planned_gap(pos_p, he_p, pos_c, he_c, face),
        "under_surface": analytical_under_surface(pos_c, he_c),
    }


def snap_to_contact_face(
    pos_p: np.ndarray,
    he_p: np.ndarray,
    pos_c: np.ndarray,
    he_c: np.ndarray,
    contact_axis: int | None = None,
    sign: float | None = None,
) -> tuple[np.ndarray, float, int, float]:
    """Slide the child along the contact axis until its face rests exactly on the parent's.

    The gradient-based repair moves a block freely and can leave it slightly sunk into its
    parent or slightly detached; the physical joint requires neither. This restores contact
    without disturbing the two criteria: only the contact-axis coordinate is written, so the
    tangential position and all three edge lengths are untouched, and the overlap and the
    thickness come out of the snap unchanged.

    Args:
        pos_p, he_p: Centre and half-extents of the parent, in metres.
        pos_c, he_c: Centre and half-extents of the child, in metres.
        contact_axis: Axis to snap on. Inferred from the geometry when not given, by the same
            rule :func:`infer_face` uses.
        sign: Direction from parent to child along that axis. Inferred when not given; a child
            whose centre coincides with the parent's on that axis is pushed in the positive
            direction.

    Returns:
        A tuple of the new child centre of shape ``(3,)``, the signed shift in metres
        (positive where a gap was closed, negative where a penetration was undone), the
        contact axis actually used, and the gap in metres before the snap.
    """
    pos_p_a = np.asarray(pos_p, dtype=np.float64)
    he_p_a = np.asarray(he_p, dtype=np.float64)
    pos_c_a = np.asarray(pos_c, dtype=np.float64)
    he_c_a = np.asarray(he_c, dtype=np.float64)

    delta = pos_c_a - pos_p_a
    if contact_axis is None:
        ratio = np.abs(delta) / (he_p_a + he_c_a + 1e-12)
        contact_axis = int(np.argmax(ratio))
    if sign is None:
        s = float(np.sign(delta[contact_axis]))
        sign = s if s != 0.0 else 1.0

    gap_before = float(abs(delta[contact_axis]) - (he_p_a[contact_axis] + he_c_a[contact_axis]))

    target = pos_p_a[contact_axis] + sign * (he_p_a[contact_axis] + he_c_a[contact_axis])
    shift = float(target - pos_c_a[contact_axis])

    pos_c_snapped = pos_c_a.copy()
    pos_c_snapped[contact_axis] = target
    return pos_c_snapped.astype(np.float32), shift, int(contact_axis), gap_before


def _contact_axes_array(face: np.ndarray) -> np.ndarray:
    """Contact axis of each face code, shape ``(N,)``."""
    contact = np.empty(len(face), dtype=np.int64)
    contact[face <= 1] = 0
    contact[(face >= 2) & (face <= 3)] = 1
    contact[face >= 4] = 2
    return contact


def analytical_overlap_array(
    pos_p: np.ndarray, he_p: np.ndarray,
    pos_c: np.ndarray, he_c: np.ndarray,
    face: np.ndarray,
) -> np.ndarray:
    """:func:`analytical_overlap` over a whole split: ``(N, 3)`` geometry, ``(N,)`` faces."""
    N = pos_p.shape[0]
    ov = np.zeros((N, 3), dtype=np.float32)
    for d in range(3):
        min_p = pos_p[:, d] - he_p[:, d]
        max_p = pos_p[:, d] + he_p[:, d]
        min_c = pos_c[:, d] - he_c[:, d]
        max_c = pos_c[:, d] + he_c[:, d]
        ov[:, d] = np.maximum(0.0, np.minimum(max_p, max_c) - np.maximum(min_p, min_c))
    contact = _contact_axes_array(face)
    all_axes = np.array([0, 1, 2])
    # Selecting the two tangential axes per row through a mask keeps the whole split in one
    # array; the reshape is safe because exactly one axis is excluded in every row.
    mask = (np.broadcast_to(all_axes, (N, 3)) != contact[:, None])
    tang_ov = ov[mask].reshape(N, 2)
    return np.minimum(tang_ov[:, 0], tang_ov[:, 1]).astype(np.float32)


def analytical_thickness_array(he_c: np.ndarray, face: np.ndarray) -> np.ndarray:
    """:func:`analytical_thickness` over a whole split: ``(N, 3)`` extents, ``(N,)`` faces."""
    N = he_c.shape[0]
    contact = _contact_axes_array(face)
    return (2.0 * he_c[np.arange(N), contact]).astype(np.float32)


# The three functions below read a shape out of a signed distance decoder instead of out of
# three edge lengths. Nothing on this branch calls them: Part I represents a part by its edge
# lengths, so its geometry is already explicit. They are what the latent formulation needs in
# order to ask what a code actually decodes to, and are kept so that the analytical checks
# live in one place across the parts of the thesis.
def _decode_sdf_grid(
    decoder,
    z: torch.Tensor,
    resolution: int = 48,
    bounds: float = 1.0,
    device: torch.device | None = None,
) -> np.ndarray:
    """Evaluate the decoder on a cubic grid, returning an array of shape ``(r, r, r)``."""
    lin = np.linspace(-bounds, bounds, resolution)
    xx, yy, zz = np.meshgrid(lin, lin, lin, indexing="ij")
    grid_points = torch.from_numpy(
        np.stack([xx, yy, zz], axis=-1).reshape(-1, 3).astype(np.float32)
    )
    if device is not None:
        grid_points = grid_points.to(device)
    z_exp = z.unsqueeze(0).expand(grid_points.shape[0], -1)
    with torch.no_grad():
        sdf = decoder(z_exp, grid_points).cpu().numpy()
    return sdf.reshape(resolution, resolution, resolution)


def extract_he_and_offset_from_mesh(
    decoder,
    z: torch.Tensor,
    resolution: int = 48,
    bounds: float = 1.0,
    device: torch.device = torch.device("cpu"),
) -> tuple[np.ndarray, np.ndarray]:
    """Measure the bounding box of the shape a code decodes to.

    Args:
        decoder: Signed distance decoder, called as ``decoder(z, points)``.
        z: One shape code.
        resolution: Samples per axis of the grid the surface is extracted from.
        bounds: Half-width of that grid in the decoder's own frame.
        device: Device the grid is evaluated on.

    Returns:
        The half-extents of shape ``(3,)`` and the displacement of the box centre from the
        origin of shape ``(3,)``, both in the decoder's frame. A displacement away from zero
        means the decoded shape no longer sits where the representation assumes it does.
        Zeros are returned when the sampled field has no surface in it at all.
    """
    # Imported here rather than at the top: the closed-form part of this module is used far
    # more often than the mesh part and should not pull in an image processing dependency.
    from skimage.measure import marching_cubes

    sdf = _decode_sdf_grid(decoder, z, resolution, bounds, device)
    try:
        spacing = tuple(2.0 * bounds / (sdf.shape[i] - 1) for i in range(3))
        verts, _, _, _ = marching_cubes(sdf, level=0.0, spacing=spacing)
        # Marching cubes returns coordinates counted from the corner of the grid; the shift
        # puts them back into the frame the decoder was queried in.
        verts = verts - bounds
    except (ValueError, RuntimeError):
        return np.zeros(3, dtype=np.float32), np.zeros(3, dtype=np.float32)

    if len(verts) < 4:
        return np.zeros(3, dtype=np.float32), np.zeros(3, dtype=np.float32)

    aabb_min = verts.min(axis=0)
    aabb_max = verts.max(axis=0)
    he = ((aabb_max - aabb_min) / 2.0).astype(np.float32)
    offset = ((aabb_max + aabb_min) / 2.0).astype(np.float32)
    return he, offset


def mesh_boxiness_score(
    decoder,
    z: torch.Tensor,
    resolution: int = 48,
    bounds: float = 1.0,
    device: torch.device = torch.device("cpu"),
) -> float:
    """How much of its own bounding box the decoded shape fills, between zero and one.

    A value of one means the shape is still an axis-aligned box; a value well below one means
    the code has been driven somewhere the decoder no longer produces one. Cheaper than
    comparing the surfaces themselves.
    """
    from skimage.measure import marching_cubes

    sdf = _decode_sdf_grid(decoder, z, resolution, bounds, device)
    try:
        spacing = tuple(2.0 * bounds / (sdf.shape[i] - 1) for i in range(3))
        verts, faces, _, _ = marching_cubes(sdf, level=0.0, spacing=spacing)
    except (ValueError, RuntimeError):
        return 0.0
    if len(verts) < 4:
        return 0.0

    # Volume from the divergence theorem over the triangles. The absolute value has to be
    # taken on the sum, not on each term: the contributions are signed, and taking them
    # positive one by one would inflate the volume.
    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    signed_contribs = np.einsum("ij,ij->i", v0, np.cross(v1, v2))
    mesh_vol = float(abs(signed_contribs.sum()) / 6.0)

    aabb = verts.max(axis=0) - verts.min(axis=0)
    aabb_vol = float(np.prod(aabb))
    if aabb_vol <= 0:
        return 0.0
    # The two volumes come from the same triangulation, so their ratio can exceed one by a
    # rounding error; the clamp keeps the score inside its stated range.
    return min(1.0, mesh_vol / aabb_vol)
