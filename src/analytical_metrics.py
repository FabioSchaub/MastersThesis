"""Closed-form geometry that decides whether a pair of blocks is buildable.

The same formulas the dataset generator uses to produce its labels, extracted as a library of
plain functions so they can be evaluated on any centre and half-extent, including repaired
configurations that lie far outside the distribution the surrogate was trained on. Everything
is in metres, in the frame the design CSVs use, with the table surface at height zero.

This module is the judge, not a second model. It is what turns a claim of the surrogate into a
verdict: the repair queries the network, this file says what the resulting geometry actually
is, and a pair the network calls feasible while these functions call it infeasible is a bluff.
It also provides :func:`snap_to_contact_face`, the correction the repair applies so that a
repaired child sits face to face against its parent instead of floating or interpenetrating.

Of the four metrics only overlap and thickness carry signal on this dataset: the generator
places every child exactly on its parent's face and above the table, so gap and under-surface
are zero by construction. They are implemented anyway because a chain of more than two blocks
makes them non-degenerate again.
"""

from __future__ import annotations

import numpy as np
import torch


# The two screwdriving criteria, in metres. They duplicate config.gnn.thresh_overlap_min and
# config.gnn.thresh_thickness_max deliberately: this module is the reference the surrogate is
# judged against, so it must not silently follow a configuration the surrogate was not trained
# under. If the configuration changes, both have to be changed by hand.
THRESH_OVERLAP: float = 0.010
THRESH_THICKNESS: float = 0.020
THRESH_UNDER_SURFACE: float = 0.001
THRESH_GAP: float = 0.001
# Slack on the comparisons, well below the millimetre the criteria are stated in, so that a
# configuration snapped exactly onto a threshold is not failed by floating-point noise.
EVAL_TOLERANCE: float = 2e-4


# The face code is 2 * axis + (0 for the positive side, 1 for the negative one), so 0/1 are
# +X/-X, 2/3 are +Y/-Y and 4/5 are +Z/-Z. The dataset generator uses the same encoding.
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
    """Infer which face of the parent the child sits on, as a code from 0 to 5.

    The contact axis is the one on which the two centres are furthest apart relative to the
    sizes involved, and the sign of that offset selects the positive or the negative face.
    Both blocks must be given in the same frame; the ratio is dimensionless, so a common
    scaling cancels.

    The alternative rule, taking the axis on which the blocks come closest to touching, is
    wrong for the configurations that occur here: a child floating above its parent overlaps
    fully in the two tangential axes and has a gap on the contact axis, and that rule would
    pick a tangential axis.

    Args:
        pos_p, he_p: Centre and half-extents of the parent, both ``(3,)`` in metres.
        pos_c, he_c: Centre and half-extents of the child.

    Returns:
        The face code, used everywhere else in this module to select the contact axis.
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
    """Return the horizontal contact between the two blocks, in metres.

    The smaller of the two axis-aligned overlaps on the axes tangential to the contact face.
    Taking the minimum rather than the area is what makes the quantity a length: the screw
    needs room in both tangential directions, so the tighter of the two decides. Larger is
    better and the criterion is ``overlap >= THRESH_OVERLAP``.

    Args:
        pos_p, he_p: Centre and half-extents of the parent, ``(3,)`` in metres.
        pos_c, he_c: Centre and half-extents of the child.
        face: Contact face code from 0 to 5.
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
    """Return the material the screw has to pass through, in metres.

    The full edge length of the child along the contact axis, so twice its half-extent there.
    It depends on the child alone: the parent is what is screwed into, not through. Smaller is
    better and the criterion is ``thickness < THRESH_THICKNESS``.
    """
    ax = _contact_axis(face)
    return float(2.0 * he_c[ax])


def analytical_planned_gap(
    pos_p: np.ndarray, he_p: np.ndarray,
    pos_c: np.ndarray, he_c: np.ndarray,
    face: int,
) -> float:
    """Return the signed distance between the two faces along the contact axis, in metres.

    Positive is a gap, negative is interpenetration. On the dataset of this branch it is
    always about zero because the generator places the child on the face; it becomes
    meaningful for a chain of more than two blocks, where a child cannot satisfy every parent.
    """
    delta = np.asarray(pos_c) - np.asarray(pos_p)
    ax = _contact_axis(face)
    sign = _face_sign(face)
    return float(sign * delta[ax] - (he_p[ax] + he_c[ax]))


def analytical_under_surface(pos_c: np.ndarray, he_c: np.ndarray) -> float:
    """Return how far the child reaches below the table surface, in metres, clamped at zero.

    The table is at height zero in this frame. Always zero on the dataset of this branch.
    """
    bottom = float(pos_c[2] - he_c[2])
    return max(0.0, -bottom)


def is_repaired_analytical(
    overlap: float,
    thickness: float,
    tol: float = EVAL_TOLERANCE,
) -> tuple[bool, list[str]]:
    """Decide feasibility from the two criteria that carry signal on this dataset.

    This is the verdict every reported repair rate of Part II is counted with. Gap and
    under-surface are omitted because the generator guarantees them; use
    :func:`is_repaired_analytical_full` where that guarantee does not hold.

    Args:
        overlap: Horizontal contact in metres.
        thickness: Joint thickness in metres.
        tol: Slack applied to both comparisons so that a value snapped onto a threshold is not
            failed by rounding.

    Returns:
        A pair ``(ok, violations)``, where ``violations`` lists the criteria that failed with
        their values, and is empty when ``ok`` is true.
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
    """Decide feasibility from all four criteria, for cases where gap and table clearance are
    not guaranteed by construction. Same return convention as
    :func:`is_repaired_analytical`."""
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
    """Evaluate all four metrics in one call.

    Returns:
        A dictionary with the keys ``overlap``, ``thickness``, ``planned_gap`` and
        ``under_surface``, all in metres. The first two are named as the surrogate's
        regression outputs so that predicted and true values can be compared by key.
    """
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
    """Slide the child along the contact axis until its face touches the parent exactly.

    This is the correction the repair applies after changing a size: growing or shrinking a
    block moves its faces, so a child that was in contact before the step is left floating or
    interpenetrating. Only the contact-axis coordinate of the child's centre is touched, which
    leaves both criteria intact, overlap because the tangential position does not move and
    thickness because no size changes. Sizes are never adjusted here.

    Args:
        pos_p, he_p: Centre and half-extents of the parent, ``(3,)`` in metres.
        pos_c, he_c: Centre and half-extents of the child.
        contact_axis: Axis index to snap along. Inferred from the geometry when omitted, by
            the same rule as :func:`infer_face`.
        sign: Direction from parent to child along that axis. Inferred when omitted.

    Returns:
        A tuple ``(pos_c_snapped, shift_m, contact_axis, gap_before_m)``: the corrected child
        centre with only the contact-axis entry changed; the signed shift in metres, positive
        when a gap was closed and negative when interpenetration was resolved; the axis that
        was used; and the signed distance between the faces before the correction.
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
        # Exactly concentric centres give no direction; pick the positive face arbitrarily
        # rather than collapsing the child onto the parent.
        sign = s if s != 0.0 else 1.0

    gap_before = float(abs(delta[contact_axis]) - (he_p_a[contact_axis] + he_c_a[contact_axis]))

    target = pos_p_a[contact_axis] + sign * (he_p_a[contact_axis] + he_c_a[contact_axis])
    shift = float(target - pos_c_a[contact_axis])

    pos_c_snapped = pos_c_a.copy()
    pos_c_snapped[contact_axis] = target
    return pos_c_snapped.astype(np.float32), shift, int(contact_axis), gap_before


# The vectorised variants below exist for evaluating a whole split at once; they compute the
# same quantities as the scalar functions above.
def _contact_axes_array(face: np.ndarray) -> np.ndarray:
    """Map ``(N,)`` face codes to ``(N,)`` contact axis indices."""
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
    """Evaluate :func:`analytical_overlap` over ``(N, 3)`` centres and half-extents and
    ``(N,)`` face codes, returning ``(N,)`` overlaps in metres."""
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
    # Every row has exactly one contact axis, so masking it out always leaves two entries and
    # the reshape to (N, 2) is safe. Boolean indexing flattens in row-major order, which keeps
    # the two tangential values of a row together.
    mask = (np.broadcast_to(all_axes, (N, 3)) != contact[:, None])
    tang_ov = ov[mask].reshape(N, 2)
    return np.minimum(tang_ov[:, 0], tang_ov[:, 1]).astype(np.float32)


def analytical_thickness_array(he_c: np.ndarray, face: np.ndarray) -> np.ndarray:
    """Evaluate :func:`analytical_thickness` over ``(N, 3)`` half-extents and ``(N,)`` face
    codes, returning ``(N,)`` thicknesses in metres."""
    N = he_c.shape[0]
    contact = _contact_axes_array(face)
    return (2.0 * he_c[np.arange(N), contact]).astype(np.float32)


# The three functions below read the shape a code stands for out of the auxiliary
# signed-distance decoder of stage 1, not out of the box decoder used in the repair loop. They
# answer a different question: whether a code still describes a box at all, rather than what
# half-extents it maps to.
def _decode_sdf_grid(
    decoder,
    z: torch.Tensor,
    resolution: int = 48,
    bounds: float = 1.0,
    device: torch.device | None = None,
) -> np.ndarray:
    """Evaluate the auxiliary decoder on a regular grid in the normalised frame, returning a
    cube of signed distances of side ``resolution``."""
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
    """Recover the bounding box of the shape a code decodes to, by marching cubes.

    Args:
        decoder: The auxiliary signed-distance decoder of stage 1.
        z: A single latent code, shape ``(latent_dim,)``.
        resolution: Number of grid samples per axis.
        bounds: Half-width of the sampled cube in the normalised frame.
        device: Device the grid is evaluated on.

    Returns:
        A pair ``(he, offset)``, both ``(3,)`` in the normalised frame: the half-extents of the
        bounding box, and the displacement of its centre from the origin. The offset is zero
        for a code the encoder could have produced and departs from zero once the optimiser
        has moved the code away from that set, which makes it a symptom rather than a value to
        be used.

        Both are zero when the field has no zero level set inside the sampled cube.
    """
    # Imported here rather than at the top so that the purely analytical part of this module
    # does not depend on scikit-image.
    from skimage.measure import marching_cubes

    sdf = _decode_sdf_grid(decoder, z, resolution, bounds, device)
    try:
        spacing = tuple(2.0 * bounds / (sdf.shape[i] - 1) for i in range(3))
        verts, _, _, _ = marching_cubes(sdf, level=0.0, spacing=spacing)
        # Marching cubes returns coordinates in grid units starting at zero; shift them back
        # into the symmetric frame the decoder was queried in.
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
    """Return how much of its own bounding box the decoded shape fills, between 0 and 1.

    One for an axis-aligned box, less for anything the auxiliary decoder has deformed. It is a
    cheap test of whether a code still stands for a box, which is what a repair working in
    latent space can quietly lose. Zero when the field has no zero level set inside the
    sampled cube.
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

    # Volume by the divergence theorem over the triangles. The absolute value is taken on the
    # sum, not per triangle: the contributions are signed and taking them positive one by one
    # would count the far side of the shape twice.
    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    signed_contribs = np.einsum("ij,ij->i", v0, np.cross(v1, v2))
    mesh_vol = float(abs(signed_contribs.sum()) / 6.0)

    aabb = verts.max(axis=0) - verts.min(axis=0)
    aabb_vol = float(np.prod(aabb))
    if aabb_vol <= 0:
        return 0.0
    # A discretised surface can enclose marginally more than its own bounding box; clamp so
    # the score stays interpretable as a fraction.
    return min(1.0, mesh_vol / aabb_vol)
