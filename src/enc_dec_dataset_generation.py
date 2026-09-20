"""The shape vocabulary and the training data of the shape autoencoder.

One sample is a shape drawn at random from the vocabulary of fourteen types, together with two
point sets: a sample of its surface, which is what the encoder reads, and a set of query points
with the signed distance at each, which is what the decoder is asked to reproduce. Everything
is computed from the analytic distance functions and kept in memory; nothing is written to
disk, so a training run is defined by the seed and the configuration alone.

The module holds the three distance functions the vocabulary needs that are not extrusions of a
plane section, namely the box, the sphere and the ellipsoid, and the eleven others come from
:mod:`src.shape_primitives`. The cylinder, capsule and torus defined here belong to an older
vocabulary and are not part of the fourteen; the box, in contrast, is used, and its surface is
sampled by a dedicated routine rather than by the generic one.

Two conventions matter downstream. Characteristic dimensions are drawn log-uniformly, so a
shape is as likely to be slender as it is to be compact. And a shape can be returned in the
canonical frame, centred and uniformly scaled to a fixed largest extent, which is the frame the
encoder is trained and queried in: the code then describes only form, and size is carried
separately.
"""

from collections import Counter

import torch
import numpy as np
from pathlib import Path
import sys
from dataclasses import dataclass, field
from typing import Callable

# Add parent directory to path to import config
sys.path.insert(0, str(Path(__file__).parent.parent))
from config.config import config


def box_sdf(points: np.ndarray, half_extents: np.ndarray) -> np.ndarray:
    """Signed distance to an axis-aligned box centred on the origin.

    Args:
        points: Query points, shape ``(N, 3)``.
        half_extents: Half side lengths, shape ``(3,)``.

    Returns:
        Signed distances, shape ``(N,)``, negative inside.
    """
    q: np.ndarray = np.abs(points) - half_extents
    outside: np.ndarray = np.linalg.norm(np.maximum(q, 0), axis=1)
    inside: np.ndarray = np.minimum(np.max(q, axis=1), 0)
    return outside + inside


def sphere_sdf(points: np.ndarray, radius: float) -> np.ndarray:
    """Signed distance to a sphere centred on the origin."""
    return np.linalg.norm(points, axis=1) - radius


def cylinder_sdf(points: np.ndarray, radius: float, height: float) -> np.ndarray:
    """Signed distance to a cylinder whose axis is Y, of full height ``height``.

    Not part of the fourteen-shape vocabulary, whose cylinder has its axis along X; see
    :func:`src.shape_primitives.cylinder_x_sdf`.
    """
    d_xz: np.ndarray = np.linalg.norm(points[:, [0, 2]], axis=1) - radius
    d_y: np.ndarray = np.abs(points[:, 1]) - height / 2
    outside: np.ndarray = np.linalg.norm(
        np.maximum(np.stack([d_xz, d_y], axis=1), 0), axis=1
    )
    inside: np.ndarray = np.minimum(np.maximum(d_xz, d_y), 0)
    return outside + inside


def capsule_sdf(points: np.ndarray, radius: float, height: float) -> np.ndarray:
    """Signed distance to a capsule whose axis is Y.

    Not part of the fourteen-shape vocabulary; see :func:`src.shape_primitives.capsule_x_sdf`.

    Args:
        height: Distance between the two cap centres, so the full extent is
            ``height + 2 * radius``.
    """
    p: np.ndarray = points.copy()
    p[:, 1] = p[:, 1] - np.clip(p[:, 1], -height / 2, height / 2)
    return np.linalg.norm(p, axis=1) - radius


def torus_sdf(
    points: np.ndarray, major_radius: float, minor_radius: float
) -> np.ndarray:
    """Signed distance to a torus lying in the XZ plane, centred on the origin.

    Not part of the fourteen-shape vocabulary: shapes with a hole through them were left out
    because the encoder and the compressed code do not represent them reliably.

    Args:
        major_radius: Distance from the centre of the torus to the centre of the tube.
        minor_radius: Radius of the tube.
    """
    q: np.ndarray = np.stack(
        [np.linalg.norm(points[:, [0, 2]], axis=1) - major_radius, points[:, 1]],
        axis=1,
    )
    return np.linalg.norm(q, axis=1) - minor_radius


def ellipsoid_sdf(points: np.ndarray, radii: np.ndarray) -> np.ndarray:
    """Approximate signed distance to an ellipsoid of semi-axes ``radii``, shape ``(3,)``.

    Exact only for a sphere. Away from that the value is a first-order correction of the
    distance in the sphere the ellipsoid is a scaling of, so it is close to the true distance
    near the surface and increasingly wrong away from it. Since the sampler only ever needs the
    surface and its immediate neighbourhood, the approximation is not a limitation here.
    """
    k: np.ndarray = np.linalg.norm(points / radii, axis=1)
    grad_norm: np.ndarray = np.linalg.norm(points / (radii**2), axis=1) + 1e-8
    return k * (k - 1) / grad_norm


def union_sdf(sdf_a: np.ndarray, sdf_b: np.ndarray) -> np.ndarray:
    """Combine two distance fields into the solid that is inside either of them.

    The result is exact outside the union and only a bound inside, which is true of all three
    of these operations and is why the surface sampler projects rather than trusting the value.
    """
    return np.minimum(sdf_a, sdf_b)


def difference_sdf(sdf_a: np.ndarray, sdf_b: np.ndarray) -> np.ndarray:
    """Combine two distance fields into the first solid with the second removed from it."""
    return np.maximum(sdf_a, -sdf_b)


def intersection_sdf(sdf_a: np.ndarray, sdf_b: np.ndarray) -> np.ndarray:
    """Combine two distance fields into the solid that is inside both of them."""
    return np.maximum(sdf_a, sdf_b)


@dataclass
class ShapeData:
    """One generated shape: what the encoder reads, and what the decoder is asked to reproduce.

    Attributes:
        surface_points: Points on the surface, shape ``(N_surface, 3)``, the encoder input.
        query_points: Points in and around the solid, shape ``(N_query, 3)``.
        sdf_values: Signed distance at each query point, shape ``(N_query,)``, the decoder
            target.
        shape_type: Name of the vocabulary member this shape was drawn from.
        half_extents: Half extents, shape ``(3,)``, set for boxes and ``None`` for every other
            type, since only the box has an encoder that reads them directly.
        sdf_fn: The analytic distance function of this shape, kept so that the shape can be
            evaluated again later, for instance to check a sampled surface or to mesh it.
    """

    surface_points: np.ndarray
    query_points: np.ndarray
    sdf_values: np.ndarray
    shape_type: str
    # Excluded from the representation: both are arrays or closures that would make printing a
    # sample unreadable.
    half_extents: np.ndarray | None = field(
        default=None, repr=False
    )
    sdf_fn: Callable[[np.ndarray], np.ndarray] | None = field(
        default=None, repr=False
    )


def find_surface_points(
    sdf_fn: Callable[[np.ndarray], np.ndarray],
    n_surface: int = 2000,
    n_candidates: int = 100000,
    broad_threshold: float = 0.15,
    threshold: float = 0.02,
    bounds: float = 1.2,
    eps: float = 0.005,
) -> np.ndarray:
    """Sample the surface of a shape given only its distance function.

    Rejection sampling alone covers the surface unevenly: a band narrow enough to be accurate
    is hit so rarely that most shapes need far more candidates than can be afforded, and a band
    wide enough to be hit reliably smears the sample around edges and corners. The routine
    therefore accepts a wide band and then pushes every accepted point onto the surface with a
    single Newton step along the gradient of the distance. Where the gradient points diagonally
    outward, which is exactly at an edge or a corner, the step lands there, so those features
    are represented instead of being rounded away. This matters more here than it would
    elsewhere, because the encoder pools its points with a maximum and a feature never sampled
    is a feature it cannot learn.

    Args:
        sdf_fn: The distance function of the shape, mapping ``(N, 3)`` to ``(N,)``.
        n_surface: Number of points to return.
        n_candidates: Number of uniform candidates drawn before the band is applied.
        broad_threshold: Half width of the band a candidate must fall in to be kept.
        threshold: Distance a point must be within after the step to be accepted.
        bounds: The cube ``[-bounds, bounds]`` the candidates are drawn from.
        eps: Step of the finite difference the gradient is estimated with.

    Returns:
        Surface points, shape ``(n_surface, 3)``. The count is always met: if too few points
        survive they are resampled with a small amount of noise, and if the band caught nothing
        at all the candidates are returned unprojected.
    """
    candidates: np.ndarray = np.random.uniform(
        -bounds, bounds, (n_candidates, 3)
    ).astype(np.float32)
    sdf_vals: np.ndarray = sdf_fn(candidates)
    broad_mask: np.ndarray = np.abs(sdf_vals) < broad_threshold
    near_pts: np.ndarray = candidates[broad_mask]
    near_sdf: np.ndarray = sdf_vals[broad_mask]

    if len(near_pts) == 0:
        return np.random.uniform(-bounds, bounds, (n_surface, 3)).astype(np.float32)

    # Central differences rather than an analytic gradient: the distance functions are built by
    # combining primitives, so a closed-form gradient would have to be written per shape.
    grad: np.ndarray = np.zeros_like(near_pts)
    for axis in range(3):
        offset: np.ndarray = np.zeros(3, dtype=np.float32)
        offset[axis] = eps
        sdf_plus: np.ndarray = sdf_fn(near_pts + offset)
        sdf_minus: np.ndarray = sdf_fn(near_pts - offset)
        grad[:, axis] = (sdf_plus - sdf_minus) / (2.0 * eps)

    # Dividing by the squared gradient instead of its norm: for a true distance function the
    # two agree, but at an edge or a seam between primitives the gradient is shorter than unity
    # and the squared form is the step that still lands on the surface.
    grad_norm_sq: np.ndarray = np.sum(grad**2, axis=1, keepdims=True) + 1e-8

    projected: np.ndarray = near_pts - near_sdf[:, np.newaxis] * grad / grad_norm_sq
    projected = np.clip(projected, -bounds, bounds)

    proj_sdf: np.ndarray = sdf_fn(projected)
    proj_mask: np.ndarray = np.abs(proj_sdf) < threshold
    surface_pts: np.ndarray = projected[proj_mask]

    if len(surface_pts) >= n_surface:
        idx: np.ndarray = np.random.choice(len(surface_pts), n_surface, replace=False)
        return surface_pts[idx].astype(np.float32)
    elif len(surface_pts) > 0:
        # Draw with replacement and jitter, rather than returning fewer points: the encoder is
        # fed fixed-size batches, and exact duplicates would contribute nothing to the pooling.
        idx: np.ndarray = np.random.choice(len(surface_pts), n_surface, replace=True)
        padded: np.ndarray = surface_pts[idx].copy()
        padded += np.random.normal(0, 0.005, padded.shape).astype(np.float32)
        return padded.astype(np.float32)
    else:
        idx: np.ndarray = np.random.choice(
            len(near_pts), n_surface, replace=len(near_pts) < n_surface
        )
        return near_pts[idx].astype(np.float32)


def sample_query_points(
    sdf_fn: Callable[[np.ndarray], np.ndarray],
    surface_points: np.ndarray,
    n_query: int = 5000,
    bounds: float = 1.2,
    near_surface_fraction: float = 0.7,
    fine_noise_std: float = 0.01,
    coarse_noise_std: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    """Draw the points the decoder is supervised at, concentrated near the surface.

    What the reconstructed shape looks like is decided entirely by where the decoded field
    changes sign, so accuracy far from the surface is worth little. The queries are drawn in
    three groups: a narrow band and a wider band around the sampled surface, which teach the
    field where the surface is and how it falls away, and a uniform group over the whole cube,
    which keeps the decoder from placing surfaces in regions it has never been asked about.

    Args:
        surface_points: The sampled surface, shape ``(N_surface, 3)``, which the two bands
            are perturbations of.
        n_query: Total number of query points.
        bounds: The cube ``[-bounds, bounds]`` the uniform group is drawn from and every point
            is clipped to.
        near_surface_fraction: Share of the queries taken from the two bands, split evenly
            between them.
        fine_noise_std: Width of the narrow band.
        coarse_noise_std: Width of the wider band.

    Returns:
        The query points, shape ``(n_query, 3)``, and the signed distance at each, shape
        ``(n_query,)``. The three groups are shuffled together, so a slice of the result is not
        a slice of one group.
    """
    n_near_surface: int = int(n_query * near_surface_fraction)
    n_uniform: int = n_query - n_near_surface

    n_fine: int = n_near_surface // 2
    n_coarse: int = n_near_surface - n_fine

    fine_idx: np.ndarray = np.random.choice(
        len(surface_points), n_fine, replace=len(surface_points) < n_fine
    )
    coarse_idx: np.ndarray = np.random.choice(
        len(surface_points), n_coarse, replace=len(surface_points) < n_coarse
    )

    fine_points: np.ndarray = surface_points[fine_idx].copy()
    fine_points += np.random.normal(0, fine_noise_std, fine_points.shape).astype(
        np.float32
    )

    coarse_points: np.ndarray = surface_points[coarse_idx].copy()
    coarse_points += np.random.normal(0, coarse_noise_std, coarse_points.shape).astype(
        np.float32
    )

    uniform_points: np.ndarray = np.random.uniform(
        -bounds, bounds, (n_uniform, 3)
    ).astype(np.float32)

    query_points: np.ndarray = np.concatenate(
        [fine_points, coarse_points, uniform_points], axis=0
    )
    query_points = np.clip(query_points, -bounds, bounds).astype(np.float32)

    # Shuffled before the distances are evaluated, so that any consumer taking a prefix of the
    # queries still gets all three groups.
    perm: np.ndarray = np.random.permutation(len(query_points))
    query_points = query_points[perm]
    sdf_values: np.ndarray = sdf_fn(query_points).astype(np.float32)

    return query_points, sdf_values


def sample_box_surface_points(
    half: np.ndarray,
    n_surface: int = 1000,
    edge_fraction: float = 0.25,
    corner_fraction: float = 0.05,
) -> np.ndarray:
    """Sample the surface of a box, with edges and corners deliberately over-represented.

    The box is the one shape whose surface is sampled without going through
    :func:`find_surface_points`, because its faces, edges and corners can be enumerated. Edges
    and corners cover no area at all next to the faces, so any sampler that is faithful to area
    would omit them; here they receive a fixed share instead. Within each of the three groups
    the weighting is by area and by length, so a thin plate puts most of its face points on its
    two large faces and most of its edge points on its long edges.

    Args:
        half: Half extents, shape ``(3,)``.
        n_surface: Number of points to return.
        edge_fraction: Share of the points placed on the twelve edges.
        corner_fraction: Share placed at the eight corners.

    Returns:
        Surface points, shape ``(n_surface, 3)``, in random order.
    """
    hx, hy, hz = float(half[0]), float(half[1]), float(half[2])
    rng = np.random

    face_fraction: float = 1.0 - edge_fraction - corner_fraction
    n_faces: int = int(n_surface * face_fraction)
    n_edges: int = int(n_surface * edge_fraction)
    # The corner count takes the remainder, so the three groups always sum to n_surface.
    n_corners: int = n_surface - n_faces - n_edges

    face_areas: np.ndarray = np.array(
        [
            4 * hy * hz,
            4 * hy * hz,  # ±X
            4 * hx * hz,
            4 * hx * hz,  # ±Y
            4 * hx * hy,
            4 * hx * hy,
        ],  # ±Z
        dtype=np.float32,
    )
    face_probs: np.ndarray = face_areas / face_areas.sum()
    face_ids: np.ndarray = rng.choice(6, size=n_faces, p=face_probs)

    face_pts_list: list[np.ndarray] = []
    for fid in range(6):
        mask = face_ids == fid
        n = int(mask.sum())
        if n == 0:
            continue
        u: np.ndarray = rng.uniform(-1, 1, (n, 2)).astype(np.float32)
        if fid == 0:  # +X face
            pts = np.stack([np.full(n, +hx), u[:, 0] * hy, u[:, 1] * hz], axis=1)
        elif fid == 1:  # -X face
            pts = np.stack([np.full(n, -hx), u[:, 0] * hy, u[:, 1] * hz], axis=1)
        elif fid == 2:  # +Y face
            pts = np.stack([u[:, 0] * hx, np.full(n, +hy), u[:, 1] * hz], axis=1)
        elif fid == 3:  # -Y face
            pts = np.stack([u[:, 0] * hx, np.full(n, -hy), u[:, 1] * hz], axis=1)
        elif fid == 4:  # +Z face
            pts = np.stack([u[:, 0] * hx, u[:, 1] * hy, np.full(n, +hz)], axis=1)
        else:  # -Z face
            pts = np.stack([u[:, 0] * hx, u[:, 1] * hy, np.full(n, -hz)], axis=1)
        face_pts_list.append(pts)
    face_pts: np.ndarray = (
        np.concatenate(face_pts_list, axis=0)
        if face_pts_list
        else np.empty((0, 3), dtype=np.float32)
    )

    edge_lengths: np.ndarray = np.array(
        [2 * hx] * 4 + [2 * hy] * 4 + [2 * hz] * 4, dtype=np.float32
    )
    edge_probs: np.ndarray = edge_lengths / edge_lengths.sum()
    edge_ids: np.ndarray = rng.choice(12, size=n_edges, p=edge_probs)

    # One entry per edge, as (index of the axis the edge runs along, value on the first of the
    # two remaining axes, value on the second). The order of the two remaining axes is
    # ascending, which is what the assignment below relies on.
    edge_defs: list[tuple] = [
        (0, +hy, +hz),
        (0, +hy, -hz),
        (0, -hy, +hz),
        (0, -hy, -hz),
        (1, +hx, +hz),
        (1, +hx, -hz),
        (1, -hx, +hz),
        (1, -hx, -hz),
        (2, +hx, +hy),
        (2, +hx, -hy),
        (2, -hx, +hy),
        (2, -hx, -hy),
    ]
    half_by_axis: np.ndarray = np.array([hx, hy, hz], dtype=np.float32)

    edge_pts_list: list[np.ndarray] = []
    for eid in range(12):
        mask = edge_ids == eid
        n = int(mask.sum())
        if n == 0:
            continue
        free_axis, fixed_a, fixed_b = edge_defs[eid]
        t: np.ndarray = (
            rng.uniform(-1, 1, n).astype(np.float32) * half_by_axis[free_axis]
        )
        pts: np.ndarray = np.zeros((n, 3), dtype=np.float32)
        pts[:, free_axis] = t
        other = [ax for ax in range(3) if ax != free_axis]
        pts[:, other[0]] = fixed_a
        pts[:, other[1]] = fixed_b
        edge_pts_list.append(pts)
    edge_pts: np.ndarray = (
        np.concatenate(edge_pts_list, axis=0)
        if edge_pts_list
        else np.empty((0, 3), dtype=np.float32)
    )

    signs: np.ndarray = np.array(
        [[sx, sy, sz] for sx in (+1, -1) for sy in (+1, -1) for sz in (+1, -1)],
        dtype=np.float32,
    )
    corner_base: np.ndarray = signs * half

    # The corners are jittered rather than repeated exactly: identical points collapse into one
    # entry of the pooled feature and the group would then count for eight points, not its share.
    corner_idx: np.ndarray = rng.choice(8, size=n_corners, replace=True)
    corner_pts: np.ndarray = corner_base[corner_idx].copy()
    corner_pts += rng.normal(0, 0.003, corner_pts.shape).astype(np.float32)

    all_pts: np.ndarray = np.concatenate([face_pts, edge_pts, corner_pts], axis=0)

    perm: np.ndarray = rng.permutation(len(all_pts))
    return all_pts[perm].astype(np.float32)


# Range the characteristic dimensions of a shape are drawn from. These are dimensionless: the
# autoencoder only ever sees shapes at roughly unit scale, so what has to vary is the aspect
# ratio and not the size. They are deliberately unrelated to the metre range under
# config.data, which belongs to the box pipeline of the earlier parts.
SHAPE_SAMPLE_MIN = 0.05
SHAPE_SAMPLE_MAX = 0.95

# Largest extent a shape is scaled to in the canonical frame. It has to agree with the value
# used at inference in validate_shape_encoder.to_canonical, otherwise the encoder is queried in
# a frame it was never trained in.
CANON_EXTENT = 0.9

# The vocabulary and how often each member is drawn. It covers two families, the prismatic
# shapes and the curved ones, so that the surrogate downstream is confronted with forms whose
# contact behaviour genuinely differs. Shapes with a hole through them are left out: neither
# the encoder nor the compressed code represents them reliably. The weights are not uniform;
# the thin-walled profiles are the hardest members and are drawn more often than the simple
# ones. They are normalised where they are used, so they need not sum to one.
SHAPE_TYPES = [
    "box", "rounded_box", "cylinder", "half_cylinder", "sphere", "ellipsoid",
    "capsule", "cone", "hex_prism", "wedge", "iprofile", "uprofile",
    "lprofile", "tprofile",
]
SHAPE_WEIGHTS = [
    0.08, 0.06, 0.07, 0.07, 0.05, 0.07, 0.07, 0.08, 0.07, 0.08, 0.09, 0.09,
    0.07, 0.05,
]


def generate_random_shape(
    n_surface: int = 1000,
    n_query: int = 5000,
    bounds: float = 1.2,
    shape_type: str | None = None,
    canonical: bool = False,
    slender: bool = False,
) -> ShapeData:
    """Draw one shape from the vocabulary, centred on the origin.

    Every characteristic dimension is drawn log-uniformly, which makes a slender shape as
    likely as a compact one rather than merely possible. The box has its surface sampled by
    :func:`sample_box_surface_points`; every other member goes through
    :func:`find_surface_points`.

    Args:
        shape_type: Which member to draw. ``None`` draws one at random, weighted by
            ``SHAPE_WEIGHTS``.
        canonical: Return the shape in the canonical frame, centred on the centroid of its
            sampled surface and scaled to a largest extent of ``CANON_EXTENT``. The query
            points and the distances are transformed with it, and so is the distance function,
            so the shape stays consistent.
        slender: Draw a long axis first and the cross-section as a fraction of it, instead of
            drawing every dimension independently. This narrows the distribution towards beams
            and plates. The autoencoder is trained on the wider default, which contains these.

    Returns:
        The shape, with ``half_extents`` set for a box and ``None`` otherwise: only the box has
        an encoder that reads its dimensions directly.
    """
    from src.shape_primitives import (  # lazy import: avoid a circular import
        cylinder_x_sdf,
        lprofile_sdf,
        tprofile_sdf,
        rounded_box_sdf,
        half_cylinder_x_sdf,
        capsule_x_sdf,
        cone_x_sdf,
        hex_prism_x_sdf,
        wedge_x_sdf,
        i_profile_sdf,
        u_profile_sdf,
    )

    def lu() -> float:
        """One characteristic dimension, log-uniform over the sampling range."""
        return float(np.exp(np.random.uniform(np.log(SHAPE_SAMPLE_MIN), np.log(SHAPE_SAMPLE_MAX))))

    def frac() -> float:
        """A cross-section side as a fraction of the long side, for the slender variant."""
        return float(np.exp(np.random.uniform(np.log(0.12), np.log(0.45))))

    if shape_type is None:
        _p = np.asarray(SHAPE_WEIGHTS, dtype=float)
        shape_type = str(np.random.choice(SHAPE_TYPES, p=_p / _p.sum()))

    half_extents: np.ndarray | None = None

    if shape_type == "box":
        if slender:
            long = lu()
            half = long * np.array([1.0, frac(), frac()], dtype=np.float32)
            # Shuffled so the long axis is not always X, which would let the encoder infer the
            # aspect from the axis rather than from the geometry.
            np.random.shuffle(half)
        else:
            half = np.array([lu(), lu(), lu()], dtype=np.float32)
        sdf_fn = lambda p, half=half: box_sdf(p, half)  # noqa: E731
        surface_points = sample_box_surface_points(half, n_surface=n_surface)
        half_extents = half
    elif shape_type == "cylinder":
        # Full dimensions are twice a drawn value, so that the half extents of every non-box
        # member land on the same scale as the half extents of a box.
        length = 2.0 * lu()
        diameter = length * frac() if slender else 2.0 * lu()
        sdf_fn = lambda p, L=length, d=diameter: cylinder_x_sdf(p, L, d)  # noqa: E731
        surface_points = find_surface_points(sdf_fn, n_surface=n_surface, bounds=bounds)
    elif shape_type == "lprofile":
        # The wall is a fraction of the smaller cross-section side, so the profile stays a
        # profile: an absolute thickness would fill the section in on the small instances.
        length = 2.0 * lu()
        sy, sz = (length * frac(), length * frac()) if slender else (2.0 * lu(), 2.0 * lu())
        t = min(sy, sz) * np.random.uniform(0.12, 0.30)
        sdf_fn = lambda p, L=length, sy=sy, sz=sz, t=t: lprofile_sdf(p, L, sy, sz, t)  # noqa: E731
        surface_points = find_surface_points(sdf_fn, n_surface=n_surface, bounds=bounds)
    elif shape_type == "tprofile":
        length = 2.0 * lu()
        sy, sz = (length * frac(), length * frac()) if slender else (2.0 * lu(), 2.0 * lu())
        t = min(sy, sz) * np.random.uniform(0.12, 0.30)
        sdf_fn = lambda p, L=length, sy=sy, sz=sz, t=t: tprofile_sdf(p, L, sy, sz, t)  # noqa: E731
        surface_points = find_surface_points(sdf_fn, n_surface=n_surface, bounds=bounds)
    elif shape_type == "rounded_box":
        if slender:
            long = lu()
            half = long * np.array([1.0, frac(), frac()], dtype=np.float32)
            np.random.shuffle(half)
        else:
            half = np.array([lu(), lu(), lu()], dtype=np.float32)
        r = float(min(half) * np.random.uniform(0.15, 0.45))
        sdf_fn = lambda p, h=half, r=r: rounded_box_sdf(p, h, r)  # noqa: E731
        surface_points = find_surface_points(sdf_fn, n_surface=n_surface, bounds=bounds)
    elif shape_type == "half_cylinder":
        length = 2.0 * lu()
        diameter = length * frac() if slender else 2.0 * lu()
        sdf_fn = lambda p, L=length, d=diameter: half_cylinder_x_sdf(p, L, d)  # noqa: E731
        surface_points = find_surface_points(sdf_fn, n_surface=n_surface, bounds=bounds)
    elif shape_type == "sphere":
        radius = lu()
        sdf_fn = lambda p, r=radius: sphere_sdf(p, r)  # noqa: E731
        surface_points = find_surface_points(sdf_fn, n_surface=n_surface, bounds=bounds)
    elif shape_type == "ellipsoid":
        if slender:
            long = lu()
            radii = long * np.array([1.0, frac(), frac()], dtype=np.float32)
            np.random.shuffle(radii)
        else:
            radii = np.array([lu(), lu(), lu()], dtype=np.float32)
        sdf_fn = lambda p, rr=radii: ellipsoid_sdf(p, rr)  # noqa: E731
        surface_points = find_surface_points(sdf_fn, n_surface=n_surface, bounds=bounds)
    elif shape_type == "capsule":
        # The caps count towards the extent, so the segment is what is left of the drawn total
        # after the diameter. Without this the capsule would be systematically longer than the
        # other members and could leave the sampling cube.
        total = 2.0 * lu()
        diameter = total * (frac() if slender else float(np.random.uniform(0.3, 0.8)))
        segment = max(total - diameter, 1e-3)
        sdf_fn = lambda p, L=segment, d=diameter: capsule_x_sdf(p, L, d)  # noqa: E731
        surface_points = find_surface_points(sdf_fn, n_surface=n_surface, bounds=bounds)
    elif shape_type == "cone":
        length = 2.0 * lu()
        d0 = length * frac() if slender else 2.0 * lu()
        d1 = d0 * float(np.random.uniform(0.2, 0.75))
        sdf_fn = lambda p, L=length, a=d0, b=d1: cone_x_sdf(p, L, a, b)  # noqa: E731
        surface_points = find_surface_points(sdf_fn, n_surface=n_surface, bounds=bounds)
    elif shape_type == "hex_prism":
        length = 2.0 * lu()
        apothem = length * frac() if slender else lu()
        sdf_fn = lambda p, L=length, a=apothem: hex_prism_x_sdf(p, L, a)  # noqa: E731
        surface_points = find_surface_points(sdf_fn, n_surface=n_surface, bounds=bounds)
    elif shape_type == "wedge":
        length = 2.0 * lu()
        sy, sz = (length * frac(), length * frac()) if slender else (2.0 * lu(), 2.0 * lu())
        sdf_fn = lambda p, L=length, a=sy, b=sz: wedge_x_sdf(p, L, a, b)  # noqa: E731
        surface_points = find_surface_points(sdf_fn, n_surface=n_surface, bounds=bounds)
    elif shape_type == "iprofile":
        length = 2.0 * lu()
        sy, sz = (length * frac(), length * frac()) if slender else (2.0 * lu(), 2.0 * lu())
        # Capped as well as scaled: past this fraction the two flanges and the web meet and the
        # section is a filled rectangle, which is a box and not an I-beam.
        t = float(np.clip(min(sy, sz) * np.random.uniform(0.12, 0.25), 1e-3, 0.45 * min(sy, sz)))
        sdf_fn = lambda p, L=length, a=sy, b=sz, tt=t: i_profile_sdf(p, L, a, b, tt)  # noqa: E731
        surface_points = find_surface_points(sdf_fn, n_surface=n_surface, bounds=bounds)
    elif shape_type == "uprofile":
        length = 2.0 * lu()
        sy, sz = (length * frac(), length * frac()) if slender else (2.0 * lu(), 2.0 * lu())
        t = float(np.clip(min(sy, sz) * np.random.uniform(0.12, 0.25), 1e-3, 0.40 * min(sy, sz)))
        sdf_fn = lambda p, L=length, a=sy, b=sz, tt=t: u_profile_sdf(p, L, a, b, tt)  # noqa: E731
        surface_points = find_surface_points(sdf_fn, n_surface=n_surface, bounds=bounds)
    else:
        raise ValueError(f"unknown shape_type: {shape_type}")

    query_points, sdf_values = sample_query_points(
        sdf_fn=sdf_fn, surface_points=surface_points, n_query=n_query, bounds=bounds,
    )

    if canonical:
        # The centre is the centroid of the sampled surface, not of the bounding box, because
        # that is what the inference-time normalisation uses and the two frames have to agree.
        # A signed distance is a length, so it is scaled by the same factor as the coordinates;
        # the half extents and the distance function are transformed with it, so the shape
        # remains consistent in the new frame.
        centre = surface_points.mean(axis=0)
        centred = surface_points - centre
        scale = CANON_EXTENT / (float(np.ptp(centred, axis=0).max()) + 1e-12)
        surface_points = (centred * scale).astype(np.float32)
        query_points = ((query_points - centre) * scale).astype(np.float32)
        sdf_values = (sdf_values * scale).astype(np.float32)
        if half_extents is not None:
            half_extents = (half_extents * scale).astype(np.float32)
        sdf_fn = lambda p, _f=sdf_fn, _c=centre, _s=scale: _f(p / _s + _c) * _s  # noqa: E731

    return ShapeData(
        surface_points=surface_points,
        query_points=query_points,
        sdf_values=sdf_values,
        shape_type=shape_type,
        half_extents=half_extents,
        sdf_fn=sdf_fn,
    )


class AutoEncoderDataset(torch.utils.data.Dataset):
    """The shapes of one training run, generated once and held in memory.

    Generation is done in the constructor rather than per item, so the same shapes are seen in
    every epoch and a run is reproducible from the seed. The cost is memory: it grows with the
    number of shapes times the points per shape, and both come from the configuration.
    """

    def __init__(
        self,
        n_shapes: int = config.autoencoder.n_shapes,
        n_surface: int = config.autoencoder.n_surface,
        n_query: int = config.autoencoder.n_query,
        canonical: bool = False,
    ):
        """Generate every shape of the run.

        Args:
            n_surface: Surface points per shape, the encoder input.
            n_query: Query points per shape, the decoder target.
            canonical: Generate the shapes in the canonical frame; see
                :func:`generate_random_shape`.
        """
        self.n_shapes: int = n_shapes
        self.n_surface: int = n_surface
        self.n_query: int = n_query

        # Pre-generate all shapes
        self.surface_points: list[np.ndarray] = []
        self.query_points: list[np.ndarray] = []
        self.sdf_values: list[np.ndarray] = []
        self.shape_types: list[str] = []
        self.half_extents: list[np.ndarray] = []

        print(f"Generating {n_shapes} random shapes...")
        for i in range(n_shapes):
            shape: ShapeData = generate_random_shape(
                n_surface=n_surface, n_query=n_query, canonical=canonical
            )
            self.surface_points.append(shape.surface_points)
            self.query_points.append(shape.query_points)
            self.sdf_values.append(shape.sdf_values)
            self.shape_types.append(shape.shape_type)
            self.half_extents.append(shape.half_extents)

            if (i + 1) % 10000 == 0:
                print(f"  Generated {i + 1}/{n_shapes} shapes")

        # The realised distribution is reported because it is drawn, not enforced: it will only
        # approach SHAPE_WEIGHTS, and on a small run it can be visibly off.
        type_counts: dict = Counter(self.shape_types)
        print("\nShape type distribution:")
        for stype, count in sorted(type_counts.items()):
            print(f"  {stype}: {count} ({count / n_shapes * 100:.1f}%)")

    def __len__(self) -> int:
        """One item per shape, not per query point."""
        return self.n_shapes

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return one shape as tensors.

        Returns:
            The surface points, shape ``(n_surface, 3)``; the query points, shape
            ``(n_query, 3)``; and the signed distances at them, shape ``(n_query,)``. The shape
            type and the half extents are not returned here; they are read off the attributes
            by whoever needs them.
        """
        surface: torch.Tensor = torch.from_numpy(self.surface_points[idx])
        query: torch.Tensor = torch.from_numpy(self.query_points[idx])
        sdf: torch.Tensor = torch.from_numpy(self.sdf_values[idx])
        return surface, query, sdf


def create_dataloaders(
    n_shapes: int = config.autoencoder.n_shapes,
    n_surface: int = config.autoencoder.n_surface,
    n_query: int = config.autoencoder.n_query,
    train_split: float = config.training.train_split,
    val_split: float = config.training.val_split,
    batch_size: int = config.autoencoder.batch_size,
) -> tuple[
    torch.utils.data.DataLoader,
    torch.utils.data.DataLoader,
    torch.utils.data.DataLoader,
]:
    """Generate the shapes and split them into loaders for training, validation and test.

    The split is unseeded and the shapes are generated in the default frame, so this is not the
    path the two-stage training takes; that one builds its own dataset and splits it with a
    fixed generator, because the stages have to agree on which shape carries which index.

    Returns:
        The loaders for training, validation and test. The test share is whatever the other two
        leave over.
    """
    dataset: AutoEncoderDataset = AutoEncoderDataset(
        n_shapes=n_shapes, n_surface=n_surface, n_query=n_query
    )

    n_train: int = int(train_split * n_shapes)
    n_val: int = int(val_split * n_shapes)
    n_test: int = n_shapes - n_train - n_val

    train_ds, val_ds, test_ds = torch.utils.data.random_split(
        dataset, [n_train, n_val, n_test]
    )

    print(f"\nSplit: {n_train} train, {n_val} val, {n_test} test")

    train_loader: torch.utils.data.DataLoader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True
    )
    val_loader: torch.utils.data.DataLoader = torch.utils.data.DataLoader(
        val_ds, batch_size=batch_size
    )
    test_loader: torch.utils.data.DataLoader = torch.utils.data.DataLoader(
        test_ds, batch_size=batch_size
    )

    return train_loader, val_loader, test_loader
