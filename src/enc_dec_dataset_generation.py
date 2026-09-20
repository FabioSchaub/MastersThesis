"""Shapes and signed-distance samples that stage 1 of the autoencoder training is fitted on.

Stage 1 learns one free latent code per shape by asking the auxiliary signed-distance decoder
of ``src/dec_sdf.py`` to reproduce that shape's field from the code. This module supplies what
that needs: a random axis-aligned box, a set of query points around it, and the exact distance
at each. The distances are analytical rather than measured, so the targets carry no sampling
error and any error at the end of stage 1 belongs to the code or the decoder.

Every shape is a box, and its three half-extents are drawn log-uniformly between
``config.data.sampling_min`` and ``config.data.sampling_max`` metres per axis. Drawing in the
logarithm rather than uniformly is what gives slender and near-cubic proportions comparable
weight, instead of filling the set with shapes near the top of the range.

Nothing is written to disk: shapes are generated into memory and consumed by
``src/enc_dec_training.py``. The decoder involved here is the auxiliary one only; the decoder
that maps a code back to three half-extents is a separate network trained afterwards.
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


# Of the primitives and the boolean operations below, only box_sdf is reached on this branch:
# the vocabulary here is boxes and nothing else. The rest are the machinery a richer vocabulary
# would need and are left in place unused.


def box_sdf(points: np.ndarray, half_extents: np.ndarray) -> np.ndarray:
    """Signed distance from points to an axis-aligned box centred at the origin.

    Args:
        points: Query coordinates, shape ``(N, 3)``.
        half_extents: Half side lengths, shape ``(3,)``, same units as the points.

    Returns:
        Signed distances of shape ``(N,)``, negative inside the box and positive outside.
    """
    q: np.ndarray = np.abs(points) - half_extents
    outside: np.ndarray = np.linalg.norm(np.maximum(q, 0), axis=1)
    inside: np.ndarray = np.minimum(np.max(q, axis=1), 0)
    return outside + inside


def sphere_sdf(points: np.ndarray, radius: float) -> np.ndarray:
    """Signed distance from points ``(N, 3)`` to a sphere centred at the origin."""
    return np.linalg.norm(points, axis=1) - radius


def cylinder_sdf(points: np.ndarray, radius: float, height: float) -> np.ndarray:
    """Signed distance from points ``(N, 3)`` to a cylinder centred at the origin.

    The axis of revolution is y, so ``radius`` is measured in the xz plane and ``height`` along
    y.
    """
    d_xz: np.ndarray = np.linalg.norm(points[:, [0, 2]], axis=1) - radius
    d_y: np.ndarray = np.abs(points[:, 1]) - height / 2
    outside: np.ndarray = np.linalg.norm(
        np.maximum(np.stack([d_xz, d_y], axis=1), 0), axis=1
    )
    inside: np.ndarray = np.minimum(np.maximum(d_xz, d_y), 0)
    return outside + inside


def capsule_sdf(points: np.ndarray, radius: float, height: float) -> np.ndarray:
    """Signed distance from points ``(N, 3)`` to a capsule aligned with y.

    ``height`` is the separation of the two cap centres, so the full extent along y is
    ``height + 2 * radius``.
    """
    p: np.ndarray = points.copy()
    p[:, 1] = p[:, 1] - np.clip(p[:, 1], -height / 2, height / 2)
    return np.linalg.norm(p, axis=1) - radius


def torus_sdf(
    points: np.ndarray, major_radius: float, minor_radius: float
) -> np.ndarray:
    """Signed distance from points ``(N, 3)`` to a torus whose ring lies in the xz plane.

    ``major_radius`` reaches from the centre to the middle of the tube and ``minor_radius`` is
    the tube itself.
    """
    q: np.ndarray = np.stack(
        [np.linalg.norm(points[:, [0, 2]], axis=1) - major_radius, points[:, 1]],
        axis=1,
    )
    return np.linalg.norm(q, axis=1) - minor_radius


def ellipsoid_sdf(points: np.ndarray, radii: np.ndarray) -> np.ndarray:
    """Approximate signed distance from points ``(N, 3)`` to an ellipsoid with semi-axes
    ``radii`` ``(3,)``.

    Only an approximation: an ellipsoid has no closed-form signed distance, so the implicit
    value is divided by the norm of its gradient to bring it close to one.
    """
    k: np.ndarray = np.linalg.norm(points / radii, axis=1)
    grad_norm: np.ndarray = np.linalg.norm(points / (radii**2), axis=1) + 1e-8
    return k * (k - 1) / grad_norm


def union_sdf(sdf_a: np.ndarray, sdf_b: np.ndarray) -> np.ndarray:
    """Combine two ``(N,)`` fields into the region inside either shape."""
    return np.minimum(sdf_a, sdf_b)


def difference_sdf(sdf_a: np.ndarray, sdf_b: np.ndarray) -> np.ndarray:
    """Combine two ``(N,)`` fields into the first shape with the second cut out of it."""
    return np.maximum(sdf_a, -sdf_b)


def intersection_sdf(sdf_a: np.ndarray, sdf_b: np.ndarray) -> np.ndarray:
    """Combine two ``(N,)`` fields into the region inside both shapes."""
    return np.maximum(sdf_a, sdf_b)


@dataclass
class ShapeData:
    """One generated shape: points on its surface, points around it, and its distances there.

    ``sdf_values`` is aligned with ``query_points`` entry by entry. ``half_extents`` is
    populated for boxes and left empty otherwise, and ``sdf_fn`` keeps the exact field callable
    so a shape can be re-evaluated at new points after it has been generated.

    Attributes:
        surface_points: Points on the surface, shape ``(n_surface, 3)``.
        query_points: Points around the shape, shape ``(n_query, 3)``.
        sdf_values: Signed distance at each query point, shape ``(n_query,)``.
        shape_type: Name of the primitive, ``"box"`` for everything generated here.
        half_extents: Half side lengths in metres, shape ``(3,)``.
        sdf_fn: The analytical field, mapping ``(N, 3)`` coordinates to ``(N,)`` distances.
    """

    surface_points: np.ndarray
    query_points: np.ndarray
    sdf_values: np.ndarray
    shape_type: str
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
    """Find points on the zero level set of an arbitrary field, by projection.

    Candidates are drawn in a band around the surface and then pushed onto it with a single
    Newton step along the gradient. Drawing points and keeping only those that land on the
    surface would never work: the surface has no volume, so no random point lies on it.

    This is the general route, for a field given only as a callable. It is not used for the
    boxes generated here, which have their surface sampled directly by
    :func:`sample_box_surface_points`.

    Args:
        sdf_fn: The field, mapping ``(N, 3)`` coordinates to ``(N,)`` distances.
        n_surface: Number of points to return.
        n_candidates: Points drawn before the band is applied.
        broad_threshold: Half-width of the band a candidate must fall in to be projected.
        threshold: Largest distance still accepted after projection.
        bounds: Half-width of the cube candidates are drawn in.
        eps: Step of the finite difference the gradient is estimated with.

    Returns:
        Surface points of shape ``(n_surface, 3)``. Padded by repetition with small noise if
        too few points survived, and filled with unprojected candidates if none did.
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

    # Central differences rather than one-sided: the field is queried on both sides of each
    # axis, which costs six evaluations instead of three but stays accurate at the creases
    # where the gradient turns.
    grad: np.ndarray = np.zeros_like(near_pts)
    for axis in range(3):
        offset: np.ndarray = np.zeros(3, dtype=np.float32)
        offset[axis] = eps
        sdf_plus: np.ndarray = sdf_fn(near_pts + offset)
        sdf_minus: np.ndarray = sdf_fn(near_pts - offset)
        grad[:, axis] = (sdf_plus - sdf_minus) / (2.0 * eps)

    # A true signed distance has unit gradient, so dividing by its square is a no-op there. It
    # matters at edges and corners, where the gradient is not unit and an undivided step would
    # overshoot. The small addition keeps a vanishing gradient from dividing by zero.
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
        # Repeating with noise rather than returning a short array, so the caller always gets
        # the count it asked for and batches stay rectangular.
        idx: np.ndarray = np.random.choice(len(surface_pts), n_surface, replace=True)
        padded: np.ndarray = surface_pts[idx].copy()
        padded += np.random.normal(0, 0.005, padded.shape).astype(np.float32)
        return padded.astype(np.float32)
    else:
        # Nothing survived the projection. Returning the unprojected band is worse than the
        # surface but better than an empty array, which would stop the generation.
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
    """Draw the points stage 1 supervises the auxiliary decoder at, biased towards the surface.

    What the shape is, is decided at the zero level set; far from it the field only says which
    side one is on. Most of the budget therefore goes to points scattered around the surface,
    in two bands of different width so both the immediate neighbourhood and a wider shell are
    covered, and the remainder fills the volume uniformly to keep inside and outside separated
    globally.

    Args:
        sdf_fn: The exact field, mapping ``(N, 3)`` coordinates to ``(N,)`` distances.
        surface_points: Points on the surface, shape ``(n_surface, 3)``, used as the seeds the
            near-surface queries are scattered around.
        n_query: Total number of queries returned.
        bounds: Half-width of the cube the uniform queries fill and every query is clipped to.
        near_surface_fraction: Share of the budget spent near the surface, the rest uniform.
        fine_noise_std, coarse_noise_std: Widths of the two bands, in the same units as the
            coordinates. The near-surface half of the budget is split evenly between them.

    Returns:
        A tuple ``(query_points, sdf_values)`` of shapes ``(n_query, 3)`` and ``(n_query,)``,
        aligned entry by entry.
    """
    # The uniform share takes the remainder rather than being computed independently, so the
    # three parts always add up to exactly n_query however the fraction rounds.
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

    # Shuffled before the field is evaluated, so the three groups are interleaved. Left in
    # order, a batch drawn from the front of the array would contain only near-surface points.
    perm: np.ndarray = np.random.permutation(len(query_points))
    query_points = query_points[perm]
    # Evaluated after the clipping, so every distance belongs to the coordinate stored beside
    # it rather than to where the point was before it was clipped.
    sdf_values: np.ndarray = sdf_fn(query_points).astype(np.float32)

    return query_points, sdf_values


def sample_box_surface_points(
    half: np.ndarray,
    n_surface: int = 1000,
    edge_fraction: float = 0.25,
    corner_fraction: float = 0.05,
) -> np.ndarray:
    """Sample the surface of a box directly, giving its edges and corners their own budget.

    A box is built from parts of different dimension, and sampling the surface uniformly would
    reach only the faces: the edges are lines and the corners are single points, so neither has
    any area to be hit. They are what makes a box a box, so each is allotted a share of the
    budget and placed exactly.

    Faces are weighted by area and edges by length, so a slender box puts its points where its
    surface actually is rather than spreading them evenly over the six faces.

    Args:
        half: Half side lengths, shape ``(3,)``.
        n_surface: Total number of points returned.
        edge_fraction, corner_fraction: Shares of the budget for the twelve edges and the eight
            corners; the faces receive whatever is left.

    Returns:
        Surface points of shape ``(n_surface, 3)``, in random order.
    """
    hx, hy, hz = float(half[0]), float(half[1]), float(half[2])
    rng = np.random

    face_fraction: float = 1.0 - edge_fraction - corner_fraction
    n_faces: int = int(n_surface * face_fraction)
    n_edges: int = int(n_surface * edge_fraction)
    # The corner count takes the remainder, so the three groups sum to n_surface exactly
    # whichever way the two fractions round.
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

    # The face identifiers run +X, -X, +Y, -Y, +Z, -Z, matching the face codes used elsewhere.
    # One coordinate is pinned to the face and the other two range over it.
    face_pts_list: list[np.ndarray] = []
    for fid in range(6):
        mask = face_ids == fid
        n = int(mask.sum())
        if n == 0:
            continue
        u: np.ndarray = rng.uniform(-1, 1, (n, 2)).astype(np.float32)
        if fid == 0:
            pts = np.stack([np.full(n, +hx), u[:, 0] * hy, u[:, 1] * hz], axis=1)
        elif fid == 1:
            pts = np.stack([np.full(n, -hx), u[:, 0] * hy, u[:, 1] * hz], axis=1)
        elif fid == 2:
            pts = np.stack([u[:, 0] * hx, np.full(n, +hy), u[:, 1] * hz], axis=1)
        elif fid == 3:
            pts = np.stack([u[:, 0] * hx, np.full(n, -hy), u[:, 1] * hz], axis=1)
        elif fid == 4:
            pts = np.stack([u[:, 0] * hx, u[:, 1] * hy, np.full(n, +hz)], axis=1)
        else:
            pts = np.stack([u[:, 0] * hx, u[:, 1] * hy, np.full(n, -hz)], axis=1)
        face_pts_list.append(pts)
    face_pts: np.ndarray = (
        np.concatenate(face_pts_list, axis=0)
        if face_pts_list
        else np.empty((0, 3), dtype=np.float32)
    )

    # Four edges run parallel to each axis, and an edge parallel to an axis has that axis's
    # full edge length. The ordering here has to match edge_defs below.
    edge_lengths: np.ndarray = np.array(
        [2 * hx] * 4 + [2 * hy] * 4 + [2 * hz] * 4, dtype=np.float32
    )
    edge_probs: np.ndarray = edge_lengths / edge_lengths.sum()
    edge_ids: np.ndarray = rng.choice(12, size=n_edges, p=edge_probs)

    # Each entry is the axis the edge runs along followed by the two coordinates that are
    # pinned, given in increasing axis order for the two remaining axes. The four sign
    # combinations per axis are the four parallel edges.
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
        # Ascending axis order, which is the order the two pinned values are stored in.
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

    # There are only eight corners, so filling the budget means repeating them. The noise
    # spreads the copies out; without it the result would be eight points of high multiplicity
    # rather than a sampled neighbourhood.
    corner_idx: np.ndarray = rng.choice(8, size=n_corners, replace=True)
    corner_pts: np.ndarray = corner_base[corner_idx].copy()
    corner_pts += rng.normal(0, 0.003, corner_pts.shape).astype(np.float32)

    all_pts: np.ndarray = np.concatenate([face_pts, edge_pts, corner_pts], axis=0)

    # Shuffled so faces, edges and corners are interleaved rather than left in three blocks.
    perm: np.ndarray = rng.permutation(len(all_pts))
    return all_pts[perm].astype(np.float32)


def generate_random_shape(
    n_surface: int = 1000,
    n_query: int = 5000,
    bounds: float = 1.2,
) -> ShapeData:
    """Generate one random box, centred at the origin, with its surface and query samples.

    Every shape is an axis-aligned box and the three half-extents are drawn independently, so
    the set covers proportions rather than a family of similar blocks. Drawing them in the
    logarithm rather than uniformly is what makes a slender plate or a rod as likely as a
    near-cube; drawn uniformly, all three would concentrate near the top of the range and
    slender shapes would be rare.

    Both ends of the range and the aspect ratios that follow from it come from
    ``config.data.sampling_min`` and ``config.data.sampling_max``, in metres. The shapes are
    kept at their real size here and are not normalised.

    Args:
        n_surface: Points sampled on the surface.
        n_query: Points sampled around the shape, where the field is supervised.
        bounds: Half-width of the cube the query points are drawn in.

    Returns:
        The generated shape, with ``shape_type`` set to ``"box"``, its half-extents recorded,
        and its exact field retained.
    """
    half: np.ndarray = np.exp(
        np.random.uniform(
            np.log(config.data.sampling_min), np.log(config.data.sampling_max), size=3
        )
    ).astype(np.float32)

    def sdf_fn(p: np.ndarray) -> np.ndarray:
        return box_sdf(p, half)

    # A box's surface is known in closed form, so it is sampled directly rather than through
    # the projection route of find_surface_points, which is for fields available only as a
    # callable and would still have to be told where the corners are.
    surface_points: np.ndarray = sample_box_surface_points(half, n_surface=n_surface)

    query_points: np.ndarray
    sdf_values: np.ndarray
    query_points, sdf_values = sample_query_points(
        sdf_fn=sdf_fn,
        surface_points=surface_points,
        n_query=n_query,
        bounds=bounds,
    )

    return ShapeData(
        surface_points=surface_points,
        query_points=query_points,
        sdf_values=sdf_values,
        shape_type="box",
        half_extents=half,
        sdf_fn=sdf_fn,
    )


class AutoEncoderDataset(torch.utils.data.Dataset):
    """All shapes stage 1 is fitted on, generated once and held in memory.

    Generation is cheap relative to reading from disk and the samples are never reused across
    runs, so nothing is written out. The consequence is that memory grows with the product of
    the shape count and the points per shape, which is what
    ``config.autoencoder.n_shapes`` and ``config.autoencoder.n_query`` together decide.

    Half-extents are stored alongside every shape and are read directly by the training: the
    box encoder takes the three half-extents, so the surface points this dataset also produces
    are not what it is fed.
    """

    def __init__(
        self,
        n_shapes: int = config.autoencoder.n_shapes,
        n_surface: int = config.autoencoder.n_surface,
        n_query: int = config.autoencoder.n_query,
    ):
        """Generate every shape up front.

        Args:
            n_shapes: Number of shapes in the set.
            n_surface: Surface points per shape.
            n_query: Query points per shape, each of which becomes one supervised distance.
        """
        self.n_shapes: int = n_shapes
        self.n_surface: int = n_surface
        self.n_query: int = n_query

        # Kept as parallel lists indexed by shape, so a shape's index is its identity. Stage 1
        # holds one free code per shape and looks it up by exactly this index.
        self.surface_points: list[np.ndarray] = []
        self.query_points: list[np.ndarray] = []
        self.sdf_values: list[np.ndarray] = []
        self.shape_types: list[str] = []
        self.half_extents: list[np.ndarray] = []

        print(f"Generating {n_shapes} random shapes...")
        for i in range(n_shapes):
            shape: ShapeData = generate_random_shape(
                n_surface=n_surface, n_query=n_query
            )
            self.surface_points.append(shape.surface_points)
            self.query_points.append(shape.query_points)
            self.sdf_values.append(shape.sdf_values)
            self.shape_types.append(shape.shape_type)
            self.half_extents.append(shape.half_extents)

            if (i + 1) % 1000 == 0:
                print(f"  Generated {i + 1}/{n_shapes} shapes")

        type_counts: dict = Counter(self.shape_types)
        print("\nShape type distribution:")
        for stype, count in sorted(type_counts.items()):
            print(f"  {stype}: {count} ({count / n_shapes * 100:.1f}%)")

    def __len__(self) -> int:
        """Number of shapes, which is also the number of free codes stage 1 fits."""
        return self.n_shapes

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return one shape as tensors.

        Returns:
            A tuple ``(surface, query, sdf)`` of shapes ``(n_surface, 3)``, ``(n_query, 3)``
            and ``(n_query,)``. The half-extents are not returned here; the training reads them
            from :attr:`half_extents` by the same index.
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
    """Generate the shapes and split them into training, validation and test loaders.

    Not used by ``src/enc_dec_training.py``, which builds the dataset itself so it can wrap it
    in an index and reach the half-extents.

    Args:
        n_shapes: Number of shapes in the set.
        n_surface: Surface points per shape.
        n_query: Query points per shape.
        train_split, val_split: Fractions of the set; the test split takes the remainder.
        batch_size: Shapes per batch, not points per batch.

    Returns:
        A tuple ``(train_loader, val_loader, test_loader)``. Only the training loader is
        shuffled.
    """
    dataset: AutoEncoderDataset = AutoEncoderDataset(
        n_shapes=n_shapes, n_surface=n_surface, n_query=n_query
    )

    # The test split takes what is left rather than its own fraction, so the three add up to
    # the whole set however the two fractions round.
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
