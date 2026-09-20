"""Analytic signed distance functions for the members of the shape vocabulary.

Every function maps ``(N, 3)`` query points in the part's local frame to ``(N,)`` signed
distances, negative inside, in whatever length unit its parameters are given in. The dataset
generation samples these fields to build the shapes the autoencoder trains on, and the mesh
export marches the same fields for the simulator, so a shape of the vocabulary is defined here
once and nowhere else.

Most members share one convention: a two-dimensional cross-section in the local Y-Z plane,
extruded along the local X axis and centred on the bounding box centre. The box, the sphere
and the ellipsoid of the vocabulary live in :mod:`src.enc_dec_dataset_generation` instead,
because they are orientation-free or carry their own surface sampler.

``lbracket_sdf`` and ``tshape_sdf`` predate that convention: they place the section in the X-Z
plane, extrude along Y and take two independent wall thicknesses. They are superseded by
``lprofile_sdf`` and ``tprofile_sdf`` and are not part of the fourteen-shape vocabulary.
"""

from __future__ import annotations

import numpy as np

from src.enc_dec_dataset_generation import box_sdf


def rounded_box_sdf(points: np.ndarray, half_extents: np.ndarray, radius: float) -> np.ndarray:
    """Box whose edges and corners are rounded off.

    Args:
        half_extents: Half extents of the resulting solid, the rounding included. The
            construction shrinks the box by ``radius`` and offsets the surface back out by
            the same amount, so the bounding box is unchanged.
    """
    he = np.asarray(half_extents, float)
    return box_sdf(points, he - radius) - radius


def lbracket_sdf(
    points: np.ndarray, Lx: float, Ly: float, Lz: float, tx: float, tz: float
) -> np.ndarray:
    """Angle bracket resting on its horizontal flange, which starts at ``z = 0``.

    The horizontal flange has footprint ``2 * Lx`` by ``2 * Ly`` and thickness ``tz``; the
    vertical flange stands on the ``-X`` side, is ``tx`` thick and reaches up to ``z = Lz``.
    Unlike the profiles below, this shape is not centred on its bounding box and uses two
    independent wall thicknesses.
    """
    hor = box_sdf(points - np.array([0.0, 0.0, tz / 2.0]), np.array([Lx, Ly, tz / 2.0]))
    ver = box_sdf(
        points - np.array([-Lx + tx / 2.0, 0.0, Lz / 2.0]),
        np.array([tx / 2.0, Ly, Lz / 2.0]),
    )
    return np.minimum(hor, ver)


def tshape_sdf(
    points: np.ndarray, Lx: float, Ly: float, Lz: float, tz: float, tx: float
) -> np.ndarray:
    """T-piece: a horizontal flange of thickness ``tz`` starting at ``z = 0``, plus a central
    vertical rib of thickness ``tx`` reaching up to ``z = Lz``. Same frame as
    :func:`lbracket_sdf`.
    """
    hor = box_sdf(points - np.array([0.0, 0.0, tz / 2.0]), np.array([Lx, Ly, tz / 2.0]))
    rib = box_sdf(points - np.array([0.0, 0.0, Lz / 2.0]), np.array([tx / 2.0, Ly, Lz / 2.0]))
    return np.minimum(hor, rib)


# The functions below follow the frame the generator downstream of this code uses to author
# parts. Deviating from it would train the encoder on a different geometry than the one it has
# to reconstruct at inference, which is what the two shapes above did.


def _rect2d_sdf(p2: np.ndarray, center: np.ndarray, half: np.ndarray) -> np.ndarray:
    """Exact 2-D signed distance to an axis-aligned rectangle (negative inside)."""
    q = np.abs(p2 - center) - half
    outside = np.linalg.norm(np.maximum(q, 0.0), axis=1)
    inside = np.minimum(np.max(q, axis=1), 0.0)
    return outside + inside


def _extrude_x(points: np.ndarray, d2: np.ndarray, length: float) -> np.ndarray:
    """Extrude a Y-Z cross-section of signed distance ``d2`` along X by ``length``.

    The combination is exact outside the solid and a lower bound inside, which is what the
    surface sampler and marching cubes need.
    """
    dx = np.abs(points[:, 0]) - length / 2.0
    w0 = np.maximum(d2, 0.0)
    w1 = np.maximum(dx, 0.0)
    outside = np.sqrt(w0 * w0 + w1 * w1)
    inside = np.minimum(np.maximum(d2, dx), 0.0)
    return outside + inside


def cylinder_x_sdf(points: np.ndarray, length: float, diameter: float) -> np.ndarray:
    """Capped cylinder whose axis is the local X axis; ``length`` is the full length."""
    radius = diameter / 2.0
    d_radial = np.linalg.norm(points[:, [1, 2]], axis=1) - radius
    return _extrude_x(points, d_radial, length)


def lprofile_sdf(
    points: np.ndarray, length: float, sy: float, sz: float, t: float
) -> np.ndarray:
    """Angle profile: a bottom flange and a left web meeting at the ``-Y``, ``-Z`` corner.

    Args:
        sy: Full width of the cross-section, along Y.
        sz: Full height of the cross-section, along Z.
        t: The single wall thickness shared by flange and web.
    """
    p2 = points[:, [1, 2]]
    flange = _rect2d_sdf(
        p2, np.array([0.0, -sz / 2.0 + t / 2.0]), np.array([sy / 2.0, t / 2.0])
    )
    web = _rect2d_sdf(
        p2, np.array([-sy / 2.0 + t / 2.0, t / 2.0]), np.array([t / 2.0, (sz - t) / 2.0])
    )
    return _extrude_x(points, np.minimum(flange, web), length)


def tprofile_sdf(
    points: np.ndarray, length: float, sy: float, sz: float, t: float
) -> np.ndarray:
    """T-profile: a top flange with a central web hanging below it.

    Args:
        sy: Full width of the cross-section, along Y.
        sz: Full height of the cross-section, along Z.
        t: The single wall thickness shared by flange and web.
    """
    p2 = points[:, [1, 2]]
    flange = _rect2d_sdf(
        p2, np.array([0.0, sz / 2.0 - t / 2.0]), np.array([sy / 2.0, t / 2.0])
    )
    web = _rect2d_sdf(
        p2, np.array([0.0, -t / 2.0]), np.array([t / 2.0, (sz - t) / 2.0])
    )
    return _extrude_x(points, np.minimum(flange, web), length)


def half_cylinder_x_sdf(points: np.ndarray, length: float, diameter: float) -> np.ndarray:
    """Half-round moulding: a semicircular section extruded along X, flat face down.

    The section is the half of a disk of radius ``diameter / 2`` above its own centre line,
    so the solid spans ``[-r, r]`` in Y but only ``[-r / 2, r / 2]`` in Z. The flat face is
    the ``-Z`` one, which is the face that rests on a support.
    """
    r = diameter / 2.0
    p2 = points[:, [1, 2]]
    disk = np.linalg.norm(p2 - np.array([0.0, -r / 2.0]), axis=1) - r
    # Negative where the point is above the flat face, so the maximum below keeps only the half
    # of the disk on that side. The second column of the section is Z, not Y.
    flat = -(p2[:, 1] + r / 2.0)
    return _extrude_x(points, np.maximum(disk, flat), length)


def capsule_x_sdf(points: np.ndarray, length: float, diameter: float) -> np.ndarray:
    """Capsule along X: a cylinder closed by two hemispherical caps.

    Args:
        length: Distance between the two cap centres, not the extent of the solid. The full
            X extent is ``length + diameter``.
    """
    p = points.copy()
    p[:, 0] -= np.clip(p[:, 0], -length / 2.0, length / 2.0)
    return np.linalg.norm(p, axis=1) - diameter / 2.0


def _dot2(v: np.ndarray) -> np.ndarray:
    """Squared length along the last axis."""
    return np.sum(v * v, axis=-1)


def cone_x_sdf(points: np.ndarray, length: float, d0: float, d1: float) -> np.ndarray:
    """Capped cone along X, of diameter ``d0`` at the ``-X`` end and ``d1`` at the ``+X`` end.

    Both ends are flat discs; a vanishing ``d1`` degenerates into a true cone with a tip.
    """
    h, r1, r2 = length / 2.0, d0 / 2.0, d1 / 2.0
    qx = np.linalg.norm(points[:, [1, 2]], axis=1)
    qy = points[:, 0]
    q = np.stack([qx, qy], axis=1)
    k1 = np.array([r2, h])
    k2 = np.array([r2 - r1, 2.0 * h])
    r_sel = np.where(qy < 0.0, r1, r2)
    ca = np.stack([qx - np.minimum(qx, r_sel), np.abs(qy) - h], axis=1)
    t = np.clip(((k1 - q) @ k2) / _dot2(k2), 0.0, 1.0)
    cb = q - k1 + k2[None, :] * t[:, None]
    s = np.where((cb[:, 0] < 0.0) & (ca[:, 1] < 0.0), -1.0, 1.0)
    return s * np.sqrt(np.minimum(_dot2(ca), _dot2(cb)))


def _hexagon2d_sdf(p2: np.ndarray, r: float) -> np.ndarray:
    """Signed distance to a regular hexagon of apothem ``r``, oriented flat top and bottom."""
    kx, ky, kz = -0.8660254, 0.5, 0.57735027
    p = np.abs(p2)
    m = 2.0 * np.minimum(kx * p[:, 0] + ky * p[:, 1], 0.0)
    p = p - np.stack([m * kx, m * ky], axis=1)
    cx = np.clip(p[:, 0], -kz * r, kz * r)
    p = p - np.stack([cx, np.full_like(cx, r)], axis=1)
    return np.linalg.norm(p, axis=1) * np.sign(p[:, 1])


def hex_prism_x_sdf(points: np.ndarray, length: float, apothem: float) -> np.ndarray:
    """Hexagonal prism along X.

    Args:
        apothem: Radius of the incircle of the hexagon, that is half the width across flats.
    """
    return _extrude_x(points, _hexagon2d_sdf(points[:, [1, 2]], apothem), length)


def _convex2d_sdf(p2: np.ndarray, planes2: list) -> np.ndarray:
    """Intersection of half planes, each given as an outward normal and a point on it."""
    vals = [(p2 - a) @ n for (n, a) in planes2]
    return np.maximum.reduce(vals)


def wedge_x_sdf(points: np.ndarray, length: float, sy: float, sz: float) -> np.ndarray:
    """Right-triangular prism along X, the shape of a gusset or brace.

    The right angle sits at the ``-Y``, ``-Z`` corner of the ``sy`` by ``sz`` cross-section
    and the hypotenuse runs from ``+Y``, ``-Z`` to ``-Y``, ``+Z``.
    """
    p2 = points[:, [1, 2]]
    A = np.array([-sy / 2.0, -sz / 2.0])
    B = np.array([sy / 2.0, -sz / 2.0])
    C = np.array([-sy / 2.0, sz / 2.0])
    n_hyp = np.array([(C - B)[1], -(C - B)[0]], float)
    n_hyp /= np.linalg.norm(n_hyp)
    if (A - B) @ n_hyp > 0:
        n_hyp = -n_hyp
    planes = [(np.array([0.0, -1.0]), A), (np.array([-1.0, 0.0]), A), (n_hyp, B)]
    return _extrude_x(points, _convex2d_sdf(p2, planes), length)


def i_profile_sdf(points: np.ndarray, length: float, sy: float, sz: float, t: float) -> np.ndarray:
    """I-beam: two flanges of full width ``sy`` joined by a central web, all ``t`` thick.

    ``sy`` and ``sz`` are the full width and height of the cross-section.
    """
    p2 = points[:, [1, 2]]
    top = _rect2d_sdf(p2, np.array([0.0, sz / 2.0 - t / 2.0]), np.array([sy / 2.0, t / 2.0]))
    bot = _rect2d_sdf(p2, np.array([0.0, -sz / 2.0 + t / 2.0]), np.array([sy / 2.0, t / 2.0]))
    web = _rect2d_sdf(p2, np.array([0.0, 0.0]), np.array([t / 2.0, sz / 2.0 - t]))
    return _extrude_x(points, np.minimum(np.minimum(top, bot), web), length)


def u_profile_sdf(points: np.ndarray, length: float, sy: float, sz: float, t: float) -> np.ndarray:
    """U-channel: a bottom flange with a web of full height ``sz`` on each side, all ``t`` thick.

    ``sy`` and ``sz`` are the full width and height of the cross-section, so the channel opens
    towards ``+Z``.
    """
    p2 = points[:, [1, 2]]
    bottom = _rect2d_sdf(p2, np.array([0.0, -sz / 2.0 + t / 2.0]), np.array([sy / 2.0, t / 2.0]))
    left = _rect2d_sdf(p2, np.array([-sy / 2.0 + t / 2.0, 0.0]), np.array([t / 2.0, sz / 2.0]))
    right = _rect2d_sdf(p2, np.array([sy / 2.0 - t / 2.0, 0.0]), np.array([t / 2.0, sz / 2.0]))
    return _extrude_x(points, np.minimum(np.minimum(bottom, left), right), length)
