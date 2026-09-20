"""Turn the analytic shape vocabulary into meshes the simulator can spawn.

This is one of the two points at which this repository meets the simulation repository. Each of
the fourteen distance fields is sampled on a grid and turned into a watertight triangle mesh by
marching cubes, and one file is written per shape into the folder the simulator reads.

Each shape is exported at one representative size. The simulator scales what it spawns, so what
has to be right here is the form and the frame, not the size: length along the local X axis,
cross-section in the Y-Z plane, centred on the origin, which is the frame the autoencoder is
trained in. Dimensions are in metres unless a scale is given.

This module is also imported by the tool that computes the code of each shape, which needs the
same definitions and the same meshing, so the two cannot describe different geometry.

Run:
    python -m tools.export_shape_objs --scale 100
    python -m tools.export_shape_objs --out <dir> --res 128 --only box cylinder

Writes:
    one ``<shape>.obj`` per shape of the vocabulary, into the output folder.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import trimesh
from skimage import measure

_CODE = Path(__file__).resolve().parents[1]
if str(_CODE) not in sys.path:
    sys.path.insert(0, str(_CODE))

from src.enc_dec_dataset_generation import box_sdf, ellipsoid_sdf, sphere_sdf  # noqa: E402
from src import shape_primitives as sp  # noqa: E402

_DEFAULT_OUT = (
    _CODE.parent
    / "Woodworking_Simulation_dataset_stls"
    / "source/Woodworking_Simulation/Woodworking_Simulation"
    / "tasks/direct/pose_orientation_two_robots/obj_files"
)

# The length every prismatic member is exported at, in metres. The cross-sections and wall
# thicknesses below are chosen around it so that the exported parts are of a size a robot could
# plausibly handle; the simulator rescales them anyway.
L = 0.100


def _shapes() -> dict:
    """The vocabulary as name to distance function and bounding half extents in metres.

    The half extents are not derived from the field; they are stated alongside it and are what
    the meshing uses to place its grid.
    """
    return {
        "box":            (lambda p: box_sdf(p, np.array([L / 2, 0.024, 0.012])),
                           [L / 2, 0.024, 0.012]),
        "rounded_box":    (lambda p: sp.rounded_box_sdf(p, np.array([L / 2, 0.024, 0.012]), 0.004),
                           [L / 2, 0.024, 0.012]),
        "cylinder":       (lambda p: sp.cylinder_x_sdf(p, L, 0.036),
                           [L / 2, 0.018, 0.018]),
        "half_cylinder":  (lambda p: sp.half_cylinder_x_sdf(p, L, 0.036),
                           [L / 2, 0.018, 0.010]),
        "sphere":         (lambda p: sphere_sdf(p, 0.030),
                           [0.030, 0.030, 0.030]),
        "ellipsoid":      (lambda p: ellipsoid_sdf(p, np.array([0.050, 0.030, 0.020])),
                           [0.050, 0.030, 0.020]),
        "capsule":        (lambda p: sp.capsule_x_sdf(p, 0.070, 0.030),
                           [0.050, 0.015, 0.015]),
        "cone":           (lambda p: sp.cone_x_sdf(p, L, 0.048, 0.016),
                           [L / 2, 0.024, 0.024]),
        "hex_prism":      (lambda p: sp.hex_prism_x_sdf(p, L, 0.022),
                           [L / 2, 0.026, 0.026]),
        "wedge":          (lambda p: sp.wedge_x_sdf(p, L, 0.048, 0.048),
                           [L / 2, 0.024, 0.024]),
        "i_profile":      (lambda p: sp.i_profile_sdf(p, L, 0.048, 0.048, 0.008),
                           [L / 2, 0.024, 0.024]),
        "u_profile":      (lambda p: sp.u_profile_sdf(p, L, 0.048, 0.048, 0.008),
                           [L / 2, 0.024, 0.024]),
        "l_profile":      (lambda p: sp.lprofile_sdf(p, L, 0.048, 0.048, 0.008),
                           [L / 2, 0.024, 0.024]),
        "t_profile":      (lambda p: sp.tprofile_sdf(p, L, 0.048, 0.048, 0.008),
                           [L / 2, 0.024, 0.024]),
    }


def _mesh_from_sdf(sdf_fn, half, res: int, pad: float = 0.18):
    """Mesh a centred distance field by marching cubes; the result is in metres."""
    half = np.asarray(half, float)
    # The grid is padded beyond the shape and then by two more cells. A surface touching the
    # boundary of the grid is left open there, and an open mesh cannot be given a volume.
    b = half * (1.0 + pad) + 2.0 * (2.0 * half.max() / res)
    axes = [np.linspace(-b[i], b[i], res) for i in range(3)]
    gx, gy, gz = np.meshgrid(*axes, indexing="ij")
    pts = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)
    vol = sdf_fn(pts).reshape(res, res, res)
    spacing = tuple((2.0 * b[i]) / (res - 1) for i in range(3))
    verts, faces, _, _ = measure.marching_cubes(vol, level=0.0, spacing=spacing)
    # Marching cubes returns vertices in grid indices scaled by the spacing, so the origin of
    # the grid has to be added back to return the mesh to the frame the field was defined in.
    verts += np.array([-b[0], -b[1], -b[2]])
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=True)
    # The simulator needs a closed solid to give the part a mass and an inertia.
    if not mesh.is_watertight:
        trimesh.repair.fill_holes(mesh)
    mesh.fix_normals()
    return mesh


def _decimate(mesh, target: int):
    """Reduce the face count if the backend for it is installed, and leave the mesh alone if not."""
    if target <= 0 or len(mesh.faces) <= target:
        return mesh
    try:
        out = mesh.simplify_quadric_decimation(face_count=target)
        out.fix_normals()
        return out
    except Exception as exc:  # noqa: BLE001
        print(f"    [decimation skipped: {exc}]")
        return mesh


def main() -> None:
    """Mesh and write every requested shape, reporting the extent and whether it came out closed."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=_DEFAULT_OUT, help="output obj_files dir")
    ap.add_argument("--scale", type=float, default=1.0, help="multiply vertices (1000 = mm)")
    ap.add_argument("--res", type=int, default=128, help="marching-cubes grid resolution")
    ap.add_argument("--decimate", type=int, default=6000, help="target face count (0 = off)")
    ap.add_argument("--only", nargs="*", default=None, help="subset of shape names")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    shapes = _shapes()
    names = args.only or list(shapes)

    print(f"exporting {len(names)} shapes -> {out}  (scale={args.scale}, res={args.res})")
    print(f"{'shape':<14}{'verts':>8}{'faces':>8}   extent (m)")
    for name in names:
        sdf_fn, half = shapes[name]
        mesh = _mesh_from_sdf(sdf_fn, half, args.res)
        mesh = _decimate(mesh, args.decimate)
        if args.scale != 1.0:
            mesh.apply_scale(args.scale)
        path = out / f"{name}.obj"
        mesh.export(path)
        ext = mesh.extents / (args.scale if args.scale else 1.0)
        wt = "watertight" if mesh.is_watertight else "NOT-watertight"
        print(f"{name:<14}{len(mesh.vertices):>8}{len(mesh.faces):>8}   "
              f"[{ext[0]:.3f} {ext[1]:.3f} {ext[2]:.3f}]  {wt}")
    print("done.")


if __name__ == "__main__":
    main()
