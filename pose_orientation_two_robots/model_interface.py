"""The contract between the design pipeline and the repair models, and the scene conversion.

A proposed design arrives as a single row of CSV holding, for every block, its centre and the
extent of its axis-aligned bounding box; which blocks are joined to which; and where the
screws are meant to go. A repair returns the same row with the geometry and the screws
replaced. Everything is in metres, relative to the centre of the table surface.

Two things about the contract are load-bearing. The connectivity is passed through untouched:
a repair adjusts a design, it does not re-decide what is joined to what. And the orientation
never appears, because a block is described by the bounding box it occupies in the world, with
its rotation already resolved into that box. A repair therefore cannot turn a part, only
resize and move it, which is the scope Part I works in.

Two entry points are defined, one per repair model, so that both can be run on the same design
and compared. The first is wired to a placeholder that perturbs the design slightly, which
exercises the pipeline without a model; the repair of this thesis is dropped in by calling
``pipeline/repair_strategies.py`` from it. The second calls the model in ``claire_opti/``.
"""

from __future__ import annotations

import csv
import io
import math
import random
from dataclasses import dataclass, field
from typing import Any

# ============================================================
# Stub jitter magnitudes -- only used until the real models are wired
# in. Multiplicative on sizes (1.0 +/- _STUB_SIZE_JITTER) per axis,
# additive on positions/screws (metres). The clamps prevent degenerate
# sizes from sneaking in if the stub jitters towards zero.
# ============================================================
_STUB_SIZE_JITTER = 0.15
_STUB_POS_JITTER_M = 0.005
_STUB_SIZE_MIN_M = 0.005
_STUB_SIZE_MAX_M = 0.300


# ============================================================
# DATA MODEL
# ============================================================


@dataclass
class BlockEntry:
    """One block in scene order (Block0..BlockN-1).

    `world_size` is the AABB extent in WORLD axes (after `quat_xyzw`
    has been applied to the part's local mesh). `screws` lists every
    screw whose joint targets THIS block as the child, in YAML order.
    """

    block_id: str
    pos: tuple[float, float, float]
    world_size: tuple[float, float, float]
    quat_xyzw: tuple[float, float, float, float]
    screws: list[tuple[float, float, float]]
    file_path: str


@dataclass
class SceneState:
    """Whole-scene packing the CSV serializer / parser operates on."""

    blocks: list[BlockEntry]
    adjacency: list[list[bool]]  # NxN, symmetric, diagonal False
    id_to_index: dict[str, int] = field(default_factory=dict)

    @property
    def n(self) -> int:
        return len(self.blocks)


# ============================================================
# CSV COLUMN NAMES
# ============================================================


def _block_pos_cols(i: int) -> list[str]:
    return [f"Block{i}_PosX", f"Block{i}_PosY", f"Block{i}_PosZ"]


def _block_size_cols(i: int) -> list[str]:
    return [f"Block{i}_SizeX", f"Block{i}_SizeY", f"Block{i}_SizeZ"]


def _block_screw_cols(i: int, k: int) -> list[str]:
    return [f"Block{i}_Screw{k}_X", f"Block{i}_Screw{k}_Y", f"Block{i}_Screw{k}_Z"]


def _adjacency_col(i: int, j: int) -> str:
    # Upper-triangle name (i < j) so we never emit a duplicate column.
    if i > j:
        i, j = j, i
    return f"Block{i}_ConnectsTo_Block{j}"


def _build_header(n: int) -> list[str]:
    cols: list[str] = []
    for i in range(n):
        cols.extend(_block_pos_cols(i))
        cols.extend(_block_size_cols(i))
    for i in range(n):
        for j in range(i + 1, n):
            cols.append(_adjacency_col(i, j))
    for i in range(n):
        for k in range(n):
            cols.extend(_block_screw_cols(i, k))
    return cols


# ============================================================
# YAML <-> SceneState
# ============================================================


def _orientation_xyzw(node: dict[str, Any]) -> tuple[float, float, float, float]:
    return (
        float(node.get("x", 0.0)),
        float(node.get("y", 0.0)),
        float(node.get("z", 0.0)),
        float(node.get("w", 1.0)),
    )


def _position_xyz(node: dict[str, Any]) -> tuple[float, float, float]:
    return (
        float(node.get("x", 0.0)),
        float(node.get("y", 0.0)),
        float(node.get("z", 0.0)),
    )


def _final_pose_per_part(
    yaml_data: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    """For every part appearing as PICKANDPLACE / PICKANDHOLD, return
    its FINAL (place_position / hold_position) pose."""
    poses: dict[str, dict[str, Any]] = {}
    steps = (yaml_data.get("instruction") or {}).get("steps") or []
    for step in steps:
        st = step.get("type")
        if st == "PICKANDPLACE":
            pid = step.get("part_id")
            place = step.get("place_position") or {}
            if pid and place.get("position") and place.get("orientation"):
                poses[pid] = {
                    "position": _position_xyz(place["position"]),
                    "orientation": _orientation_xyzw(place["orientation"]),
                }
        elif st == "PICKANDHOLD":
            pid = step.get("part_id")
            hold = step.get("hold_position") or {}
            if pid and hold.get("position") and hold.get("orientation"):
                poses[pid] = {
                    "position": _position_xyz(hold["position"]),
                    "orientation": _orientation_xyzw(hold["orientation"]),
                }
    return poses


def _screws_per_child_part(
    yaml_data: dict[str, Any],
) -> dict[str, list[tuple[float, float, float]]]:
    """Walk every SCREWPARTS step and bucket each screw under the CHILD
    block of the joint_connection that introduces it. A SCREWPARTS step
    can contain multiple joint_connections + multiple screws; we match
    them positionally (1st screw <-> 1st joint, etc.). If counts mismatch
    we attribute the leftover screws to the first declared child."""
    out: dict[str, list[tuple[float, float, float]]] = {}
    steps = (yaml_data.get("instruction") or {}).get("steps") or []
    for step in steps:
        if step.get("type") != "SCREWPARTS":
            continue
        joints = step.get("joint_connections") or []
        screws = step.get("screw_positions") or []
        children = [j.get("child_part_id") for j in joints]
        for idx, scrw in enumerate(screws):
            pose = (scrw.get("pose") or {}).get("position")
            if pose is None:
                continue
            if not children:
                continue
            cid = children[idx] if idx < len(children) else children[0]
            if not cid:
                continue
            out.setdefault(cid, []).append(_position_xyz(pose))
    return out


def _world_aabb_from_local(
    local_size: tuple[float, float, float],
    quat_xyzw: tuple[float, float, float, float],
) -> tuple[float, float, float]:
    """Apply the 90-deg-multiple orientation to the local dims to get
    the world-axis-aligned extents. Lives here as a self-contained copy
    so model_interface has no dependency on refine_winner."""
    from scipy.spatial.transform import Rotation as R

    M = R.from_quat(list(quat_xyzw)).as_matrix()
    perm = [max(range(3), key=lambda k: abs(M[w, k])) for w in range(3)]
    return tuple(local_size[perm[w]] for w in range(3))  # type: ignore[return-value]


def scene_state_from_yaml(
    yaml_data: dict[str, Any],
    local_size_lookup: dict[str, tuple[float, float, float]],
) -> SceneState:
    """Build a SceneState from a parsed YAML dict.

    `local_size_lookup` maps part_id -> (sx, sy, sz) in metres, local
    frame. The caller computes this from the YAML's file entries (the
    USD filename encodes the local dims). We don't parse the filename
    here so this module stays decoupled from the library naming.
    """
    parts = ((yaml_data.get("instruction") or {}).get("parts") or {}).get(
        "active_parts"
    ) or []
    poses = _final_pose_per_part(yaml_data)
    screws = _screws_per_child_part(yaml_data)

    block_ids_in_order: list[str] = []
    for p in parts:
        pid = p.get("id")
        if pid and pid in poses and pid in local_size_lookup:
            block_ids_in_order.append(pid)

    blocks: list[BlockEntry] = []
    id_to_index: dict[str, int] = {}
    for idx, pid in enumerate(block_ids_in_order):
        pose = poses[pid]
        local = local_size_lookup[pid]
        world = _world_aabb_from_local(local, pose["orientation"])
        block_file = next(
            (p.get("file", "") for p in parts if p.get("id") == pid),
            "",
        )
        blocks.append(
            BlockEntry(
                block_id=pid,
                pos=pose["position"],
                world_size=world,
                quat_xyzw=pose["orientation"],
                screws=screws.get(pid, []),
                file_path=block_file,
            )
        )
        id_to_index[pid] = idx

    n = len(blocks)
    adjacency = [[False] * n for _ in range(n)]
    steps = (yaml_data.get("instruction") or {}).get("steps") or []
    for step in steps:
        if step.get("type") != "SCREWPARTS":
            continue
        for conn in step.get("joint_connections") or []:
            p_id = conn.get("parent_part_id")
            c_id = conn.get("child_part_id")
            if p_id in id_to_index and c_id in id_to_index:
                i, j = id_to_index[p_id], id_to_index[c_id]
                adjacency[i][j] = True
                adjacency[j][i] = True

    return SceneState(blocks=blocks, adjacency=adjacency, id_to_index=id_to_index)


# ============================================================
# SceneState <-> CSV
# ============================================================


def csv_from_scene(scene: SceneState) -> str:
    """Serialize the whole scene into a single-row CSV string with
    header (the ML model's expected input format)."""
    n = scene.n
    header = _build_header(n)
    row: dict[str, str] = {}

    for i, b in enumerate(scene.blocks):
        for col, val in zip(_block_pos_cols(i), b.pos):
            row[col] = repr(float(val))
        for col, val in zip(_block_size_cols(i), b.world_size):
            row[col] = repr(float(val))

    for i in range(n):
        for j in range(i + 1, n):
            row[_adjacency_col(i, j)] = "1" if scene.adjacency[i][j] else "0"

    nan_str = "NaN"
    for i in range(n):
        slot_screws = scene.blocks[i].screws
        for k in range(n):
            cols = _block_screw_cols(i, k)
            if k < len(slot_screws):
                for col, val in zip(cols, slot_screws[k]):
                    row[col] = repr(float(val))
            else:
                for col in cols:
                    row[col] = nan_str

    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=header)
    writer.writeheader()
    writer.writerow(row)
    return buf.getvalue()


def apply_csv_output_to_scene(
    output_csv: str,
    base_scene: SceneState,
) -> SceneState:
    """Parse the model's CSV output and return an updated SceneState.

    Adjacency and the orientation/file_path of every block carry over
    from `base_scene` unchanged (the model is not expected to refine
    those). The model's pos/size/screw values overwrite the originals.
    """
    reader = csv.DictReader(io.StringIO(output_csv))
    rows = list(reader)
    if not rows:
        raise ValueError("Model returned an empty CSV")
    row = rows[0]

    def _read_float(col: str) -> float:
        return float(row[col])

    def _read_screw(i: int, k: int) -> tuple[float, float, float] | None:
        x_s, y_s, z_s = (row[c] for c in _block_screw_cols(i, k))
        if any(v.lower() == "nan" for v in (x_s, y_s, z_s)):
            return None
        return (float(x_s), float(y_s), float(z_s))

    new_blocks: list[BlockEntry] = []
    n = base_scene.n
    for i, b in enumerate(base_scene.blocks):
        new_pos = tuple(_read_float(c) for c in _block_pos_cols(i))
        new_size = tuple(_read_float(c) for c in _block_size_cols(i))
        new_screws: list[tuple[float, float, float]] = []
        for k in range(n):
            s = _read_screw(i, k)
            if s is not None:
                new_screws.append(s)
        new_blocks.append(
            BlockEntry(
                block_id=b.block_id,
                pos=new_pos,  # type: ignore[arg-type]
                world_size=new_size,  # type: ignore[arg-type]
                quat_xyzw=b.quat_xyzw,
                screws=new_screws,
                file_path=b.file_path,
            )
        )

    return SceneState(
        blocks=new_blocks,
        adjacency=[row[:] for row in base_scene.adjacency],
        id_to_index=dict(base_scene.id_to_index),
    )


# ============================================================
# ML MODEL ENTRY POINTS
#
# Each function takes the CSV input string described in the module
# docstring and returns a CSV output string with the same column layout
# (adjacency columns may be passed through unchanged or omitted -- the
# orchestrator only reads pos/size/screw columns from the output).
#
# Two separate functions are exposed so the two ML colleagues can drop
# their own model in without coordinating. Both currently call the same
# stub with a different random seed so end-to-end pipeline runs produce
# two visibly different refinements.
# ============================================================


# ╔════════════════════════════════════════════════════════════════════╗
# ║ FABIO— REPLACE THIS FUNCTION BODY WITH YOUR MODEL CALL   ║
# ║                                                                    ║
# ║ Input  : `input_csv` — header + one data row using the schema      ║
# ║          documented at the top of this module.                     ║
# ║ Output : CSV string with the same header layout; preserve the      ║
# ║          adjacency columns (or omit them — they will be ignored).  ║
# ║          Pos/Size/Screw values may be modified.                    ║
# ╚════════════════════════════════════════════════════════════════════╝
def refine_full_yaml_model_a(input_csv: str) -> str:

    return placeholder(input_csv)


# ╔════════════════════════════════════════════════════════════════════╗
# ║ CLAIRE — REPLACE THIS FUNCTION BODY WITH YOUR MODEL CALL   ║
# ║                                                                    ║
# ║ Same input/output contract as `refine_full_yaml_model_a`. The      ║
# ║ orchestrator calls both A and B in parallel on every winner YAML.  ║
# ╚════════════════════════════════════════════════════════════════════╝
def refine_full_yaml_model_b(input_csv: str) -> str:
    try:
        from .claire_opti import geometric_optimization
    except ImportError:
        import sys as _sys
        from pathlib import Path as _Path

        _sys.path.insert(0, str(_Path(__file__).resolve().parent))
        from claire_opti import geometric_optimization
    return geometric_optimization(input_csv)


# ============================================================
# STUB — random jitter on pos / size / screw columns, adjacency passed
# through unchanged. Used by both model_a and model_b until the real
# ML calls are wired in.
# ============================================================


def _stub_inference(input_csv: str, seed: int) -> str:
    """Identity-with-jitter pass-through for end-to-end testing."""
    rng = random.Random(seed)
    reader = csv.DictReader(io.StringIO(input_csv))
    header = list(reader.fieldnames or [])
    rows = list(reader)
    if not rows:
        return input_csv
    row = dict(rows[0])

    def jitter_size(s: float) -> float:
        if math.isnan(s):
            return s
        factor = 1.0 + rng.uniform(-_STUB_SIZE_JITTER, _STUB_SIZE_JITTER)
        return max(_STUB_SIZE_MIN_M, min(_STUB_SIZE_MAX_M, s * factor))

    def jitter_pos(p: float) -> float:
        if math.isnan(p):
            return p
        return p + rng.uniform(-_STUB_POS_JITTER_M, _STUB_POS_JITTER_M)

    for col in header:
        if "_Size" in col:
            try:
                row[col] = repr(jitter_size(float(row[col])))
            except (TypeError, ValueError):
                pass
        elif "_Pos" in col or "_Screw" in col:
            try:
                val = float(row[col])
            except (TypeError, ValueError):
                continue
            row[col] = "NaN" if math.isnan(val) else repr(jitter_pos(val))
        # adjacency columns pass through untouched

    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=header)
    writer.writeheader()
    writer.writerow(row)
    return buf.getvalue()
