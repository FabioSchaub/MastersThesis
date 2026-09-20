"""Repair a whole design and place its fasteners: one CSV string in, one CSV string out.

This is the interface the rest of the system calls. A design arrives as a single row holding
every block's position and size, which blocks are joined to which, and where the screws were
meant to go; the same row comes back with all three updated and no other change. Keeping the
schema identical is what makes the repair a step that can be inserted into an existing
pipeline instead of a stage that has to be integrated into it.

The work happens in four stages, and the order matters.

First, any pair of blocks that interpenetrate in all three axes is separated in closed form,
both blocks giving up half the depth on the cheapest axis. Second, every joined pair is
brought into face-to-face contact, walking outwards from the largest block. Both stages are
purely geometric, and they exist to hand the surrogate a design of the kind it was fitted on:
blocks that touch, and do not occupy the same space.

Third, the surrogate-driven repair runs upwards from the blocks resting on the table. Each
joint is repaired against a parent that has already been fixed, so a child is always fitted to
what will actually be built rather than to what was proposed.

Fourth, the screws are placed. Each one is assigned to the joint it was closest to in the
original design and then moved into the repaired contact rectangle, with an inset from the
edges and a check against every screw already placed. A repaired design with a screw hanging
over an edge would not be buildable, so the geometry and the fasteners have to be repaired
together.

    python pipeline/repair_strategies.py design.csv
    python pipeline/repair_strategies.py design.csv repaired.csv

Called without arguments the file lists the CSVs next to it and asks which to use. The output
keeps the input's columns and order; connectivity is passed through, and screw slots that are
unused stay empty.
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.analytical_metrics import infer_face, snap_to_contact_face  # noqa: E402
import src.repair_optimizer as _repair_optimizer  # noqa: E402
from src.repair_process import (  # noqa: E402
    BlockData,
    load_models,
    repair_chain_single,
)

# The thickness limit the repair aims at, in metres, written onto the optimiser module so that
# the hinge, the early stop and the commit gate all use it. It currently equals the limit in
# ``config/config.yaml``; the indirection exists so that this deployment path can aim below the
# nominal limit without changing what the surrogate was trained against.
REPAIR_TARGET_THICKNESS_M = 0.02
_repair_optimizer.THRESH_THICKNESS = REPAIR_TARGET_THICKNESS_M

# How close to the table a block has to be to count as resting on it, in metres. The repair
# starts from these blocks and works upwards.
FLOOR_TOLERANCE_M = 1e-3

# Fastener geometry, in metres: the length of a screw, how far it is kept from the edge of the
# contact rectangle, and how far two screws must stay apart in the horizontal plane. The
# length is used to decide whether two screws are at different heights and can therefore be
# closer together than the separation alone would allow.
SCREW_LENGTH_M = 0.022
SCREW_SAFETY_M = 0.005
SCREW_SEPARATION_M = 0.010

# A last shrink along the contact axis, in metres, after the repair has finished. The
# surrogate's thickness is a prediction, so a joint it accepts may still sit just above the
# limit in closed form; taking a millimetre off and re-establishing contact puts it under,
# and leaves the overlap alone because nothing tangential is touched.
POST_SHRINK_M = 0.001

# Smallest edge length a block may be shrunk to, matching the floor used inside the repair.
SIZE_MIN_M = 0.008


def detect_n_blocks(df: pd.DataFrame) -> int:
    """Number of blocks the schema provides for, counted from the position columns."""
    n = 0
    while f"Block{n}_PosX" in df.columns:
        n += 1
    return n


def parse_blocks(row: pd.Series, n: int) -> dict[int, dict]:
    """Blocks of one design, keyed by index, with centre and edge lengths in metres.

    Slots the design does not use are left out rather than filled in, so the schema can
    provide for more blocks than a given design has.
    """
    blocks: dict[int, dict] = {}
    for i in range(n):
        pos = np.array(
            [row[f"Block{i}_PosX"], row[f"Block{i}_PosY"], row[f"Block{i}_PosZ"]],
            dtype=float,
        )
        size = np.array(
            [row[f"Block{i}_SizeX"], row[f"Block{i}_SizeY"], row[f"Block{i}_SizeZ"]],
            dtype=float,
        )
        if np.isnan(pos).any() or np.isnan(size).any():
            continue
        blocks[i] = {"pos": pos, "size": size}
    return blocks


def parse_connections(row: pd.Series, n: int) -> list[tuple[int, int]]:
    """Joined pairs of the design, each with the lower index first."""
    conns: list[tuple[int, int]] = []
    for i in range(n):
        for j in range(i + 1, n):
            col = f"Block{i}_ConnectsTo_Block{j}"
            if col in row.index and int(row[col]) == 1:
                conns.append((i, j))
    return conns


def parse_original_screws(
    row: pd.Series, n_blocks: int, n_screws: int = 5
) -> dict[int, list[tuple[int, np.ndarray]]]:
    """Screws of the design, grouped by the block whose columns they are stored in.

    Each entry keeps its slot number, so a repaired screw can be written back into the column
    it came from. Empty slots are skipped.
    """
    out: dict[int, list[tuple[int, np.ndarray]]] = {i: [] for i in range(n_blocks)}
    for i in range(n_blocks):
        for s in range(n_screws):
            cols = [f"Block{i}_Screw{s}_{ax}" for ax in "XYZ"]
            if not all(c in row.index for c in cols):
                continue
            vals = np.array([float(row[c]) for c in cols], dtype=float)
            if np.isnan(vals).any():
                continue
            out[i].append((s, vals))
    return out


def _aabb_overlap_xyz(b_i: dict, b_j: dict) -> np.ndarray:
    """Per-axis overlap of two blocks in metres; all three positive means they interpenetrate."""
    delta = np.abs(b_i["pos"] - b_j["pos"])
    return (b_i["size"] + b_j["size"]) / 2.0 - delta


def resolve_penetrations(
    blocks: dict[int, dict], conns_set: set[tuple[int, int]]
) -> dict[int, dict]:
    """Separate every pair of blocks that occupies the same space, by shrinking both.

    The axis with the least overlap is chosen, since removing material there costs the design
    the least, and the two blocks share the correction equally rather than one absorbing all
    of it. Resolving one pair can create another, so the process repeats until nothing
    interpenetrates.

    Args:
        blocks: The design, keyed by block index.
        conns_set: Joined pairs. Not used to decide what to separate: two blocks that occupy
            the same space have to be separated whether or not they are meant to be joined.

    Returns:
        A new design; the input is left untouched.
    """
    new = {
        k: {"pos": v["pos"].copy(), "size": v["size"].copy()} for k, v in blocks.items()
    }
    keys = sorted(new.keys())
    # Bounded rather than run to convergence: the correction is not guaranteed to terminate on
    # a pathological design, and a repair that hangs is worse than one that gives up.
    for _ in range(50):
        candidates: list[tuple[int, int, np.ndarray]] = []
        for ii, i in enumerate(keys):
            for j in keys[ii + 1 :]:
                ov = _aabb_overlap_xyz(new[i], new[j])
                if (ov > 1e-9).all():
                    candidates.append((i, j, ov))
        if not candidates:
            break
        # The deepest interpenetration is resolved first, so a single pass makes the largest
        # difference and the remaining ones are corrected in later rounds.
        candidates.sort(key=lambda t: -float(t[2].min()))
        i, j, ov = candidates[0]
        ax = int(np.argmin(ov))
        depth = float(ov[ax])
        if new[i]["pos"][ax] <= new[j]["pos"][ax]:
            lo, hi = i, j
        else:
            lo, hi = j, i
        delta = depth / 2.0
        new[lo]["size"][ax] -= delta
        new[lo]["pos"][ax] -= delta / 2.0
        new[hi]["size"][ax] -= delta
        new[hi]["pos"][ax] += delta / 2.0
    return new


def apply_snaps(
    blocks: dict[int, dict], conns: list[tuple[int, int]]
) -> dict[int, dict]:
    """Bring every joined pair into face-to-face contact.

    The walk starts at the largest block and moves outwards along the connections, so the
    bulkiest part of the design stays where it was placed and the smaller ones are moved onto
    it. Each block is snapped once, against the neighbour it was reached from.

    Returns:
        A new design; the input is left untouched.
    """
    if not blocks:
        return blocks
    largest = max(blocks, key=lambda i: float(np.prod(blocks[i]["size"])))
    adj: dict[int, set[int]] = {i: set() for i in blocks}
    for i, j in conns:
        adj[i].add(j)
        adj[j].add(i)
    visited = {largest}
    queue = [largest]
    edges_order: list[tuple[int, int]] = []
    while queue:
        cur = queue.pop(0)
        for nbr in sorted(adj[cur]):
            if nbr in visited:
                continue
            edges_order.append((cur, nbr))
            visited.add(nbr)
            queue.append(nbr)
    snapped = {
        k: {"pos": v["pos"].copy(), "size": v["size"].copy()} for k, v in blocks.items()
    }
    for parent, child in edges_order:
        p, c = snapped[parent], snapped[child]
        pos_c_new, _shift, _axis, _gap = snap_to_contact_face(
            p["pos"], p["size"] / 2.0, c["pos"], c["size"] / 2.0
        )
        snapped[child]["pos"] = pos_c_new
    return snapped


def _make_block_data(
    pos_m: np.ndarray,
    size_m: np.ndarray,
    scale_factor: float,
    device: torch.device,
    prefix: str,
) -> BlockData:
    """Convert one block from metres into the form the repair works in."""
    pos_scaled = torch.from_numpy((pos_m * scale_factor).astype(np.float32)).to(device)
    size_scaled = torch.from_numpy((size_m * scale_factor).astype(np.float32)).to(
        device
    )
    return BlockData(prefix=prefix, pos=pos_scaled, size=size_scaled)


def _post_shrink_and_snap(
    parent: BlockData,
    child: BlockData,
    contact_axis: int,
    scale_factor: float,
    device: torch.device,
    shrink_m: float = POST_SHRINK_M,
) -> BlockData:
    """Take a last millimetre off the child's contact axis and put it back onto its parent.

    Guards against the surrogate's thickness sitting slightly below the true value: the joint
    it accepted may be marginally over the limit in closed form, and this pushes it under
    without disturbing the overlap.
    """
    sf = float(scale_factor)
    shrink_scaled = shrink_m * sf
    min_scaled = SIZE_MIN_M * sf

    new_size = child.size.clone()
    new_size[contact_axis] = torch.clamp(
        new_size[contact_axis] - shrink_scaled, min=min_scaled,
    )

    pos_p_m = parent.pos.cpu().numpy() / sf
    he_p_m = (parent.size / 2.0).cpu().numpy() / sf
    pos_c_m = child.pos.cpu().numpy() / sf
    he_c_m = (new_size / 2.0).cpu().numpy() / sf
    pos_c_snapped, _, _, _ = snap_to_contact_face(
        pos_p_m, he_p_m, pos_c_m, he_c_m, contact_axis=contact_axis,
    )
    new_pos = torch.from_numpy(
        (pos_c_snapped * sf).astype(np.float32)
    ).to(device)
    return BlockData(prefix=child.prefix, pos=new_pos, size=new_size)


def gnn_repair_layered(
    blocks: dict[int, dict],
    conns: list[tuple[int, int]],
    gnn_model,
    scale_factor: float,
    device: torch.device,
    freeze_pos_if_overlap_ok: bool = True,
    post_shrink_m: float = POST_SHRINK_M,
) -> dict[int, dict]:
    """Repair the design joint by joint, working upwards from the blocks that rest on the table.

    Those blocks are frozen and become the anchors for whatever is attached to them; each
    repaired child is then itself frozen and anchors the next layer. A child is always fitted
    to a parent that is already final, so no joint has to be revisited.

    A design in which nothing rests on the table falls back to anchoring the largest block, so
    that the repair has somewhere to start rather than refusing the design.

    Args:
        blocks: The design in metres, keyed by block index.
        conns: Joined pairs.
        gnn_model: The surrogate.
        scale_factor: Factor between the scaled domain and metres.
        device: Device the repair runs on.
        freeze_pos_if_overlap_ok: Passed through to the repair.
        post_shrink_m: Safety margin taken off the contact axis afterwards; zero disables it.

    Returns:
        The repaired design in metres, in the same form as the input.
    """
    adj: dict[int, set[int]] = {i: set() for i in blocks}
    for i, j in conns:
        adj[i].add(j)
        adj[j].add(i)

    bd: dict[int, BlockData] = {
        i: _make_block_data(
            b["pos"], b["size"], scale_factor, device, prefix=f"Block{i}"
        )
        for i, b in blocks.items()
    }

    floor = {
        i
        for i, b in blocks.items()
        if abs(b["pos"][2] - b["size"][2] / 2.0) < FLOOR_TOLERANCE_M
    }
    if not floor:
        floor = {max(blocks, key=lambda i: float(np.prod(blocks[i]["size"])))}

    committed: set[int] = set(floor)
    current_layer: set[int] = set(floor)
    while current_layer:
        next_layer: set[int] = set()
        for parent in sorted(current_layer):
            for child in sorted(adj[parent]):
                # A block with several parents is repaired against the first of them only.
                # Repairing it again against the next would overwrite the result and
                # accumulate drift, and the block can only sit in one place.
                if child in committed or child in next_layer:
                    continue
                prev = bd[parent]
                curr = bd[child]

                # The parent may have shrunk while it was being repaired, leaving a gap the
                # earlier geometric stage knew nothing about. Closing it before the surrogate
                # is asked matters twice over: the prediction then describes the joint as it
                # will be built, and the decision to leave the child's position alone is made
                # on the right overlap rather than on a stale one.
                sf = float(scale_factor)
                pos_p_m = prev.pos.cpu().numpy() / sf
                he_p_m = (prev.size / 2.0).cpu().numpy() / sf
                pos_c_m = curr.pos.cpu().numpy() / sf
                he_c_m = (curr.size / 2.0).cpu().numpy() / sf
                face = infer_face(pos_p_m, he_p_m, pos_c_m, he_c_m)
                contact_ax = face // 2
                pos_c_snapped, _, _, _ = snap_to_contact_face(
                    pos_p_m,
                    he_p_m,
                    pos_c_m,
                    he_c_m,
                    contact_axis=contact_ax,
                )
                curr = BlockData(
                    prefix=curr.prefix,
                    pos=torch.from_numpy((pos_c_snapped * sf).astype(np.float32)).to(
                        device
                    ),
                    size=curr.size,
                )
                bd[child] = curr

                chain = repair_chain_single(
                    [prev, curr],
                    gnn_model,
                    scale_factor,
                    device,
                    freeze_pos_if_overlap_ok=freeze_pos_if_overlap_ok,
                    verbose=False,
                )
                step = chain.steps[0] if chain.steps else None
                if (
                    step is not None
                    and step.size_after is not None
                    and step.pos_after is not None
                ):
                    new_pos = (
                        torch.from_numpy(step.pos_after).float().to(device)
                        if not isinstance(step.pos_after, torch.Tensor)
                        else step.pos_after.float().to(device)
                    )
                    bd[child] = BlockData(
                        prefix=curr.prefix,
                        pos=new_pos,
                        size=step.size_after.detach(),
                    )
                    if post_shrink_m > 0:
                        bd[child] = _post_shrink_and_snap(
                            bd[parent],
                            bd[child],
                            contact_ax,
                            scale_factor,
                            device,
                            shrink_m=post_shrink_m,
                        )
                next_layer.add(child)
        committed |= next_layer
        current_layer = next_layer

    out: dict[int, dict] = {}
    for i, b in bd.items():
        pos_m = b.pos.cpu().numpy() / scale_factor
        size_m = b.size.cpu().numpy() / scale_factor
        out[i] = {"pos": pos_m.astype(float), "size": size_m.astype(float)}
    return out


def _distribute_in_rect(
    x_min: float, x_max: float, y_min: float, y_max: float, n: int
) -> list[tuple[float, float]]:
    """Spread ``n`` screws over a rectangle, as far apart as the rectangle allows.

    A single screw goes in the middle, two go to the ends of the longer side, three or four to
    the corners, and more are laid out on a grid. The point is to resist rotation of the
    joint, which two screws close together would not.
    """
    if n <= 0:
        return []
    if n == 1:
        return [((x_min + x_max) / 2.0, (y_min + y_max) / 2.0)]
    if n == 2:
        if (x_max - x_min) >= (y_max - y_min):
            return [
                (x_min, (y_min + y_max) / 2.0),
                (x_max, (y_min + y_max) / 2.0),
            ]
        return [
            ((x_min + x_max) / 2.0, y_min),
            ((x_min + x_max) / 2.0, y_max),
        ]
    if n <= 4:
        return [
            (x_min, y_min),
            (x_max, y_min),
            (x_min, y_max),
            (x_max, y_max),
        ][:n]
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))
    out: list[tuple[float, float]] = []
    for r in range(rows):
        for c in range(cols):
            if len(out) >= n:
                break
            fx = c / max(1, cols - 1)
            fy = r / max(1, rows - 1)
            out.append((x_min + fx * (x_max - x_min), y_min + fy * (y_max - y_min)))
    return out


def update_screws_for_repair(
    orig_screws: dict[int, list[tuple[int, np.ndarray]]],
    orig_blocks: dict[int, dict],
    repaired_blocks: dict[int, dict],
    conns: list[tuple[int, int]],
) -> dict[int, list[tuple[int, np.ndarray]]]:
    """Move every screw of the design into the contact rectangle of the joint it belongs to.

    Which joint that is has to be worked out, because the input stores a screw against a block
    rather than against a joint. Each screw is assigned to the joint whose upper face and
    whose contact rectangle it was closest to in the original design, so the intent of the
    design is preserved even though the geometry underneath it has moved.

    Screws are then laid out inside the repaired rectangle, kept clear of its edges, and each
    one is nudged away from any screw already placed that is too close horizontally and at a
    similar height. A screw over an edge or into another screw would make an otherwise
    feasible design unbuildable.

    Args:
        orig_screws: Screws as read from the input, grouped by the block they are stored on.
        orig_blocks: The design as it arrived, used only to decide which joint a screw meant.
        repaired_blocks: The design after the repair, which the new positions are computed in.
        conns: Joined pairs.

    Returns:
        The new screws, in the same grouping, so each keeps the column it came from. A screw
        whose joint is not part of the repaired design is passed through unchanged.
    """

    def _xy_dist_to_rect(
        px: float, py: float, x_min: float, x_max: float, y_min: float, y_max: float
    ) -> float:
        """Horizontal distance from a point to a rectangle, zero inside it."""
        dx = max(x_min - px, 0.0, px - x_max)
        dy = max(y_min - py, 0.0, py - y_max)
        return float(np.hypot(dx, dy))

    pair_to_screws: dict[tuple[int, int], list[tuple[int, int, np.ndarray]]] = {}
    for stored_block_idx, screw_list in orig_screws.items():
        for slot, orig_xyz in screw_list:
            best_pair: tuple[int, int] | None = None
            best_score = float("inf")
            for a, b in conns:
                if orig_blocks[a]["pos"][2] >= orig_blocks[b]["pos"][2]:
                    upper_o, lower_o = a, b
                else:
                    upper_o, lower_o = b, a
                upper_top = (
                    orig_blocks[upper_o]["pos"][2]
                    + orig_blocks[upper_o]["size"][2] / 2.0
                )
                z_dist = abs(float(orig_xyz[2]) - upper_top)
                lo, up = orig_blocks[lower_o], orig_blocks[upper_o]
                ox_min = max(
                    lo["pos"][0] - lo["size"][0] / 2.0,
                    up["pos"][0] - up["size"][0] / 2.0,
                )
                ox_max = min(
                    lo["pos"][0] + lo["size"][0] / 2.0,
                    up["pos"][0] + up["size"][0] / 2.0,
                )
                oy_min = max(
                    lo["pos"][1] - lo["size"][1] / 2.0,
                    up["pos"][1] - up["size"][1] / 2.0,
                )
                oy_max = min(
                    lo["pos"][1] + lo["size"][1] / 2.0,
                    up["pos"][1] + up["size"][1] / 2.0,
                )
                xy_dist = _xy_dist_to_rect(
                    float(orig_xyz[0]),
                    float(orig_xyz[1]),
                    ox_min,
                    ox_max,
                    oy_min,
                    oy_max,
                )
                score = z_dist + xy_dist
                if score < best_score:
                    best_score = score
                    best_pair = (lower_o, upper_o)
            if best_pair is None:
                continue
            pair_to_screws.setdefault(best_pair, []).append(
                (stored_block_idx, slot, orig_xyz)
            )

    new_screws: dict[int, list[tuple[int, np.ndarray]]] = {
        i: [] for i in orig_blocks.keys() | repaired_blocks.keys()
    }
    all_placed: list[np.ndarray] = []

    for (lower_idx, upper_idx), screws_in_pair in pair_to_screws.items():
        if lower_idx not in repaired_blocks or upper_idx not in repaired_blocks:
            for stored_idx, slot, orig_xyz in screws_in_pair:
                new_screws[stored_idx].append((slot, orig_xyz.copy()))
            continue
        lo = repaired_blocks[lower_idx]
        up = repaired_blocks[upper_idx]
        ox_min = max(
            lo["pos"][0] - lo["size"][0] / 2.0,
            up["pos"][0] - up["size"][0] / 2.0,
        )
        ox_max = min(
            lo["pos"][0] + lo["size"][0] / 2.0,
            up["pos"][0] + up["size"][0] / 2.0,
        )
        oy_min = max(
            lo["pos"][1] - lo["size"][1] / 2.0,
            up["pos"][1] - up["size"][1] / 2.0,
        )
        oy_max = min(
            lo["pos"][1] + lo["size"][1] / 2.0,
            up["pos"][1] + up["size"][1] / 2.0,
        )
        sx_min, sx_max = ox_min + SCREW_SAFETY_M, ox_max - SCREW_SAFETY_M
        sy_min, sy_max = oy_min + SCREW_SAFETY_M, oy_max - SCREW_SAFETY_M
        # A contact rectangle narrower than twice the inset leaves nothing to place into. The
        # centre line is used instead, which is the best available compromise.
        if sx_min > sx_max:
            sx_min = sx_max = (ox_min + ox_max) / 2.0
        if sy_min > sy_max:
            sy_min = sy_max = (oy_min + oy_max) / 2.0
        upper_top_new = up["pos"][2] + up["size"][2] / 2.0

        n_screws = len(screws_in_pair)
        positions = _distribute_in_rect(sx_min, sx_max, sy_min, sy_max, n_screws)
        orig_order = sorted(
            range(n_screws),
            key=lambda k: (
                float(screws_in_pair[k][2][0]),
                float(screws_in_pair[k][2][1]),
            ),
        )
        # Both lists are sorted the same way before they are paired up, so a screw ends up in
        # roughly the position within the joint that it occupied in the original design.
        pos_sorted = sorted(positions, key=lambda p: (p[0], p[1]))

        for idx, (px, py) in zip(orig_order, pos_sorted):
            stored_idx, slot, _orig_xyz = screws_in_pair[idx]
            pos = np.array([px, py, upper_top_new], dtype=float)
            for prev in all_placed:
                d_xy = float(np.hypot(pos[0] - prev[0], pos[1] - prev[1]))
                d_z = abs(pos[2] - prev[2])
                # Two screws only conflict if they are close horizontally and at similar
                # heights: further apart vertically than a screw is long, they cannot meet.
                if d_xy < SCREW_SEPARATION_M and d_z < SCREW_LENGTH_M:
                    if (sx_max - sx_min) >= (sy_max - sy_min):
                        if pos[0] >= prev[0]:
                            pos[0] = min(sx_max, pos[0] + SCREW_SEPARATION_M)
                        else:
                            pos[0] = max(sx_min, pos[0] - SCREW_SEPARATION_M)
                    else:
                        if pos[1] >= prev[1]:
                            pos[1] = min(sy_max, pos[1] + SCREW_SEPARATION_M)
                        else:
                            pos[1] = max(sy_min, pos[1] - SCREW_SEPARATION_M)
            all_placed.append(pos)
            new_screws[stored_idx].append((slot, pos))

    return new_screws


_MODEL_CACHE: tuple | None = None


def _get_models_cached() -> tuple:
    """Load the surrogate once per process.

    The caller may repair many designs in a row, and reading the checkpoint each time would
    dominate the cost of a repair that otherwise takes seconds.
    """
    global _MODEL_CACHE
    if _MODEL_CACHE is None:
        device = torch.device("cpu")
        gnn_model, scale_factor = load_models(device)
        _MODEL_CACHE = (gnn_model, scale_factor, device)
    return _MODEL_CACHE


def repair_csv_string(input_csv: str) -> str:
    """Repair one design and return it in the form it arrived in.

    Args:
        input_csv: A CSV string with one data row, holding the position and edge lengths of
            each block, which blocks are joined, and the screw positions. Everything in
            metres.

    Returns:
        A CSV string with the same columns in the same order. Positions, sizes and screws are
        replaced; the connectivity is passed through, and screw slots the design does not use
        are left empty.
    """
    df = pd.read_csv(io.StringIO(input_csv))
    row = df.iloc[0]
    n = detect_n_blocks(df)
    blocks = parse_blocks(row, n)
    conns = parse_connections(row, n)
    conns_set = {(min(i, j), max(i, j)) for (i, j) in conns}

    resolved = resolve_penetrations(blocks, conns_set)
    snapped = apply_snaps(resolved, conns)

    gnn_model, scale_factor, device = _get_models_cached()
    repaired = gnn_repair_layered(
        snapped,
        conns,
        gnn_model,
        scale_factor,
        device,
    )

    # The original design is passed alongside the repaired one: it is what tells the placement
    # which joint each screw was intended for.
    orig_screws = parse_original_screws(row, n)
    new_screws = update_screws_for_repair(orig_screws, blocks, repaired, conns)

    out_row = row.copy()
    for bi, b in repaired.items():
        for k, ax in enumerate("XYZ"):
            out_row[f"Block{bi}_Pos{ax}"] = float(b["pos"][k])
            out_row[f"Block{bi}_Size{ax}"] = float(b["size"][k])
    # Cleared before the new screws are written, so a slot the repair no longer fills does not
    # keep a stale position from the input.
    for bi in range(n):
        for slot in range(5):
            for ax in "XYZ":
                col = f"Block{bi}_Screw{slot}_{ax}"
                if col in out_row.index:
                    out_row[col] = np.nan
    for bi, lst in new_screws.items():
        for slot, xyz in lst:
            for k, ax in enumerate("XYZ"):
                col = f"Block{bi}_Screw{slot}_{ax}"
                if col in out_row.index:
                    out_row[col] = float(xyz[k])

    return pd.DataFrame([out_row], columns=df.columns).to_csv(index=False, na_rep="NaN")


PIPELINE_DIR: Path = Path(__file__).resolve().parent


def _list_input_csvs() -> list[Path]:
    """CSVs next to this file that look like inputs rather than earlier outputs."""
    return sorted(
        p
        for p in PIPELINE_DIR.glob("*.csv")
        if "repaired" not in p.stem.lower() and "output" not in p.stem.lower()
    )


def _pick_input_csv() -> Path | None:
    """Ask which of the available inputs to repair, or nothing if there are none."""
    candidates = _list_input_csvs()
    if not candidates:
        print(f"No input CSVs found in {PIPELINE_DIR}/")
        print(
            "Drop a CSV next to repair_strategies.py "
            "(or pass the path as the first CLI argument)."
        )
        return None

    if len(candidates) == 1:
        only = candidates[0]
        print(f"Single input found in {PIPELINE_DIR.name}/: {only.name}")
        return only

    print(f"\nFound {len(candidates)} input CSV(s) in {PIPELINE_DIR.name}/:")
    for i, p in enumerate(candidates, start=1):
        print(f"  [{i:2d}]  {p.name}")
    print()

    while True:
        try:
            raw = input(
                f"Select input [1-{len(candidates)}] " f"or 'q' to quit: "
            ).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return None
        if raw.lower() in ("q", "quit", "exit", ""):
            return None
        try:
            idx = int(raw)
        except ValueError:
            print(f"  not a number — please enter 1-{len(candidates)} or 'q'.")
            continue
        if 1 <= idx <= len(candidates):
            return candidates[idx - 1]
        print(f"  out of range — please enter 1-{len(candidates)} or 'q'.")


def main() -> int:
    if len(sys.argv) < 2:
        in_path = _pick_input_csv()
        if in_path is None:
            return 1
    else:
        in_path = Path(sys.argv[1])
        if not in_path.exists():
            print(f"ERROR: {in_path} not found")
            return 1

    if len(sys.argv) >= 3:
        out_path = Path(sys.argv[2])
    else:
        out_path = in_path.parent / f"{in_path.stem}_repaired.csv"

    print(f"\nReading:  {in_path}")
    print(f"Writing:  {out_path}")
    print()

    out_csv = repair_csv_string(in_path.read_text())
    out_path.write_text(out_csv)
    print(f"\nDone — wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
