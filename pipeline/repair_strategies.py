"""Repair one whole design: a single-row CSV of blocks and connections in, the repaired CSV out.

This is the level above :mod:`src.repair_process`. That module repairs one chain of pairs; this
one turns a design into those chains, runs them, and writes the result back into the schema the
generative stage produced, with the same columns in the same order. Everything in the CSV is in
metres. ``tools/repair_csv_sweep.py`` applies it to the whole evaluation set.

Four stages, in this order. Two closed-form corrections come first, because a design in which
blocks interpenetrate or float is not a configuration the surrogate was trained on: every
interpenetrating pair is shrunk until it only touches, then every connected pair is snapped
face to face. The repair itself then works upwards in layers from the blocks resting on the
table, each already repaired block acting as the frozen parent of the blocks above it. Finally
the screws are placed inside the contact rectangle the repair produced.

Run as ``python pipeline/repair_strategies.py <input.csv> [<output.csv>]``. With no argument it
lists the CSVs beside it and asks which to use; with no output path it writes
``<input>_repaired.csv`` next to the input.
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

# The thickness the repair is held to, in metres. Assigning it replaces the module-level
# constant that src.repair_optimizer reads for its early-stop test and for the endpoint of its
# threshold relaxation, which is how this pipeline can hold the optimiser to a stricter target
# than the surrogate was trained under without editing the configuration. At the value set here
# it coincides with config.gnn.thresh_thickness_max, so nothing is in fact tightened during the
# optimisation; the margin against the surrogate's error at the threshold is taken afterwards
# instead, by POST_SHRINK_M below.
REPAIR_TARGET_THICKNESS_M = 0.02
_repair_optimizer.THRESH_THICKNESS = REPAIR_TARGET_THICKNESS_M

# A block whose underside lies within a millimetre of the table counts as resting on it and
# becomes a starting point of the layered repair. Testing for exact contact would fail on
# designs whose coordinates carry rounding from the generative stage.
FLOOR_TOLERANCE_M = 1e-3

# Screw geometry, in metres. The length enters only as the vertical range over which two screws
# can foul each other; the inset keeps a screw off the edge of the contact rectangle; the
# separation is the smallest horizontal distance allowed between two screws.
SCREW_LENGTH_M = 0.022
SCREW_SAFETY_M = 0.005
SCREW_SEPARATION_M = 0.010

# One millimetre of thickness given up after the repair has already been accepted. The
# surrogate is not exact at the threshold, so a pair it reports as just thin enough can still be
# over the limit once the closed-form formulas are applied to the geometry. Taking a millimetre
# off the contact axis and putting the child back in contact buys a margin against that error,
# and it touches neither the tangential position nor the tangential edge lengths, so the overlap
# is left exactly as the repair produced it. The result is clamped at SIZE_MIN_M, the same
# absolute floor the optimiser enforces on any edge length.
POST_SHRINK_M = 0.001
SIZE_MIN_M = 0.008


def detect_n_blocks(df: pd.DataFrame) -> int:
    """Count the block slots the schema provides, by walking ``Block0``, ``Block1`` and so on.

    Counting stops at the first missing index, so the schema has to number its blocks
    contiguously from zero. A slot that exists but is empty is still counted here and dropped
    later by :func:`parse_blocks`.
    """
    n = 0
    while f"Block{n}_PosX" in df.columns:
        n += 1
    return n


def parse_blocks(row: pd.Series, n: int) -> dict[int, dict]:
    """Read the blocks of one design row into ``{index: {"pos": (3,), "size": (3,)}}``.

    Both arrays are in metres: the centre of the block and its full edge lengths, not its
    half-extents. Slots whose coordinates are empty are skipped rather than defaulted, so the
    returned keys are the blocks that actually exist and need not be contiguous.
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
    """Read the adjacency of one design row as a list of pairs.

    Only the upper triangle of the matrix is read, so every connection appears once and always
    with the lower index first. The direction of a joint is not stated in the schema; which
    block is the parent and which the part under test is decided later, by the layered order.
    """
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
    """Read the screws of one design row, keyed by the block whose columns hold them.

    Returns ``{block: [(slot, (3,) position in metres), ...]}`` for every screw that is filled
    in, with an empty list for a block that has none. The block a screw is stored under is not
    necessarily the pair it belongs to; :func:`update_screws_for_repair` works that out from
    the geometry, and the slot is carried along so the repaired screw is written back into the
    column it came from.
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
    """Signed overlap of two blocks per axis, ``(3,)`` in metres, negative where they are apart."""
    delta = np.abs(b_i["pos"] - b_j["pos"])
    return (b_i["size"] + b_j["size"]) / 2.0 - delta


def resolve_penetrations(
    blocks: dict[int, dict], conns_set: set[tuple[int, int]]
) -> dict[int, dict]:
    """Separate any two blocks that occupy the same space, by shrinking both.

    A pair overlapping on all three axes shares a volume, which no arrangement of real parts
    can. The two blocks each give up half the penetration depth on one axis and move apart by
    half of that again, so the shared volume collapses to a contact and the midpoint of the
    joint stays where it was. Every pair is treated, whether the design declares them connected
    or not, because an undeclared overlap is just as impossible to build.

    Args:
        blocks: Blocks as returned by :func:`parse_blocks`, in metres.
        conns_set: The declared connections, each with the lower index first.

    Returns:
        A new mapping in the same form; the input is not modified.
    """
    new = {
        k: {"pos": v["pos"].copy(), "size": v["size"].copy()} for k, v in blocks.items()
    }
    keys = sorted(new.keys())
    # One pair is resolved per pass and the overlaps are recomputed, because shrinking a block
    # changes every pair it takes part in. The cap stops a design in which two corrections keep
    # undoing each other from looping without end.
    for _ in range(50):
        candidates: list[tuple[int, int, np.ndarray]] = []
        for ii, i in enumerate(keys):
            for j in keys[ii + 1 :]:
                ov = _aabb_overlap_xyz(new[i], new[j])
                if (ov > 1e-9).all():
                    candidates.append((i, j, ov))
        if not candidates:
            break
        # The deepest penetration is dealt with first, and on the axis where the two blocks
        # overlap least, which is the axis on which the least material has to be removed.
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
    """Bring every connected pair into exact face-to-face contact, moving one block per joint.

    The design may declare two blocks joined while leaving a gap between them, and the
    surrogate would then be asked about a pair that does not touch. Walking the connection
    graph breadth-first fixes each joint once, and because a block is only ever moved when it
    is first reached, an earlier contact is never broken by a later one.

    The walk starts at the block of largest volume, which is the one most likely to carry the
    design and least appropriate to move. Sizes are not changed here, only positions.

    Args:
        blocks: Blocks as returned by :func:`parse_blocks`, in metres.
        conns: The declared connections.

    Returns:
        A new mapping in the same form; the input is not modified.
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
        # The contact axis is left to be inferred from the geometry: at this point nothing has
        # yet decided which face of the parent the child belongs on.
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
    """Wrap a centre and edge lengths given in metres into a block in the scaled frame."""
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
    """Take ``shrink_m`` metres off the child's contact axis and put its face back against the
    parent, leaving the tangential position and edge lengths, and therefore the overlap,
    untouched."""
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
    """Repair the design layer by layer, working upwards from the blocks resting on the table.

    The blocks on the table form the first layer and are never modified. Every block connected
    to a block of the current layer is repaired against it, with the parent frozen, and then
    becomes a parent of the next layer. Each repaired child is written straight back into the
    working state, so a block repaired late is judged against where its parent actually ended
    up rather than where the design put it.

    Each pair goes through the same two steps: the child is snapped onto its parent's contact
    face before the surrogate ever sees it, and after the repair a millimetre of thickness is
    given up by :func:`_post_shrink_and_snap`. The repair itself is the latent-space one, which
    holds the child's learned code rather than its three edge lengths and recovers a size
    through the frozen box decoder of :mod:`src.dec_box`; nothing here touches the auxiliary
    signed-distance decoder of stage 1.

    Args:
        blocks: Blocks in metres, already penetration-resolved and snapped.
        conns: The declared connections.
        gnn_model: The surrogate, which alone decides whether a pair is accepted.
        scale_factor: Factor between the scaled frame the repair works in and metres.
        device: Device the tensors live on.
        freeze_pos_if_overlap_ok: Passed through to the repair; keeps a child's position fixed
            when its overlap already passes.
        post_shrink_m: Thickness given up after a successful repair, in metres; 0 disables it.

    Returns:
        A new mapping in the same form as ``blocks``, in metres.
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
    # A design in which nothing touches the table would otherwise have no layer to start from
    # and no block would be repaired at all. The largest block stands in as the anchor.
    if not floor:
        floor = {max(blocks, key=lambda i: float(np.prod(blocks[i]["size"])))}

    committed: set[int] = set(floor)
    current_layer: set[int] = set(floor)
    while current_layer:
        next_layer: set[int] = set()
        for parent in sorted(current_layer):
            for child in sorted(adj[parent]):
                # A block with several parents is repaired against the first of them only, in
                # index order. This is the limit of the pairwise decomposition: satisfying a
                # second parent afterwards would overwrite the first repair, and each pass
                # would add drift without ever satisfying both at once.
                if child in committed or child in next_layer:
                    continue
                prev = bd[parent]
                curr = bd[child]

                # The parent may have been shrunk while it was itself repaired, which leaves a
                # gap under a child that was in contact before. Closing it here means the
                # surrogate is asked about a pair that touches, so its overlap prediction
                # describes the real geometry and the position freeze acts on a true reading.
                sf = float(scale_factor)
                pos_p_m = prev.pos.cpu().numpy() / sf
                he_p_m = (prev.size / 2.0).cpu().numpy() / sf
                pos_c_m = curr.pos.cpu().numpy() / sf
                he_c_m = (curr.size / 2.0).cpu().numpy() / sf
                face = infer_face(pos_p_m, he_p_m, pos_c_m, he_c_m)
                # The face code is 2 * axis + side, so integer division recovers the axis. It
                # is fixed here and reused for the shrink after the repair, so that the two
                # cannot end up working on different axes.
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
                # The geometry the repair ended on is written back whether or not the surrogate
                # accepted it: ``step.repaired`` is deliberately not consulted. A design has to
                # be written out complete, and the best attempt at a pair is a better thing to
                # hand on than the infeasible original. Whether a pair was in fact repaired is
                # settled afterwards by the closed-form check, not here.
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
    """Spread ``n`` points over a rectangle: one at the centre, two at the ends of its longer
    side, three or four at its corners, and more on a regular grid."""
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
    """Move every screw of the design into the contact rectangle its pair now has.

    The repair changes both the size and the position of a block, so a screw placed against the
    original geometry can end up outside the joint it was meant to fasten. Each screw is first
    matched to the pair it belongs to, using the geometry as designed, and then placed inside
    that pair's overlap rectangle as repaired, inset by ``SCREW_SAFETY_M`` from its edges.

    A screw is put at the top face of the upper block of its pair. Screws already placed are
    kept clear of: a new screw closer than ``SCREW_SEPARATION_M`` horizontally and within
    ``SCREW_LENGTH_M`` vertically is pushed along the longer side of the rectangle, and stays
    inside it.

    Args:
        orig_screws: Screws as returned by :func:`parse_original_screws`, in metres.
        orig_blocks: The design as it came in, used only to decide which pair a screw belongs to.
        repaired_blocks: The design after the repair, which the new positions are computed from.
        conns: The declared connections.

    Returns:
        The same mapping shape as ``orig_screws``, with each screw kept in the slot it came
        from. A screw whose pair has no repaired geometry is passed through unchanged.
    """

    def _xy_dist_to_rect(
        px: float, py: float, x_min: float, x_max: float, y_min: float, y_max: float
    ) -> float:
        """Horizontal distance from a point to a rectangle, zero when the point is inside it."""
        dx = max(x_min - px, 0.0, px - x_max)
        dy = max(y_min - py, 0.0, py - y_max)
        return float(np.hypot(dx, dy))

    pair_to_screws: dict[tuple[int, int], list[tuple[int, int, np.ndarray]]] = {}
    for stored_block_idx, screw_list in orig_screws.items():
        for slot, orig_xyz in screw_list:
            # The schema does not record which joint a screw fastens, so it is recovered from
            # where the screw was: the pair whose upper face it sits closest to, vertically and
            # horizontally combined. The two distances are simply added, both being lengths in
            # metres, which is enough to separate the candidates in practice.
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
        # An overlap narrower than twice the inset leaves nothing between the two edges. The
        # centre line is then the best available compromise, rather than a rectangle turned
        # inside out.
        if sx_min > sx_max:
            sx_min = sx_max = (ox_min + ox_max) / 2.0
        if sy_min > sy_max:
            sy_min = sy_max = (oy_min + oy_max) / 2.0
        upper_top_new = up["pos"][2] + up["size"][2] / 2.0

        n_screws = len(screws_in_pair)
        positions = _distribute_in_rect(sx_min, sx_max, sy_min, sy_max, n_screws)
        # Both the screws and the new positions are sorted the same way before being paired up,
        # so a screw keeps its neighbours: the leftmost screw of a joint stays the leftmost.
        orig_order = sorted(
            range(n_screws),
            key=lambda k: (
                float(screws_in_pair[k][2][0]),
                float(screws_in_pair[k][2][1]),
            ),
        )
        pos_sorted = sorted(positions, key=lambda p: (p[0], p[1]))

        for idx, (px, py) in zip(orig_order, pos_sorted):
            stored_idx, slot, _orig_xyz = screws_in_pair[idx]
            pos = np.array([px, py, upper_top_new], dtype=float)
            for prev in all_placed:
                d_xy = float(np.hypot(pos[0] - prev[0], pos[1] - prev[1]))
                d_z = abs(pos[2] - prev[2])
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
    """Load the surrogate and the scale factor once per process and reuse them for every design.

    :func:`src.repair_process.load_models` also warms the caches of the box encoder and the box
    decoder, so a sweep over many designs reads none of the three from disk more than once.
    """
    global _MODEL_CACHE
    if _MODEL_CACHE is None:
        device = torch.device("cpu")
        gnn_model, scale_factor = load_models(device)
        _MODEL_CACHE = (gnn_model, scale_factor, device)
    return _MODEL_CACHE


def repair_csv_string(input_csv: str) -> str:
    """Repair one design and return it in the schema it arrived in.

    The public entry point of this module and the unit the evaluation is run over.

    Args:
        input_csv: A CSV with one data row, using the columns ``BlockN_PosX/Y/Z`` for centres,
            ``BlockN_SizeX/Y/Z`` for full edge lengths, ``BlockI_ConnectsTo_BlockJ`` for
            adjacency and ``BlockN_ScrewK_X/Y/Z`` for screws. Every value is in metres.

    Returns:
        A CSV with the same columns in the same order. Centres, edge lengths and screws are
        replaced by their repaired values; the adjacency columns are passed through untouched,
        and screw slots that stay unused are written as ``NaN``.
    """
    df = pd.read_csv(io.StringIO(input_csv))
    row = df.iloc[0]
    n = detect_n_blocks(df)
    blocks = parse_blocks(row, n)
    conns = parse_connections(row, n)
    conns_set = {(min(i, j), max(i, j)) for (i, j) in conns}

    # The two closed-form corrections run before the surrogate is consulted at all: a design
    # with interpenetrating or floating blocks is not something it was trained on, and its
    # prediction there would say little.
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

    # The screws are matched against the design as it came in, before any correction, because
    # that is the geometry they were placed against.
    orig_screws = parse_original_screws(row, n)
    new_screws = update_screws_for_repair(orig_screws, blocks, repaired, conns)

    out_row = row.copy()
    for bi, b in repaired.items():
        for k, ax in enumerate("XYZ"):
            out_row[f"Block{bi}_Pos{ax}"] = float(b["pos"][k])
            out_row[f"Block{bi}_Size{ax}"] = float(b["size"][k])
    # Every screw column is cleared before the new values are written, so that a screw the
    # repair did not reproduce leaves an empty slot rather than its stale original position.
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
    """The CSVs beside this file that look like inputs, so that a previously written result is
    not offered as one."""
    return sorted(
        p
        for p in PIPELINE_DIR.glob("*.csv")
        if "repaired" not in p.stem.lower() and "output" not in p.stem.lower()
    )


def _pick_input_csv() -> Path | None:
    """Ask which of the available input CSVs to repair, returning ``None`` if the choice is
    abandoned or there is nothing to offer."""
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
