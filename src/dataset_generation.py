"""Generator that builds labelled block assemblies backwards, from the target metrics.

Rather than placing blocks at random and measuring what came out, this generator draws the two
criteria first and then constructs a geometry that realises them. That is what lets the
distribution be controlled: without it, a random placement almost always lands far from either
threshold, and a surrogate trained on such a set never sees the region where the decision is
actually made.

Each pair is drawn from one of three buckets per criterion, weighted towards the threshold, and
the realised values are recomputed from the constructed geometry rather than assumed, so a
target that could not be met is labelled by what was built and not by what was asked for. Only
overlap and thickness can fail here: every child is placed exactly on its parent's face and
above the table, so gap and table clearance are zero by construction and are not columns.

Written as a comma-separated table under the ``--out`` directory, with the same column layout
``src/simulation_dataset.py`` reads back, plus a sanity report beside it as JSON::

    python src/dataset_generation.py --num-blocks 2 --num-samples 500000 --out data
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np

# The failure codes are the simulator's own numbering and are not contiguous: the values in
# between belong to failure modes that cannot occur in this scope. They are kept as they are so
# a table written here can be read alongside a simulated one.
FAIL_NONE = 0
FAIL_OVERLAP = 2
FAIL_THICKNESS = 7

FAIL_REASON_NAMES: dict[int, str] = {
    FAIL_NONE: "NONE",
    FAIL_OVERLAP: "OVERLAP_INSUFFICIENT",
    FAIL_THICKNESS: "BLOCK_TOO_THICK",
}


# Everything below is in metres. The two thresholds restate config.gnn.thresh_overlap_min and
# config.gnn.thresh_thickness_max: a generator that read them from the configuration would
# relabel an existing dataset silently if the configuration were ever changed, so they are
# fixed here and have to be changed by hand together.
BLOCK_MIN_SIZE = 0.007
BLOCK_MAX_SIZE = 0.2
THICKNESS_THRESHOLD = 0.02
OVERLAP_THRESHOLD = 0.010

# The table's pose in the robot cell's frame. Blocks are constructed here and written out
# relative to it, which is what puts the table surface at height zero in the tables and in
# src/analytical_metrics.py.
TABLE_SURFACE_HEIGHT = 0.842
TABLE_CENTER_X = 0.4
TABLE_CENTER_Y = 0.6
# Six decimals is a micrometre, comfortably finer than the millimetre the criteria are stated
# in, so rounding cannot move a pair across a threshold.
METRIC_DECIMALS = 6
CLEARANCE_OFFSET = 0.001


@dataclass
class BucketConfig:
    """How far from its threshold each criterion is drawn, and how often.

    Both criteria use the same three-way split and differ only in their ranges, because
    overlap improves upwards and thickness downwards. A fifth of pairs are drawn on the failing
    side, three tenths just inside the feasible side, and the rest well clear of it.

    The failing side is split again so that a third of those failures sit just past the
    threshold rather than far beyond it. Together with the boundary bucket this concentrates
    the sample where the decision boundary is, which is the region the surrogate has to resolve
    and the region the repair has to cross. All ranges are in metres.
    """

    p_fail: float = 0.20
    p_boundary: float = 0.30
    p_near_within_fail: float = 0.35

    # Overlap fails below OVERLAP_THRESHOLD, so its failing ranges lie under it and its
    # feasible ranges above.
    overlap_fail_near: tuple[float, float] = (0.005, 0.010)
    overlap_fail_far: tuple[float, float] = (0.0, 0.005)
    overlap_boundary: tuple[float, float] = (0.010, 0.015)
    overlap_far: tuple[float, float] = (0.015, 0.035)

    # Thickness fails above THICKNESS_THRESHOLD, so the ordering is reversed. The far bucket
    # bottoms out at the smallest block the generator will build.
    thickness_fail_near: tuple[float, float] = (0.020, 0.024)
    thickness_fail_far: tuple[float, float] = (0.024, 0.040)
    thickness_boundary: tuple[float, float] = (0.016, 0.020)
    thickness_far: tuple[float, float] = (BLOCK_MIN_SIZE, 0.016)


@dataclass
class SamplingConfig:
    """The bucket settings together with the mix of contact faces."""

    bucket: BucketConfig = field(default_factory=BucketConfig)

    # Only five of the six faces are generated. Face 5 would hang the child underneath its
    # parent, where a small parent pushes it through the table, and table interference is
    # outside what this dataset is meant to cover. This value gives each remaining face an
    # equal share.
    p_vertical_face: float = 0.20


# The face code is 2 * axis + (0 for the positive side, 1 for the negative one), the same
# encoding src/analytical_metrics.py reads back.
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


# The overlap a pair can reach is capped by the smaller of the two blocks on that axis, so a
# small parent would silently pull a requested overlap down and the pair would be labelled with
# a value nobody asked for. This floor is high enough to realise every failing and every
# boundary target exactly. Targets in the far bucket above it can still be capped, which is
# accepted: those pairs are far from the threshold either way, and the label is always
# recomputed from the geometry.
BLOCK0_MIN_SIZE = 0.015


def _sample_block0_position(size_z: float) -> np.ndarray:
    """Place the first block centred on the table and resting on its surface."""
    return np.array(
        [
            TABLE_CENTER_X,
            TABLE_CENTER_Y,
            TABLE_SURFACE_HEIGHT + size_z / 2 + CLEARANCE_OFFSET,
        ]
    )


def _sample_block0_size() -> np.ndarray:
    """Draw the first block, whose lower bound is raised so it can support any overlap target.

    Restricting it does not bias the labels: the first block has no parent, so neither
    criterion is defined on it and its size enters only through what it lets its child reach.
    """
    return np.random.uniform(BLOCK0_MIN_SIZE, BLOCK_MAX_SIZE, size=3)


def _sample_bucket(cfg: BucketConfig) -> str:
    """Draw one of ``fail``, ``boundary`` or ``far``."""
    r = random.random()
    if r < cfg.p_fail:
        return "fail"
    if r < cfg.p_fail + cfg.p_boundary:
        return "boundary"
    return "far"


def _bucket_value(
    bucket: str,
    ranges: dict[str, tuple[float, float]],
    p_near_within_fail: float,
) -> float:
    """Draw a value in metres from the range the bucket names, splitting ``fail`` once more
    into a near and a far side."""
    if bucket == "fail":
        if random.random() < p_near_within_fail:
            return random.uniform(*ranges["fail_near"])
        return random.uniform(*ranges["fail_far"])
    if bucket == "boundary":
        return random.uniform(*ranges["boundary"])
    return random.uniform(*ranges["far"])


def _sample_pair_targets(cfg: SamplingConfig) -> dict:
    """Draw the overlap and thickness one pair should realise, and the buckets they came from.

    The two are drawn independently, so a pair may fail on either criterion, on both, or on
    neither. Coupling them would let the surrogate infer one from the other.
    """
    bcfg = cfg.bucket
    pn = bcfg.p_near_within_fail

    bucket_overlap = _sample_bucket(bcfg)
    bucket_thickness = _sample_bucket(bcfg)

    overlap_value = _bucket_value(
        bucket_overlap,
        {
            "fail_near": bcfg.overlap_fail_near,
            "fail_far": bcfg.overlap_fail_far,
            "boundary": bcfg.overlap_boundary,
            "far": bcfg.overlap_far,
        },
        p_near_within_fail=pn,
    )
    thickness_value = _bucket_value(
        bucket_thickness,
        {
            "fail_near": bcfg.thickness_fail_near,
            "fail_far": bcfg.thickness_fail_far,
            "boundary": bcfg.thickness_boundary,
            "far": bcfg.thickness_far,
        },
        p_near_within_fail=pn,
    )

    return {
        "overlap": overlap_value,
        "overlap_bucket": bucket_overlap,
        "thickness": thickness_value,
        "thickness_bucket": bucket_thickness,
    }


def _sample_face(cfg: SamplingConfig) -> int:
    """Draw a contact face from the five that are generated, omitting the downward one."""
    if random.random() < cfg.p_vertical_face:
        return 4
    return random.randint(0, 3)


def _construct_pair(
    face: int,
    size_0: np.ndarray,
    pos_0: np.ndarray,
    target_overlap: float,
    thickness: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Build the child that realises a given overlap and thickness against a fixed parent.

    Thickness is trivial: it is the child's edge length on the contact axis, so it is set
    directly. Overlap is harder, because it is the smaller of the two tangential overlaps and
    only one of them can be placed on the target. One tangential axis is therefore made the
    binding one and given exactly the target, and the other is deliberately given more so it
    cannot become the minimum instead.

    The child is placed touching its parent's face and lifted if it would otherwise reach below
    the table, so both of the criteria this dataset does not carry are zero by construction.

    Args:
        face: Contact face code from 0 to 5.
        size_0, pos_0: Full edge lengths and centre of the parent, ``(3,)`` each in metres.
        target_overlap: Overlap the pair should realise, in metres.
        thickness: Edge length the child should have along the contact axis, in metres.

    Returns:
        The child's full edge lengths and centre, ``(3,)`` each in metres. The realised metrics
        are measured afterwards rather than returned here, because a target that the parent is
        too small to support is not met.
    """
    contact_ax = _contact_axis(face)
    tang_axes = list(_tangential_axes(face))
    sign = _face_sign(face)

    size_1 = np.zeros(3)
    size_1[contact_ax] = thickness

    # Both tangential edges are drawn above the target. A child narrower than the target on
    # either axis could not overlap that far however it were placed, and the pair would land in
    # the failing range without having been drawn there.
    tang_floor = max(BLOCK_MIN_SIZE, min(target_overlap, BLOCK_MAX_SIZE))
    for ax in tang_axes:
        size_1[ax] = random.uniform(tang_floor, BLOCK_MAX_SIZE)

    pos_1 = pos_0.copy()
    pos_1[contact_ax] = pos_0[contact_ax] + sign * (
        size_0[contact_ax] / 2 + size_1[contact_ax] / 2
    )

    # For a sideways face the vertical axis is tangential and may be raised by the table clamp
    # below, which would move the overlap after it had been set. The binding axis is therefore
    # never the vertical one unless the contact is vertical, in which case no tangential axis
    # is affected by the clamp and either will do.
    if contact_ax != 2:
        binding_ax = next(a for a in tang_axes if a != 2)
    else:
        binding_ax = random.choice(tang_axes)
    other_ax = next(a for a in tang_axes if a != binding_ax)

    # Offsetting to either side with equal probability, rather than always the same way, keeps
    # the sign of the tangential displacement from correlating with the overlap.
    max_o_binding = min(size_0[binding_ax], size_1[binding_ax])
    if max_o_binding >= target_overlap:
        delta_abs = (size_0[binding_ax] + size_1[binding_ax]) / 2 - target_overlap
        pos_1[binding_ax] = pos_0[binding_ax] + random.choice([-1.0, 1.0]) * delta_abs
        binding_realized = target_overlap
    else:
        # The parent is too narrow for the target. Centring the blocks yields the largest
        # overlap this pair can reach, and the label recorded later says what that was.
        pos_1[binding_ax] = pos_0[binding_ax]
        binding_realized = max_o_binding

    # The second axis is given strictly more than the first, by a margin far below the
    # micrometre the labels are rounded to, so that the binding axis stays the minimum and the
    # realised overlap is the value that was drawn.
    max_o_other = min(size_0[other_ax], size_1[other_ax])
    if max_o_other > binding_realized + 1e-5:
        other_realized = random.uniform(binding_realized + 1e-5, max_o_other)
        delta_abs = (size_0[other_ax] + size_1[other_ax]) / 2 - other_realized
        pos_1[other_ax] = pos_0[other_ax] + random.choice([-1.0, 1.0]) * delta_abs
    else:
        # This axis cannot exceed the binding one, so centring gives it its own maximum and the
        # two simply tie.
        pos_1[other_ax] = pos_0[other_ax]

    # Only a sideways contact can push the child below the table, since a child on top of its
    # parent is already above it. Raising it here can only increase the vertical overlap, and
    # the vertical axis was excluded from the binding choice for exactly that reason.
    if contact_ax != 2:
        min_pos_z = TABLE_SURFACE_HEIGHT + size_1[2] / 2 + CLEARANCE_OFFSET
        if pos_1[2] < min_pos_z:
            pos_1[2] = min_pos_z

    return size_1, pos_1


def _compute_planned_overlap(pos_p, size_p, pos_c, size_c, face):
    """Measure the realised overlap in metres, as the smaller of the two tangential overlaps."""
    ov = np.zeros(3)
    for d in range(3):
        min1, max1 = pos_p[d] - size_p[d] / 2, pos_p[d] + size_p[d] / 2
        min2, max2 = pos_c[d] - size_c[d] / 2, pos_c[d] + size_c[d] / 2
        ov[d] = max(0.0, min(max1, max2) - max(min1, min2))
    tang = _tangential_axes(face)
    return float(min(ov[tang[0]], ov[tang[1]]))


def _classify_from_metrics(overlap: float, thickness: float) -> int:
    """Reduce the two measured criteria to one failure code.

    A pair that violates both is recorded as an overlap failure, since only one code fits in
    the column. The two metric columns are written out unreduced, so nothing is lost.
    """
    if overlap < OVERLAP_THRESHOLD:
        return FAIL_OVERLAP
    if thickness > THICKNESS_THRESHOLD:
        return FAIL_THICKNESS
    return FAIL_NONE


def generate_assembly(num_blocks: int, cfg: SamplingConfig) -> dict:
    """Generate one assembly as a chain, each block placed against the one before it.

    Returns:
        A dictionary with ``positions`` and ``sizes``, one ``(3,)`` array per block in metres;
        ``faces`` and ``fail_reasons``, one entry per block, with -1 and no failure for the
        first block, which has no parent; and ``overlap``, ``thickness`` and ``bucket_labels``,
        one entry per pair and so one shorter than the rest. The metrics are the realised
        values, measured from the constructed geometry.
    """
    positions: list[np.ndarray] = []
    sizes: list[np.ndarray] = []
    faces: list[int] = []
    fail_reasons: list[int] = []
    bucket_labels: list[dict] = []

    metric_overlap: list[float] = []
    metric_thickness: list[float] = []

    # The first block has no parent, so neither criterion applies to it. Its face is recorded
    # as -1 and it never fails.
    size_0 = _sample_block0_size()
    pos_0 = _sample_block0_position(size_0[2])
    sizes.append(size_0)
    positions.append(pos_0)
    faces.append(-1)
    fail_reasons.append(FAIL_NONE)

    # Each block is built against its immediate predecessor, so a chain of any length is a
    # sequence of independent pairs. This is the same decomposition the repair walks.
    for b in range(1, num_blocks):
        targets = _sample_pair_targets(cfg)
        face = _sample_face(cfg)

        size_b, pos_b = _construct_pair(
            face=face,
            size_0=sizes[b - 1],
            pos_0=positions[b - 1],
            target_overlap=targets["overlap"],
            thickness=targets["thickness"],
        )

        sizes.append(size_b)
        positions.append(pos_b)
        faces.append(face)
        bucket_labels.append(
            {
                "overlap": targets["overlap_bucket"],
                "thickness": targets["thickness_bucket"],
                "face": face,
            }
        )

        # Both labels come from the geometry that was built, never from the targets that were
        # drawn. A target the parent was too small to support is therefore labelled honestly.
        overlap_real = _compute_planned_overlap(
            positions[b - 1], sizes[b - 1], pos_b, size_b, face
        )
        contact_ax = _contact_axis(face)
        thickness_real = float(size_b[contact_ax])

        overlap_real = round(overlap_real, METRIC_DECIMALS)
        thickness_real = round(thickness_real, METRIC_DECIMALS)

        metric_overlap.append(overlap_real)
        metric_thickness.append(thickness_real)

        fail_reasons.append(
            _classify_from_metrics(overlap=overlap_real, thickness=thickness_real)
        )

    return {
        "positions": positions,
        "sizes": sizes,
        "faces": faces,
        "fail_reasons": fail_reasons,
        "bucket_labels": bucket_labels,
        "overlap": metric_overlap,
        "thickness": metric_thickness,
    }


def _build_header(num_blocks: int) -> list[str]:
    """Build the column names, grouped per block and then per column family."""
    cols = []
    for b in range(num_blocks):
        cols += [
            f"Block{b}_PosX",
            f"Block{b}_PosY",
            f"Block{b}_PosZ",
            f"Block{b}_SizeX",
            f"Block{b}_SizeY",
            f"Block{b}_SizeZ",
        ]
    cols += [f"Block{b}_Failed?" for b in range(num_blocks)]
    cols += [f"Block{b}_FailureReason" for b in range(num_blocks)]
    # The metric and face columns start at block 1: they describe a joint, and the first block
    # has none. This offset is why the per-pair lists are one shorter than the per-block ones.
    cols += [f"Block{b}_Overlap_m" for b in range(1, num_blocks)]
    cols += [f"Block{b}_Thickness_m" for b in range(1, num_blocks)]
    cols += [f"Block{b}_Face" for b in range(1, num_blocks)]
    cols += ["Assembly_Good?"]
    return cols


def _row_from_assembly(assembly: dict, num_blocks: int) -> list[str]:
    """Format one assembly as a row, converting its positions to the table-relative frame."""
    row: list[str] = []
    for b in range(num_blocks):
        pos = assembly["positions"][b]
        size = assembly["sizes"][b]
        # Positions are constructed in the cell's frame and written relative to the table, so
        # the tables put the table centre at the origin and its surface at height zero. Every
        # consumer of these files assumes that frame.
        px = pos[0] - TABLE_CENTER_X
        py = pos[1] - TABLE_CENTER_Y
        pz = pos[2] - TABLE_SURFACE_HEIGHT
        row += [f"{px:.6f}", f"{py:.6f}", f"{pz:.6f}"]
        row += [f"{size[0]:.6f}", f"{size[1]:.6f}", f"{size[2]:.6f}"]
    for b in range(num_blocks):
        row.append(str(assembly["fail_reasons"][b] != FAIL_NONE))
    for b in range(num_blocks):
        row.append(FAIL_REASON_NAMES[assembly["fail_reasons"][b]])
    for b in range(1, num_blocks):
        row.append(f"{assembly['overlap'][b - 1]:.6f}")
    for b in range(1, num_blocks):
        row.append(f"{assembly['thickness'][b - 1]:.6f}")
    for b in range(1, num_blocks):
        row.append(str(assembly["faces"][b]))
    # An assembly counts as good only if every one of its pairs does; one bad joint condemns
    # the whole design.
    assembly_good = all(r == FAIL_NONE for r in assembly["fail_reasons"])
    row.append(str(assembly_good))
    return row


# The two functions below re-derive the bucket from a realised value, so the report measures
# what was built rather than repeating what was requested. Their boundaries repeat the upper
# end of the overlap boundary range and the lower end of the thickness one.
def _bucket_overlap(v: float) -> str:
    if v < OVERLAP_THRESHOLD:
        return "fail"
    if v < 0.015:
        return "boundary"
    return "far"


def _bucket_thickness(v: float) -> str:
    if v > THICKNESS_THRESHOLD:
        return "fail"
    if v > 0.016:
        return "boundary"
    return "far"


def build_sanity_report(
    overlap_arr: np.ndarray,
    thickness_arr: np.ndarray,
    face_arr: np.ndarray,
) -> dict:
    """Summarise what was actually generated, so the realised distribution can be checked.

    Reports how often each criterion failed, the class imbalance that follows from it, the
    bucket shares as built, the correlation between the two failure modes, and the face mix.
    The correlation is the one to watch: the two criteria are drawn independently, so a value
    far from zero would mean the construction coupled them after the fact.

    Args:
        overlap_arr, thickness_arr: Realised metrics in metres, one entry per pair.
        face_arr: Contact face code per pair.

    Returns:
        A dictionary of the statistics, written beside the dataset as JSON. ``pos_weights`` is
        the ratio of feasible to failing pairs, the weight a balanced loss would need.
    """
    overlap_fail = (overlap_arr < OVERLAP_THRESHOLD).astype(np.float32)
    thickness_fail = (thickness_arr > THICKNESS_THRESHOLD).astype(np.float32)

    def _pw(fail_rate: float) -> float:
        return (1.0 - fail_rate) / max(fail_rate, 1e-9)

    fail_rates = {
        "overlap": float(overlap_fail.mean()),
        "thickness": float(thickness_fail.mean()),
    }
    pos_weights = {k: _pw(v) for k, v in fail_rates.items()}

    def _bucket_marginals(arr: np.ndarray, fn) -> dict:
        labels = np.array([fn(v) for v in arr])
        n = len(labels)
        return {
            "fail": float((labels == "fail").sum() / n),
            "boundary": float((labels == "boundary").sum() / n),
            "far": float((labels == "far").sum() / n),
        }

    buckets = {
        "overlap": _bucket_marginals(overlap_arr, _bucket_overlap),
        "thickness": _bucket_marginals(thickness_arr, _bucket_thickness),
    }

    fails = np.stack([overlap_fail, thickness_fail], axis=1)
    corr = np.corrcoef(fails.T)
    pairwise = {
        "overlap_thickness": float(corr[0, 1]),
    }

    face_counts: dict = {}
    for f in range(6):
        face_counts[f] = int((face_arr == f).sum())

    # An assembly fails if either criterion does, so its failure rate is higher than either
    # alone. This is the rate the feasibility head has to reproduce.
    assembly_fail = overlap_fail.astype(bool) | thickness_fail.astype(bool)
    assembly_fail_rate = float(assembly_fail.mean())

    return {
        "n_child_blocks": int(len(overlap_arr)),
        "fail_rates": fail_rates,
        "pos_weights": pos_weights,
        "bucket_marginals": buckets,
        "pairwise_correlations": pairwise,
        "face_counts": face_counts,
        "assembly_fail_rate": assembly_fail_rate,
        "assembly_pos_weight": _pw(assembly_fail_rate),
    }


def print_sanity_report(report: dict) -> None:
    """Print the report, flagging any figure that has drifted from what was configured."""
    print("\n" + "=" * 64)
    print(" DATASET SANITY REPORT")
    print("=" * 64)
    print(f" Child blocks analyzed: {report['n_child_blocks']:,}")
    print()

    print(" Per-mode fail rates and pos_weights")
    print(" ─────────────────────────────────────────────────")
    print(f" {'Mode':<16} {'Fail %':>9} {'pos_weight':>12}   Status")
    for mode in ["overlap", "thickness"]:
        fp = 100 * report["fail_rates"][mode]
        pw = report["pos_weights"][mode]
        # The band brackets the configured failure share; a realised rate outside it means the
        # construction could not meet the targets it was given.
        ok = "OK " if 17.5 <= fp <= 22.5 and 3.5 <= pw <= 4.7 else "OFF"
        print(f" {mode:<16} {fp:>8.2f}% {pw:>12.3f}   [{ok}]")
    print()

    print(" Bucket marginals (target: fail 20% / boundary 30% / far 50%)")
    print(" ─────────────────────────────────────────────────")
    for mode, b in report["bucket_marginals"].items():
        print(
            f" {mode:<16} fail {100*b['fail']:>5.1f}%  "
            f"boundary {100*b['boundary']:>5.1f}%  far {100*b['far']:>5.1f}%"
        )
    print()

    print(" Pairwise mode correlations (target |r| < 0.15)")
    print(" ─────────────────────────────────────────────────")
    for pair, val in report["pairwise_correlations"].items():
        flag = "OK " if abs(val) < 0.15 else "OFF"
        print(f" {pair:<22} r = {val:>+.4f}   [{flag}]")
    print()

    print(" Face distribution")
    print(" ─────────────────────────────────────────────────")
    n_total = sum(report["face_counts"].values())
    face_names = {0: "+X", 1: "-X", 2: "+Y", 3: "-Y", 4: "+Z", 5: "-Z"}
    for f, count in sorted(report["face_counts"].items()):
        pct = 100 * count / max(n_total, 1)
        print(f" face {f} ({face_names[f]}): {count:>9,}  ({pct:>5.1f}%)")
    print()

    print(" Assembly-level (Assembly_Good)")
    print(" ─────────────────────────────────────────────────")
    print(f" Fail rate (= infeasible):  {100*report['assembly_fail_rate']:>5.2f}%")
    print(f" pos_weight (Assembly_Good): {report['assembly_pos_weight']:.3f}")
    print("=" * 64)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-blocks", type=int, default=2)
    parser.add_argument("--num-samples", type=int, default=2500_000)
    parser.add_argument("--out", type=str, default="data")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Only generate 10k samples for sanity check, do not write file",
    )
    args = parser.parse_args()

    # Both generators are seeded: the buckets are drawn with the standard library's and the
    # sizes with NumPy's, so seeding one alone would leave half the dataset irreproducible.
    random.seed(args.seed)
    np.random.seed(args.seed)

    cfg = SamplingConfig()

    n_samples = 10_000 if args.report_only else args.num_samples
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / (
        f"synth_num_blocks_{args.num_blocks}_samples_{n_samples}_"
        + datetime.now().strftime("%Y%m%d_%H%M")
        + ".txt"
    )

    print(f"Generating {n_samples:,} assemblies (num_blocks={args.num_blocks})")
    if not args.report_only:
        print(f"Output path: {out_path}")
    print(
        f"Bucket config: fail={cfg.bucket.p_fail:.2f}  "
        f"boundary={cfg.bucket.p_boundary:.2f}  "
        f"far={1.0 - cfg.bucket.p_fail - cfg.bucket.p_boundary:.2f}  "
        f"(p_near_within_fail={cfg.bucket.p_near_within_fail:.2f})"
    )

    header = _build_header(args.num_blocks)

    overlap_all: list[float] = []
    thickness_all: list[float] = []
    face_all: list[int] = []

    # Rows are streamed to disk as they are generated rather than collected first, so the
    # sample count is limited by disk rather than by memory. Only the two metric columns and
    # the face are kept in memory, for the report.
    f_out = None if args.report_only else open(out_path, "w")
    if f_out is not None:
        f_out.write(",".join(header) + "\n")

    try:
        for i in range(n_samples):
            assembly = generate_assembly(args.num_blocks, cfg)
            if f_out is not None:
                row = _row_from_assembly(assembly, args.num_blocks)
                f_out.write(",".join(row) + "\n")

            for b in range(1, args.num_blocks):
                overlap_all.append(assembly["overlap"][b - 1])
                thickness_all.append(assembly["thickness"][b - 1])
                face_all.append(assembly["faces"][b])

            if (i + 1) % 50_000 == 0:
                print(f"  {i + 1:>7,}/{n_samples:,}")
    finally:
        if f_out is not None:
            f_out.close()

    if not args.report_only:
        print(f"\nDone. Wrote {n_samples:,} samples → {out_path}")

    report = build_sanity_report(
        np.array(overlap_all, dtype=np.float32),
        np.array(thickness_all, dtype=np.float32),
        np.array(face_all, dtype=np.int8),
    )
    print_sanity_report(report)

    if not args.report_only:
        report_path = out_path.with_suffix(".sanity.json")
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        print(f" Sanity report saved: {report_path}")


if __name__ == "__main__":
    main()
