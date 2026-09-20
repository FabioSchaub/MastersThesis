"""Construct labelled configurations directly, without going through the simulator.

Rather than draw a pair of boxes at random and see what comes out, this generator picks the
two measured quantities first and then builds a pair that realises them. That is what makes
the dataset controllable: the density of examples near each limit is a parameter rather than
an accident, and around a fifth of the pairs violate each criterion instead of the small
fraction that random geometry would produce.

Feasibility follows from the same two criteria the rest of Part I uses, and the two remaining
failure modes of the simulator, table interference and a gap at the joint, are excluded by
construction: the child is placed in contact with its parent and clamped above the table.

A report is printed after every run and saved next to the data. It compares the realised
distribution against the intended one, which is the only way to notice that the construction
has failed to reach a target -- something that does happen, because a target overlap larger
than the parent's own extent cannot be realised.

    python src/dataset_generation.py --num-blocks 2 --num-samples 500000 --out data/

Writes a comma-separated table to the output directory, with the report beside it as JSON.
Passing ``--report-only`` prints the report for a small draw and writes nothing.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np

# The codes are those of the simulator, not a fresh enumeration, so that a table written here
# and a table exported from the simulator can be concatenated.
FAIL_NONE = 0
FAIL_OVERLAP = 2
FAIL_THICKNESS = 7

FAIL_REASON_NAMES: dict[int, str] = {
    FAIL_NONE: "NONE",
    FAIL_OVERLAP: "OVERLAP_INSUFFICIENT",
    FAIL_THICKNESS: "BLOCK_TOO_THICK",
}


# Edge lengths in metres a block may take, and the two screwdriving limits. All four repeat
# values held elsewhere -- the simulator's own configuration and ``config/config.yaml`` -- and
# have to be kept in step with them by hand.
BLOCK_MIN_SIZE = 0.007
BLOCK_MAX_SIZE = 0.2
THICKNESS_THRESHOLD = 0.02
OVERLAP_THRESHOLD = 0.010

# Where the table is in the cell's frame. Positions are written to file relative to this
# point, so that the dataset is expressed in the frame the criteria are stated in.
TABLE_SURFACE_HEIGHT = 0.842
TABLE_CENTER_X = 0.4
TABLE_CENTER_Y = 0.6
METRIC_DECIMALS = 6

# A block is placed a millimetre above the surface rather than exactly on it, so that a
# rounded coordinate can never read as reaching below the table.
CLEARANCE_OFFSET = 0.001


@dataclass
class BucketConfig:
    """How the two measured quantities are drawn relative to their limits.

    Each quantity is drawn from one of three regions: past its limit, just short of it, or
    comfortably inside. The shares are chosen so that roughly a fifth of the pairs violate
    each criterion, which is what makes the class balance workable without reweighting the
    data afterwards, and so that a large part of the mass sits near the limit, which is where
    a surrogate that decides feasibility has to be accurate.

    The failing region is split again, most of it just past the limit and the rest well past
    it, so that failure is not represented only by configurations that are obviously wrong.

    The ranges are given in metres and are stated per quantity, since overlap is better when
    it is large and thickness when it is small.
    """

    p_fail: float = 0.20
    p_boundary: float = 0.30
    p_near_within_fail: float = 0.35

    overlap_fail_near: tuple[float, float] = (0.005, 0.010)
    overlap_fail_far: tuple[float, float] = (0.0, 0.005)
    overlap_boundary: tuple[float, float] = (0.010, 0.015)
    overlap_far: tuple[float, float] = (0.015, 0.035)

    thickness_fail_near: tuple[float, float] = (0.020, 0.024)
    thickness_fail_far: tuple[float, float] = (0.024, 0.040)
    thickness_boundary: tuple[float, float] = (0.016, 0.020)
    thickness_far: tuple[float, float] = (BLOCK_MIN_SIZE, 0.016)


@dataclass
class SamplingConfig:
    """Everything the generator draws from.

    Attributes:
        bucket: How the two quantities are distributed relative to their limits.
        p_vertical_face: Share of pairs stacked on top of the parent. The five faces that are
            used are given equal weight. The sixth, a child hanging underneath its parent, is
            excluded: with a small parent it would place the child below the table, which is a
            failure mode Part I does not cover.
    """

    bucket: BucketConfig = field(default_factory=BucketConfig)

    p_vertical_face: float = 0.20


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


# A parent smaller than the overlap being asked for cannot realise it, and the pair would come
# out with an overlap below target. This floor is set so that every failing and every
# near-limit target is reachable; only the largest targets, well inside the feasible region,
# can still fall short, and there the exact value does not matter.
BLOCK0_MIN_SIZE = 0.015


def _sample_block0_position(size_z: float) -> np.ndarray:
    return np.array(
        [
            TABLE_CENTER_X,
            TABLE_CENTER_Y,
            TABLE_SURFACE_HEIGHT + size_z / 2 + CLEARANCE_OFFSET,
        ]
    )


def _sample_block0_size() -> np.ndarray:
    """Edge lengths of the parent, shape ``(3,)``.

    Bounded below by :data:`BLOCK0_MIN_SIZE`. That does not bias the labels: the parent has no
    parent of its own, so neither of the two quantities is measured on it.
    """
    return np.random.uniform(BLOCK0_MIN_SIZE, BLOCK_MAX_SIZE, size=3)


def _sample_bucket(cfg: BucketConfig) -> str:
    """Draw which region relative to the limit a quantity will be sampled from."""
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
    """Draw a value in metres from the range that belongs to a region."""
    if bucket == "fail":
        if random.random() < p_near_within_fail:
            return random.uniform(*ranges["fail_near"])
        return random.uniform(*ranges["fail_far"])
    if bucket == "boundary":
        return random.uniform(*ranges["boundary"])
    return random.uniform(*ranges["far"])


def _sample_pair_targets(cfg: SamplingConfig) -> dict:
    """Draw the overlap and the thickness a pair is to be built to.

    The two are drawn independently, so that the surrogate cannot learn to infer one criterion
    from the other instead of from the geometry.
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
    """Draw the contact face: one of the four sides, or the top."""
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
    """Build a child that realises the two targets on the given face.

    The thickness is easy: it is the child's edge length along the contact axis, so it is set
    directly. The overlap is the harder half, because it is the smaller of two tangential
    overlaps. One tangential axis is therefore made to realise the target exactly, and the
    other is offset so that its overlap is strictly larger and does not become the minimum.

    Which axis takes that role is not arbitrary. On a side contact, the vertical axis is
    tangential and is also the one the table clamp acts on, so the non-vertical axis is chosen
    and the clamp cannot silently change the realised overlap.

    Returns:
        The child's edge lengths and centre, both shape ``(3,)`` and in metres, in contact
        with the parent and above the table.
    """
    contact_ax = _contact_axis(face)
    tang_axes = list(_tangential_axes(face))
    sign = _face_sign(face)

    size_1 = np.zeros(3)
    size_1[contact_ax] = thickness

    # Both tangential edges are at least as long as the target overlap. Otherwise the
    # non-binding axis could be the smaller of the two and the pair would come out failing
    # when a feasible one was asked for.
    tang_floor = max(BLOCK_MIN_SIZE, min(target_overlap, BLOCK_MAX_SIZE))
    for ax in tang_axes:
        size_1[ax] = random.uniform(tang_floor, BLOCK_MAX_SIZE)

    pos_1 = pos_0.copy()
    pos_1[contact_ax] = pos_0[contact_ax] + sign * (
        size_0[contact_ax] / 2 + size_1[contact_ax] / 2
    )

    if contact_ax != 2:
        binding_ax = next(a for a in tang_axes if a != 2)
    else:
        binding_ax = random.choice(tang_axes)
    other_ax = next(a for a in tang_axes if a != binding_ax)

    max_o_binding = min(size_0[binding_ax], size_1[binding_ax])
    if max_o_binding >= target_overlap:
        delta_abs = (size_0[binding_ax] + size_1[binding_ax]) / 2 - target_overlap
        pos_1[binding_ax] = pos_0[binding_ax] + random.choice([-1.0, 1.0]) * delta_abs
        binding_realized = target_overlap
    else:
        # The target cannot be realised: neither box is wide enough. Centring them gives the
        # largest overlap available, which falls short of the target. The report at the end
        # of a run is what surfaces how often this happens.
        pos_1[binding_ax] = pos_0[binding_ax]
        binding_realized = max_o_binding

    max_o_other = min(size_0[other_ax], size_1[other_ax])
    if max_o_other > binding_realized + 1e-5:
        other_realized = random.uniform(binding_realized + 1e-5, max_o_other)
        delta_abs = (size_0[other_ax] + size_1[other_ax]) / 2 - other_realized
        pos_1[other_ax] = pos_0[other_ax] + random.choice([-1.0, 1.0]) * delta_abs
    else:
        pos_1[other_ax] = pos_0[other_ax]

    # Only side contacts need the clamp: a child stacked on top of its parent is above the
    # table already, whereas one placed beside it may have been offset downwards.
    if contact_ax != 2:
        min_pos_z = TABLE_SURFACE_HEIGHT + size_1[2] / 2 + CLEARANCE_OFFSET
        if pos_1[2] < min_pos_z:
            pos_1[2] = min_pos_z

    return size_1, pos_1


def _compute_planned_overlap(pos_p, size_p, pos_c, size_c, face):
    """Overlap of a pair as built, in metres, which may differ from the target."""
    ov = np.zeros(3)
    for d in range(3):
        min1, max1 = pos_p[d] - size_p[d] / 2, pos_p[d] + size_p[d] / 2
        min2, max2 = pos_c[d] - size_c[d] / 2, pos_c[d] + size_c[d] / 2
        ov[d] = max(0.0, min(max1, max2) - max(min1, min2))
    tang = _tangential_axes(face)
    return float(min(ov[tang[0]], ov[tang[1]]))


def _classify_from_metrics(overlap: float, thickness: float) -> int:
    """Failure code of a joint. Insufficient overlap is reported first when both fail."""
    if overlap < OVERLAP_THRESHOLD:
        return FAIL_OVERLAP
    if thickness > THICKNESS_THRESHOLD:
        return FAIL_THICKNESS
    return FAIL_NONE


def generate_assembly(num_blocks: int, cfg: SamplingConfig) -> dict:
    """Build one assembly and label every joint in it.

    Blocks are chained: each new block is built against the one before it, so an assembly of
    ``num_blocks`` blocks has ``num_blocks - 1`` labelled joints. The first block sits on the
    table and has no joint of its own, which is why the per-joint lists are one shorter than
    the per-block ones.

    Both quantities are recorded as realised, not as targeted, and the labels follow from the
    realised values, so a pair whose construction fell short of its target is labelled for
    what it is rather than for what was asked.
    """
    positions: list[np.ndarray] = []
    sizes: list[np.ndarray] = []
    faces: list[int] = []
    fail_reasons: list[int] = []
    bucket_labels: list[dict] = []

    metric_overlap: list[float] = []
    metric_thickness: list[float] = []

    # The first block has no joint, so it cannot fail; its face is recorded as absent.
    size_0 = _sample_block0_size()
    pos_0 = _sample_block0_position(size_0[2])
    sizes.append(size_0)
    positions.append(pos_0)
    faces.append(-1)
    fail_reasons.append(FAIL_NONE)

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
    """Column names of the output table, for the given number of blocks."""
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
    cols += [f"Block{b}_Overlap_m" for b in range(1, num_blocks)]
    cols += [f"Block{b}_Thickness_m" for b in range(1, num_blocks)]
    cols += [f"Block{b}_Face" for b in range(1, num_blocks)]
    cols += ["Assembly_Good?"]
    return cols


def _row_from_assembly(assembly: dict, num_blocks: int) -> list[str]:
    """Format one assembly as a row of the output table.

    Positions are written relative to the centre of the table surface, so a coordinate of zero
    on the vertical axis means resting on the table. Every consumer of the dataset works in
    that frame.
    """
    row: list[str] = []
    for b in range(num_blocks):
        pos = assembly["positions"][b]
        size = assembly["sizes"][b]
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
    assembly_good = all(r == FAIL_NONE for r in assembly["fail_reasons"])
    row.append(str(assembly_good))
    return row


def _bucket_overlap(v: float) -> str:
    """Region a realised overlap ended up in, for the report."""
    if v < OVERLAP_THRESHOLD:
        return "fail"
    if v < 0.015:
        return "boundary"
    return "far"


def _bucket_thickness(v: float) -> str:
    """Region a realised thickness ended up in, for the report."""
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
    """Summarise what the generator actually produced.

    Reports the realised failure rate per criterion and the class weight it implies, how the
    realised values are distributed relative to the limits, the correlation between the two
    failure modes, and the face mix.

    The correlation is the one number worth watching: the two quantities are drawn
    independently, so a correlation away from zero means the construction is coupling them,
    and a surrogate trained on such a set could pass by predicting one criterion from the
    other.

    Args:
        overlap_arr, thickness_arr: Realised values in metres, one entry per joint.
        face_arr: Contact face per joint.
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
    """Print the report, flagging any figure that has drifted from what was intended."""
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
        # A band around the intended fifth. Outside it the construction is not realising
        # what was asked for often enough for the class balance to hold.
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
