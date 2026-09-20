"""Check whether the box encoder still behaves on blocks smaller than it was trained on.

Blocks in a design are scaled into the encoder's domain before they are encoded, and the
smallest of them land below ``config.data.sampling_min``, which is where the encoder stops
interpolating and starts extrapolating. Everything downstream inherits whatever it does there,
so this script tests it directly. It uses the stage-1 pair, the box encoder together with the
auxiliary signed-distance decoder, because that is the only combination against which the code
can be scored on geometry rather than on a size it was fitted to reproduce.

Three tests, all printed, nothing written to disk:

1. Reconstruction error. Encode a sampled box, decode the signed distance at query points, and
   compare against the analytical box field. Reported separately for boxes inside and outside
   the training range, and per decile of the smallest scaled half-extent.
2. Smoothness. Sweep an isotropic half-extent through and below the training range and report
   the norm of the change in code per change in extent. A spike at the boundary is the encoder
   extrapolating non-linearly.
3. Injectivity. Sample small boxes of differing shape and compare the distance between their
   codes against the distance between their extents. Codes that collapse together are codes
   the surrogate cannot tell apart, however well it is trained.

Run:
    python tools/check_encoder_range.py
    python tools/check_encoder_range.py --data-file data/synth_<...>.txt \\
        --num-boxes 200 --num-query 4000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config.config import config  # noqa: E402
from src.dec_sdf import SDFDecoder  # noqa: E402
from src.enc_box import BoxEncoder  # noqa: E402
from src.simulation_dataset import get_scale_factor  # noqa: E402

DATA_DIR = ROOT / "data"
ENCODER_DIR = ROOT / config.autoencoder.autoencoder_folder
DEFAULT_CKPT = ENCODER_DIR / f"best_encoder_decoder_latentdim{config.autoencoder.latent_dim}.pth"

TRAINING_MIN = float(config.data.sampling_min)
TRAINING_MAX = float(config.data.sampling_max)
# The same clamp the auxiliary decoder was trained under. Scoring outside it would compare the
# network against distances it was never asked to fit.
SDF_CLAMP = float(config.training.sdf_clamp_delta)


def load_encoder_decoder(ckpt_path: Path, device: str):
    """Load the box encoder and the auxiliary signed-distance decoder from one stage-1 checkpoint."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    enc = BoxEncoder().to(device)
    enc.load_state_dict(ckpt["encoder"])
    enc.eval()
    dec = SDFDecoder().to(device)
    dec.load_state_dict(ckpt["decoder"])
    dec.eval()
    return enc, dec


def get_dataset_scale_factor(data_file: Path | None) -> float:
    """Return the factor that maps dataset metres into the encoder domain, and the file it came from."""
    if data_file is None:
        candidates = sorted(
            DATA_DIR.glob("synth_num_blocks_*.txt"), key=lambda p: p.stat().st_mtime
        )
        if not candidates:
            raise SystemExit("No synth_num_blocks_*.txt file found in data/.")
        data_file = candidates[-1]
    df = pd.read_csv(data_file, usecols=lambda c: "Size" in c)
    sf = get_scale_factor(df)
    return sf, data_file


def box_sdf(p: np.ndarray, h: np.ndarray) -> np.ndarray:
    """Exact signed distance from ``(N, 3)`` points to a box of half-extents ``h`` at the origin."""
    q = np.abs(p) - h
    outside = np.linalg.norm(np.maximum(q, 0.0), axis=-1)
    inside = np.minimum(np.max(q, axis=-1), 0.0)
    return outside + inside


def test1_sdf_reconstruction(
    encoder: BoxEncoder,
    decoder: SDFDecoder,
    scale_factor: float,
    num_boxes: int,
    num_query: int,
    block_min: float,
    block_max: float,
    device: str,
    rng: np.random.Generator,
) -> dict:
    """Encode boxes across the dataset size range and score the decoded field against the exact one."""
    # Log-uniform per axis, matching how the training shapes were drawn, so that thin and
    # near-cubic boxes are equally represented rather than the large ones dominating.
    h_log_min = np.log(block_min / 2)
    h_log_max = np.log(block_max / 2)
    halves = np.exp(rng.uniform(h_log_min, h_log_max, size=(num_boxes, 3))).astype(
        np.float32
    )

    halves_scaled = halves * scale_factor

    h_t = torch.from_numpy(halves_scaled).to(device)
    with torch.no_grad():
        z = encoder(h_t)

    # A box counts as out of range as soon as its *smallest* axis falls below the training
    # floor: one thin axis is enough to put the code into extrapolation.
    box_min_scaled = halves_scaled.min(axis=1)
    in_range_mask = box_min_scaled >= TRAINING_MIN

    mae_scaled = np.zeros(num_boxes, dtype=np.float32)
    rel_err = np.zeros(num_boxes, dtype=np.float32)

    n_surface = num_query // 2
    n_uniform = num_query - n_surface
    for i in range(num_boxes):
        h_i = halves_scaled[i]
        # Half the points near the surface and half in the volume, which is the mix the
        # decoder was trained on. Scoring on uniform points alone would flatter it: away from
        # the surface the field is easy and the clamp hides most of the error.
        face_axis = rng.integers(0, 3, size=n_surface)
        face_sign = rng.choice([-1.0, 1.0], size=n_surface)
        p_surf = rng.uniform(-1.0, 1.0, size=(n_surface, 3)).astype(np.float32) * h_i
        for j in range(n_surface):
            p_surf[j, face_axis[j]] = face_sign[j] * h_i[face_axis[j]]
        # Normal noise of the same width as config.data.sigma, so the surface points sit where
        # the training points sat rather than exactly on the face.
        p_surf += rng.normal(0.0, 0.01, size=(n_surface, 3)).astype(np.float32)

        bound = 1.5 * float(h_i.max())
        p_unif = rng.uniform(-bound, bound, size=(n_uniform, 3)).astype(np.float32)
        p = np.concatenate([p_surf, p_unif], axis=0)

        sdf_true = box_sdf(p, h_i).astype(np.float32)
        with torch.no_grad():
            z_i = z[i : i + 1].repeat(num_query, 1)
            p_t = torch.from_numpy(p).to(device)
            sdf_pred = decoder(z_i, p_t).cpu().numpy()
        # Both sides are clamped, not just the prediction: the training loss only ever saw the
        # band, so an unclamped comparison would charge the decoder for a region it was told
        # to ignore.
        sdf_true_c = np.clip(sdf_true, -SDF_CLAMP, SDF_CLAMP)
        sdf_pred_c = np.clip(sdf_pred, -SDF_CLAMP, SDF_CLAMP)
        err = np.abs(sdf_pred_c - sdf_true_c)
        mae_scaled[i] = float(err.mean())
        # Expressed as a fraction of the clamp width, so the column reads between 0 and 1.
        rel_err[i] = float(err.mean() / (2.0 * SDF_CLAMP))

    mae_mm = mae_scaled / scale_factor * 1000.0

    print("=" * 72)
    print(" TEST 1 — SDF reconstruction MAE (lower is better)")
    print("=" * 72)
    print(f" {num_boxes} boxes, {num_query} query points each (50% surface-biased)")
    print(f" SDF clamped to ±{SDF_CLAMP} (matches training)")
    print(f" Scale factor: {scale_factor:.4f}, training range: [{TRAINING_MIN}, {TRAINING_MAX}]")
    print()
    print(f" {'Bucket':<28} {'#':>5} {'MAE (mm)':>10} {'MAE/clamp':>10}")
    print(" " + "─" * 65)
    if in_range_mask.any():
        i_mae = np.median(mae_mm[in_range_mask])
        i_rel = np.median(rel_err[in_range_mask])
        print(f" {'in-range  (h_min ≥ 0.05)':<28} {int(in_range_mask.sum()):>5} {i_mae:>10.3f} {i_rel:>10.4f}")
    if (~in_range_mask).any():
        o_mae = np.median(mae_mm[~in_range_mask])
        o_rel = np.median(rel_err[~in_range_mask])
        print(f" {'out-of-range (h_min < 0.05)':<28} {int((~in_range_mask).sum()):>5} {o_mae:>10.3f} {o_rel:>10.4f}")
    print(f" {'overall':<28} {num_boxes:>5} {float(np.median(mae_mm)):>10.3f} {float(np.median(rel_err)):>10.4f}")
    print()

    # Deciles rather than fixed buckets: the interesting question is whether the error rises
    # monotonically as boxes get thinner, and equal-count buckets keep every row comparable.
    deciles = np.percentile(box_min_scaled, np.arange(0, 101, 10))
    print(f" {'box_min_scaled bucket':<28} {'#':>5} {'MAE (mm)':>10} {'MAE/clamp':>10}")
    print(" " + "─" * 65)
    for d in range(10):
        lo, hi = deciles[d], deciles[d + 1]
        sel = (box_min_scaled >= lo) & (box_min_scaled < hi if d < 9 else box_min_scaled <= hi)
        if not sel.any():
            continue
        bmae = float(np.median(mae_mm[sel]))
        brel = float(np.median(rel_err[sel]))
        flag = " " if lo >= TRAINING_MIN else "*"
        print(f" {flag}[{lo:.4f}, {hi:.4f}]            {int(sel.sum()):>5} {bmae:>10.3f} {brel:>10.4f}")
    print(" * = below encoder training_min (0.05)")
    print()

    return {
        "halves": halves,
        "halves_scaled": halves_scaled,
        "box_min_scaled": box_min_scaled,
        "mae_mm": mae_mm,
        "rel_err": rel_err,
        "in_range_mask": in_range_mask,
    }


def test2_smoothness(
    encoder: BoxEncoder, scale_factor: float, device: str
) -> None:
    """Sweep an isotropic half-extent and report how fast the code changes with it."""
    print("=" * 72)
    print(" TEST 2 — Encoder smoothness (||dz/dh|| over isotropic h-sweep)")
    print("=" * 72)

    # Swept in the scaled domain so that the training boundary is a fixed landmark on the axis
    # and the in-range and out-of-range halves can be reported separately.
    h_scaled = np.linspace(0.005, TRAINING_MAX, 400, dtype=np.float32)
    H = np.stack([h_scaled] * 3, axis=1)
    with torch.no_grad():
        z = encoder(torch.from_numpy(H).to(device)).cpu().numpy()
    dz_dh = np.linalg.norm(np.diff(z, axis=0), axis=1) / np.diff(h_scaled)

    # A finite difference is one shorter than the sweep, so the mask drops the last point.
    out_of_range = h_scaled[:-1] < TRAINING_MIN
    in_range = ~out_of_range
    print(f" Sweep: h_scaled ∈ [0.005, {TRAINING_MAX}], 400 pts")
    print(f" ||dz/dh||  median           in-range: {float(np.median(dz_dh[in_range])):.3f}")
    print(f" ||dz/dh||  median       out-of-range: {float(np.median(dz_dh[out_of_range])):.3f}")
    print(f" ||dz/dh||  max               in-range: {float(np.max(dz_dh[in_range])):.3f}")
    print(f" ||dz/dh||  max           out-of-range: {float(np.max(dz_dh[out_of_range])):.3f}")
    # A peak sitting on the training floor is the signature of extrapolation starting there,
    # which is why the location is printed next to the floor itself.
    idx_peak = int(np.argmax(dz_dh))
    print(f" peak at h_scaled = {float(h_scaled[idx_peak]):.4f}  (training_min = {TRAINING_MIN})")
    print()


def test3_injectivity(
    encoder: BoxEncoder, scale_factor: float, block_min: float, device: str,
    rng: np.random.Generator,
) -> None:
    """Compare distances between codes against distances between extents, for small boxes."""
    print("=" * 72)
    print(" TEST 3 — Latent injectivity below the training range")
    print("=" * 72)

    h_low = block_min / 2.0
    # Scaling maps real metres into the encoder domain, so the real half-extent that lands
    # exactly on the training floor is the floor divided by the scale factor.
    h_target_max = TRAINING_MIN / scale_factor

    n = 200
    halves = np.zeros((n, 3), dtype=np.float32)
    for i in range(n):
        # One axis is forced below the floor and the other two are left free, so that every
        # box is out of range but the sample still spans a wide set of aspect ratios.
        out_axis = rng.integers(0, 3)
        halves[i, out_axis] = rng.uniform(h_low, h_target_max)
        for ax in range(3):
            if ax == out_axis:
                continue
            halves[i, ax] = rng.uniform(h_low, block_min * 12.5 / 2)
    halves_scaled = halves * scale_factor
    with torch.no_grad():
        z = encoder(torch.from_numpy(halves_scaled).to(device)).cpu().numpy()

    idx_a = rng.integers(0, n, size=500)
    idx_b = rng.integers(0, n, size=500)
    dh = np.linalg.norm(halves_scaled[idx_a] - halves_scaled[idx_b], axis=1)
    dz = np.linalg.norm(z[idx_a] - z[idx_b], axis=1)
    # Pairs are drawn with replacement, so a pair can hit the same box twice; those have zero
    # extent distance and would divide by zero.
    valid = dh > 1e-6
    ratio = dz[valid] / dh[valid]

    print(f" {n} small boxes (≥ 1 axis < {TRAINING_MIN/scale_factor:.4f} in real metres)")
    print(f" 500 random pairs; ratio = ||Δz|| / ||Δh_scaled||")
    print(f"   median ratio = {float(np.median(ratio)):.3f}")
    print(f"   p10/p90       = {float(np.percentile(ratio, 10)):.3f} / {float(np.percentile(ratio, 90)):.3f}")
    # A ratio near zero means two visibly different boxes received nearly the same code, which
    # the surrogate could never undo; 0.05 is the reporting cut-off for that warning.
    if (ratio < 0.05).any():
        n_collapsed = int((ratio < 0.05).sum())
        print(f"   WARN: {n_collapsed}/{len(ratio)} pairs have ratio < 0.05 — possible collapse")
    else:
        print("   no near-collapse pairs (all ratios > 0.05)")
    print()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", type=str, default=str(DEFAULT_CKPT))
    parser.add_argument(
        "--data-file", type=str, default=None,
        help="Dataset CSV to compute scale_factor from (default: latest synth_num_blocks_*.txt)"
    )
    parser.add_argument("--block-min", type=float, default=0.008,
                        help="BLOCK_MIN_SIZE in real metres (default: 0.008)")
    parser.add_argument("--block-max", type=float, default=0.200,
                        help="BLOCK_MAX_SIZE in real metres (default: 0.200)")
    parser.add_argument("--num-boxes", type=int, default=300,
                        help="Boxes for SDF reconstruction test")
    parser.add_argument("--num-query", type=int, default=3000,
                        help="Query points per box for SDF test")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default=None,
                        help="cuda or cpu (default: auto)")
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(args.seed)

    print(f"Device: {device}")
    print(f"Checkpoint: {args.ckpt}")

    sf, data_file = get_dataset_scale_factor(
        Path(args.data_file) if args.data_file else None
    )
    print(f"Dataset for scale_factor: {data_file.name}")
    print(f"scale_factor = {sf:.4f}")
    print()

    enc, dec = load_encoder_decoder(Path(args.ckpt), device)

    test1_sdf_reconstruction(
        enc, dec, sf,
        num_boxes=args.num_boxes, num_query=args.num_query,
        block_min=args.block_min, block_max=args.block_max,
        device=device, rng=rng,
    )
    test2_smoothness(enc, sf, device)
    test3_injectivity(enc, sf, args.block_min, device, rng)


if __name__ == "__main__":
    main()
