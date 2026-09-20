"""Train the box decoder that maps a latent code back to three half-extents.

This trains :class:`~src.dec_box.BoxDecoder`, the decoder that stands in the repair loop. It is
not the auxiliary signed-distance decoder of stage 1, which is set aside once the codes exist.
Because the repair optimises the code rather than the edge lengths, it needs a differentiable
way back to a size for the hard caps, the analytical snap and the design it writes out, and
that way back has to invert the same mapping the surrogate was trained under.

The training set is generated rather than read: half-extents are drawn log-uniformly per axis
between ``config.data.sampling_min`` and ``config.data.sampling_max``, the same distribution
the encoder saw, and are then run through the frozen encoder to give the codes. The loss is a
mean squared error in log space, so the error stays relative across a range that spans two
orders of magnitude. The encoder is loaded from
``encoder_decoder_model/best_encoder_decoder_latentdim{N}.pth`` and never updated.

Writes ``encoder_decoder_model/best_box_decoder_latentdim{N}.pth`` on every improvement of the
validation loss. Besides the weights the checkpoint carries the sampling range, the name of the
encoder checkpoint it inverts, and the validation reconstruction error in both forms:
``val_mae_mm`` in millimetres and ``val_rel_err`` as a fraction. Those two fields are the
source of the reconstruction figures quoted for Part II. A final pass over a held-out test set
reports the same error per axis and per size bucket; that pass is printed only.

Run:
    python tools/train_box_decoder.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config.config import config  # noqa: E402
from src.dec_box import BoxDecoder  # noqa: E402
from src.enc_box import BoxEncoder  # noqa: E402

MODEL_FOLDER: Path = ROOT / config.autoencoder.autoencoder_folder
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

N_TRAIN = 300_000
N_VAL = 15_000
N_TEST = 10_000
BATCH = 4096
EPOCHS = 200
LR = 1e-3
PATIENCE = 20
SEED = config.general.random_seed or 42

LATENT_DIM = config.autoencoder.latent_dim
SAMP_MIN = float(config.data.sampling_min)
SAMP_MAX = float(config.data.sampling_max)

ENCODER_CKPT = MODEL_FOLDER / f"best_encoder_decoder_latentdim{LATENT_DIM}.pth"
DECODER_CKPT = MODEL_FOLDER / f"best_box_decoder_latentdim{LATENT_DIM}.pth"


def sample_half_extents(n: int, rng: np.random.Generator) -> torch.Tensor:
    """Draw ``(n, 3)`` half-extents in metres, log-uniform per axis, as the encoder saw them."""
    log_min, log_max = np.log(SAMP_MIN), np.log(SAMP_MAX)
    half = np.exp(rng.uniform(log_min, log_max, size=(n, 3))).astype(np.float32)
    return torch.from_numpy(half)


def load_frozen_encoder() -> BoxEncoder:
    """Load the box encoder in evaluation mode with its parameters detached from the graph."""
    if not ENCODER_CKPT.exists():
        raise SystemExit(
            f"Encoder checkpoint not found: {ENCODER_CKPT}\n"
            "Train the auto-encoder first (src/enc_dec_training.py)."
        )
    ckpt = torch.load(ENCODER_CKPT, map_location="cpu", weights_only=False)
    if "encoder" not in ckpt:
        raise SystemExit(
            f"Encoder checkpoint {ENCODER_CKPT.name} has no 'encoder' key; "
            f"available keys: {list(ckpt.keys())}"
        )
    encoder = BoxEncoder(latent_dim=LATENT_DIM)
    encoder.load_state_dict(ckpt["encoder"])
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad_(False)
    return encoder.to(DEVICE)


@torch.no_grad()
def encode_in_batches(
    encoder: BoxEncoder, half: torch.Tensor, batch: int = 8192
) -> torch.Tensor:
    """Encode all half-extents once up front, returning the codes on the CPU."""
    out: list[torch.Tensor] = []
    for i in range(0, half.shape[0], batch):
        chunk = half[i : i + batch].to(DEVICE)
        z = encoder(chunk).cpu()
        out.append(z)
    return torch.cat(out, dim=0)


def main() -> None:
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    rng = np.random.default_rng(SEED)
    MODEL_FOLDER.mkdir(parents=True, exist_ok=True)

    print(f"Device: {DEVICE}  latent_dim={LATENT_DIM}")
    print(f"Encoder checkpoint: {ENCODER_CKPT.name}")
    print(
        f"half_extent range: [{SAMP_MIN*1000:.1f} mm, {SAMP_MAX*1000:.1f} mm] "
        f"log-uniform per axis"
    )

    encoder = load_frozen_encoder()

    # The encoder is frozen, so every code is fixed for the whole run and can be computed once
    # instead of on every epoch.
    t0 = time.time()
    half_train = sample_half_extents(N_TRAIN, rng)
    half_val = sample_half_extents(N_VAL, rng)
    half_test = sample_half_extents(N_TEST, rng)
    z_train = encode_in_batches(encoder, half_train)
    z_val = encode_in_batches(encoder, half_val)
    z_test = encode_in_batches(encoder, half_test)
    print(
        f"Data ready ({N_TRAIN} train / {N_VAL} val / {N_TEST} test) "
        f"in {time.time()-t0:.1f}s"
    )
    print(
        f"  z range: [{z_train.min().item():.3f}, {z_train.max().item():.3f}]  "
        f"|z| mean={z_train.norm(dim=1).mean().item():.3f}"
    )

    decoder = BoxDecoder(latent_dim=LATENT_DIM).to(DEVICE)
    n_params = sum(p.numel() for p in decoder.parameters())
    print(f"BoxDecoder params: {n_params:,}")

    optimizer = torch.optim.Adam(decoder.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=4, min_lr=1e-6
    )

    half_train_dev = half_train.to(DEVICE)
    z_train_dev = z_train.to(DEVICE)
    half_val_dev = half_val.to(DEVICE)
    z_val_dev = z_val.to(DEVICE)

    # The loss is taken in log space so that a millimetre of error on a 4 mm block counts as
    # much as on a 300 mm one; the targets are logged once rather than per batch.
    log_half_train_dev = torch.log(half_train_dev)
    log_half_val_dev = torch.log(half_val_dev)

    best_val = float("inf")
    patience_counter = 0

    for epoch in range(1, EPOCHS + 1):
        decoder.train()
        perm = torch.randperm(N_TRAIN, device=DEVICE)
        t_loss = 0.0
        n_batches = 0
        for i in range(0, N_TRAIN, BATCH):
            idx = perm[i : i + BATCH]
            z = z_train_dev[idx]
            log_target = log_half_train_dev[idx]
            log_pred = decoder.forward_log(z)
            loss = F.mse_loss(log_pred, log_target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            t_loss += loss.item()
            n_batches += 1

        decoder.eval()
        with torch.no_grad():
            log_val_pred = decoder.forward_log(z_val_dev)
            val_logmse = F.mse_loss(log_val_pred, log_half_val_dev).item()
            half_val_pred = torch.exp(log_val_pred)
            val_mse = F.mse_loss(half_val_pred, half_val_dev).item()
            val_mae_mm = (half_val_pred - half_val_dev).abs().mean().item() * 1000.0
            rel_err = ((half_val_pred - half_val_dev).abs() / half_val_dev).mean().item()

        scheduler.step(val_logmse)
        lr_now = optimizer.param_groups[0]["lr"]
        print(
            f"E{epoch:03d}/{EPOCHS}  train_logmse={t_loss/n_batches:.3e}  "
            f"val_logmse={val_logmse:.3e}  val_MAE={val_mae_mm:.3f} mm  "
            f"rel_err={rel_err*100:.2f}%  lr={lr_now:.1e}"
        )

        if val_logmse < best_val - 1e-9:
            best_val = val_logmse
            patience_counter = 0
            ckpt = {
                "decoder": decoder.state_dict(),
                "latent_dim": LATENT_DIM,
                "decoder_type": "box",
                "samp_min": SAMP_MIN,
                "samp_max": SAMP_MAX,
                "encoder_ckpt": ENCODER_CKPT.name,
                "val_logmse": val_logmse,
                "val_mse_m": val_mse,
                "val_mae_mm": val_mae_mm,
                "val_rel_err": rel_err,
            }
            torch.save(ckpt, DECODER_CKPT)
            print(f"  -> saved {DECODER_CKPT.name}  (val MAE {val_mae_mm:.3f} mm, rel {rel_err*100:.2f}%)")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"  Early stopping at epoch {epoch} (val plateau)")
                break

    # Reload the best checkpoint rather than reporting on the last epoch, which early stopping
    # has usually left worse than the one on disk.
    print("\n--- Test-set roundtrip diagnostic ---")
    ckpt = torch.load(DECODER_CKPT, map_location=DEVICE, weights_only=False)
    decoder.load_state_dict(ckpt["decoder"])
    decoder.eval()
    with torch.no_grad():
        half_test_pred = decoder(z_test.to(DEVICE)).cpu()
    err_mm = (half_test_pred - half_test).abs() * 1000.0
    rel_err = (half_test_pred - half_test).abs() / half_test
    per_axis = err_mm.mean(dim=0)
    print(
        f"Test MAE: {err_mm.mean().item():.3f} mm  "
        f"(per axis x={per_axis[0]:.3f}  y={per_axis[1]:.3f}  z={per_axis[2]:.3f})"
    )
    print(
        f"Test max abs error: {err_mm.max().item():.3f} mm  "
        f"95th percentile: {err_mm.flatten().quantile(0.95).item():.3f} mm"
    )
    print(
        f"Test mean relative error: {rel_err.mean().item()*100:.2f}%  "
        f"95th percentile: {rel_err.flatten().quantile(0.95).item()*100:.2f}%"
    )

    # A mean over the whole range hides the only bucket that decides whether the repair can
    # work at all. The thinnest one, up to 10 mm half-extent, is 20 mm full and therefore holds
    # exactly the beams that sit at or below config.gnn.thresh_thickness_max; if the decoder is
    # inaccurate there, no amount of optimisation in latent space can realise a thin enough
    # member.
    for lo_mm, hi_mm in [(1, 10), (10, 25), (25, 75), (75, 150)]:
        mask = ((half_test * 1000.0 >= lo_mm) & (half_test * 1000.0 <= hi_mm)).all(dim=1)
        n = int(mask.sum())
        if n > 0:
            mae = err_mm[mask].mean().item()
            rel = rel_err[mask].mean().item() * 100
            print(f"  bucket [{lo_mm}, {hi_mm}] mm half-extent: n={n}  MAE={mae:.3f} mm  rel={rel:.2f}%")


if __name__ == "__main__":
    main()
