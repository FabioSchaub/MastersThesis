"""Training of the shape autoencoder, in three stages, with the Least Volume compression.

Training encoder and decoder together from the start does not work here: the decoder can reach
a fair average error by ignoring the code and returning one mean shape, and once it does the
encoder receives no reason to distinguish shapes either. The three stages take the problem
apart so that this cannot happen.

    Stage 1     The code of every shape is a free parameter, one row of an embedding, trained
                jointly with the decoder and without any encoder. A code that is not useful is
                changed by its own gradient, so the codes cannot all become the same one.
    Stage 2     The decoder and the codes are frozen and the encoder is trained to reproduce
                them from the surface points. It is a regression onto a fixed target, with no
                decoder in the backward pass, so there is nothing for it to collapse onto.
    Stage 3     Encoder and decoder are unfrozen and trained together at a small learning rate,
                starting from an encoder that already produces distinct codes.

The Least Volume compression acts in stage 3, and only there. Stage 1 is an auto-decoder rather
than an autoencoder: nothing maps a shape to its code, so nothing prevents the penalty from
shrinking every code towards zero. In stage 3 the code is the output of the encoder, which has
to keep distinct shapes apart in order to reconstruct them, and the penalty then removes what
is genuinely unused. This is why the sweep of the thesis runs stage 3 alone on top of an
already trained autoencoder.

The settings of the compression are read from environment variables, with the configuration as
the fallback, so that a sweep can vary the penalty weight, the spectral normalisation and the
number of epochs without editing a file shared by every run. ``AE_RUN_TAG`` gives each run of a
sweep its own checkpoint name, which is what allows them to run in parallel. The value of
``lambda_vol`` in the configuration is not the one the thesis reports.

Run:
    python -m src.enc_dec_training
    GENERAL_SHAPES=1 LV_S3_ONLY=1 LV_SPECTRAL=1 LV_LAMBDA_VOL=0.05 python -m src.enc_dec_training
"""

import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import wandb
from torch import nn
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from config.config import config
from src.enc_box import BoxEncoder
from src.enc_dec_dataset_generation import AutoEncoderDataset
from src.dec_sdf import SDFDecoder
from src.least_volume import volume_penalty, sigma_spectrum, active_dims

MODEL_FOLDER: Path = (
    Path(__file__).parent.parent / config.autoencoder.autoencoder_folder
)
MODEL_FOLDER.mkdir(parents=True, exist_ok=True)

# Fixed once when the module is loaded, so that every checkpoint of a run shares it. Each
# checkpoint is written twice, once under this suffix as an archive and once without it, and
# the unsuffixed name is the one everything downstream reads. A new run therefore never
# destroys the record of an old one, but it does take over the name.
TIMESTAMP: str = time.strftime("%Y%m%d-%H%M%S")


def set_seed(seed: int = config.general.random_seed or 42) -> None:
    """Seed the random number generators and make the convolutions deterministic."""
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


class IndexedDataset(torch.utils.data.Dataset):
    """Wraps the shape dataset so that every sample also carries its index.

    Stage 1 stores one code per shape in an embedding and stage 2 uses the same codes as
    targets, so both need to know which shape a sample is, and the index has to survive the
    split into training and validation.
    """

    def __init__(self, base_dataset: AutoEncoderDataset) -> None:
        """Keep a reference to the dataset being wrapped; nothing is copied."""
        self.base = base_dataset

    def __len__(self) -> int:
        """Same length as the wrapped dataset."""
        return len(self.base)

    def __getitem__(
        self, idx: int
    ) -> tuple[int, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return the index followed by the surface, the queries, the distances and the half
        extents of one shape.
        """
        surface, query, sdf = self.base[idx]
        he = self.base.half_extents[idx]
        # Only boxes have half extents. A batch cannot hold a missing entry, and the path that
        # reads surface points ignores the field anyway, so the others get zeros.
        half = (
            torch.from_numpy(he)
            if he is not None
            else torch.zeros(3, dtype=torch.float32)
        )
        return idx, surface, query, sdf, half


def decode_batch(
    decoder: SDFDecoder,
    z: torch.Tensor,
    query_pts: torch.Tensor,
) -> torch.Tensor:
    """Evaluate the decoder for a batch of shapes at all of their query points at once.

    Args:
        z: Latent codes, shape ``(B, latent_dim)``, one per shape.
        query_pts: Query coordinates, shape ``(B, N_query, 3)``.

    Returns:
        The predicted signed distances, shape ``(B, N_query)``.
    """
    B, N_query, _ = query_pts.shape
    latent_dim = z.shape[1]
    z_exp = z.unsqueeze(1).expand(B, N_query, latent_dim)
    inp = torch.cat([z_exp, query_pts], dim=-1).reshape(B * N_query, latent_dim + 3)
    out = decoder.net(inp).squeeze(-1).reshape(B, N_query)
    # This calls the layers directly and so bypasses the decoder's own forward method, where
    # the output scale of the spectral variant is applied. It has to be repeated here, or the
    # decoder would be trained on one scale and queried on another.
    if getattr(decoder, "spectral", False) and getattr(decoder, "out_scale", 1.0) != 1.0:
        out = out * decoder.out_scale
    return out


def sdf_reconstruction_loss(
    sdf_pred: torch.Tensor,
    sdf_gt: torch.Tensor,
    z: torch.Tensor,
    lambda_reg: float,
    delta: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reconstruction loss on the distance field, plus a pull on the size of the codes.

    Both prediction and target are clipped to a band around zero before they are compared, so
    that the loss only asks the decoder to be right near the surface. Far from it the sign is
    what matters and the magnitude is not worth fitting; without the clipping the few queries
    deep inside or far outside would dominate the average.

    Args:
        lambda_reg: Weight of the term that keeps the codes from growing without bound.
        delta: Half width of the band the distances are clipped to.

    Returns:
        The total loss, and its reconstruction and regularisation parts separately, so that a
        run can be diagnosed from which of the two moved.
    """
    sdf_pred_c = torch.clamp(sdf_pred, -delta, delta)
    sdf_gt_c = torch.clamp(sdf_gt, -delta, delta)
    recon = F.l1_loss(sdf_pred_c, sdf_gt_c)
    reg = torch.mean(z.norm(dim=1) ** 2)
    return recon + lambda_reg * reg, recon, reg


@torch.no_grad()
def z_diversity_metrics(
    z_matrix: torch.Tensor, n_pairs: int = 1000
) -> dict[str, float]:
    """Measure how far apart a set of codes is, which is the diagnostic for a collapse.

    The cosine similarity between codes is the direct symptom: if it approaches one for every
    pair, the codes point the same way and the encoder has stopped distinguishing shapes. It is
    estimated from randomly drawn pairs rather than from all of them, since the number of pairs
    grows quadratically with the number of shapes.

    Args:
        z_matrix: Latent codes, shape ``(N, D)``.
        n_pairs: Number of pairs the similarity is estimated from.
    """
    norms = z_matrix.norm(dim=1)
    N = z_matrix.shape[0]
    idx_a = torch.randint(0, N, (n_pairs,))
    idx_b = torch.randint(0, N, (n_pairs,))
    # A code compared with itself has similarity one and would bias the estimate upward, that
    # is towards a collapse that is not there.
    same = idx_a == idx_b
    idx_b[same] = (idx_b[same] + 1) % N
    za = F.normalize(z_matrix[idx_a], dim=-1)
    zb = F.normalize(z_matrix[idx_b], dim=-1)
    cos_vals = (za * zb).sum(dim=-1)
    return {
        "z_norm_mean": float(norms.mean()),
        "z_norm_std": float(norms.std()),
        "z_norm_min": float(norms.min()),
        "z_norm_max": float(norms.max()),
        "cos_sim_mean": float(cos_vals.mean()),
        "cos_sim_std": float(cos_vals.std()),
        "cos_sim_max": float(cos_vals.max()),
    }


def train_stage1(
    decoder: SDFDecoder,
    latent_codes: nn.Embedding,
    stage1_loader: torch.utils.data.DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    patience: int,
    recon_threshold: float,
    lambda_reg: float,
    delta: float,
    run: "wandb.sdk.wandb_run.Run",
    name_tag: str = "",
    lambda_vol: float = 0.0,
    vol_eta: float = 0.1,
) -> float:
    """Train the decoder together with one free latent code per shape.

    There is no encoder and no validation split: every shape has its own code, so a shape the
    optimiser has not seen has no code either and holding some out would measure nothing. The
    checkpoint of this stage is judged on the training loss for the same reason.

    If a volume penalty is requested it is ramped in over the first part of the run rather than
    applied at once, so that a usable reconstruction exists before the compression starts to
    pull against it. Because the total loss then rises while the ramp is under way, everything
    that reads the total loss has to be suspended for that period: the learning rate schedule,
    the selection of the best epoch and both early stops. That is why the compressed runs of
    the thesis are the ones that use their full epoch budget.

    Args:
        latent_codes: One row per shape, the free codes of this stage.
        recon_threshold: Reconstruction error at which the stage stops early, in the units of
            the canonical frame.
        lambda_vol: Weight of the volume penalty. Zero disables it and restores the plain
            behaviour, including the schedule and the early stops.
        vol_eta: Offset of the volume penalty.

    Returns:
        The best training loss reached, measured after the ramp when there is one.
    """
    vol_warm_len = max(1, int(0.4 * epochs))
    optimizer = torch.optim.Adam(
        [
            {"params": decoder.parameters(), "lr": lr},
            {
                # The codes are updated faster than the decoder: they are the only thing that
                # tells one shape from another, and a code that moves too slowly leaves the
                # decoder averaging over shapes it cannot yet distinguish.
                "params": latent_codes.parameters(),
                "lr": lr * 3,
            },
        ]
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=7, min_lr=1e-6
    )

    best_train_loss = float("inf")
    patience_counter = 0

    for epoch in range(epochs):
        decoder.train()
        latent_codes.train()
        t_total = t_recon = t_reg = t_vol = 0.0
        vol_lam_eff = lambda_vol * min(1.0, epoch / vol_warm_len) if lambda_vol > 0 else 0.0

        for idx, _surf, query_pts, sdf_gt, _half in tqdm(
            stage1_loader, desc=f"S1 E{epoch + 1}", leave=False
        ):
            idx = idx.to(device)
            query_pts = query_pts.to(device)
            sdf_gt = sdf_gt.to(device)

            z = latent_codes(idx)
            sdf_pred = decode_batch(decoder, z, query_pts)
            loss, recon, reg = sdf_reconstruction_loss(
                sdf_pred, sdf_gt, z, lambda_reg, delta
            )
            if vol_lam_eff > 0:
                # Over the whole table of codes rather than the batch: here the codes are free
                # parameters and all of them are available, so the standard deviations need not
                # be estimated from a sample.
                vol = volume_penalty(latent_codes.weight, vol_eta)
                loss = loss + vol_lam_eff * vol
                t_vol += float(vol.item())

            optimizer.zero_grad()
            loss.backward()
            # Clipped because a code is updated by its own gradient alone and receives it only
            # in the batches it appears in, so a single large step is not averaged away.
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(latent_codes.parameters(), max_norm=1.0)
            optimizer.step()

            t_total += loss.item()
            t_recon += recon.item()
            t_reg += reg.item()

        n = len(stage1_loader)
        train_loss = t_total / n
        train_recon = t_recon / n
        train_reg = t_reg / n
        train_vol = t_vol / n

        all_z = latent_codes.weight.detach().cpu()
        div = z_diversity_metrics(all_z)
        current_lr = optimizer.param_groups[0]["lr"]

        sig = sigma_spectrum(all_z)
        # Two thresholds, because the drop in the spectrum is gradual: the loose one counts
        # every entry that still varies at all, the strict one only those that carry the shape.
        adims = active_dims(all_z)
        adims10 = active_dims(all_z, frac=0.10)
        sig_min = float(sig[-1])

        vol_str = (f" | vol {train_vol:.4f} (lam_eff {vol_lam_eff:.4f}) "
                   f"active {adims}/{adims10} sig_min {sig_min:.4f}") if lambda_vol > 0 else ""
        print(
            f"S1 E{epoch + 1}/{epochs} | "
            f"train {train_loss:.5f} (recon {train_recon:.5f}, reg {train_reg:.5f}) | "
            f"z_norm {div['z_norm_mean']:.3f}±{div['z_norm_std']:.3f} | "
            f"cos mean={div['cos_sim_mean']:.3f} max={div['cos_sim_max']:.3f} | "
            f"LR {current_lr:.2e}{vol_str}"
        )

        run.log(
            {
                "stage1/train_total": train_loss,
                "stage1/train_recon": train_recon,
                "stage1/train_reg": train_reg,
                "stage1/train_vol": train_vol,
                "stage1/vol_lam_eff": vol_lam_eff,
                "stage1/active_dims": adims,
                "stage1/active_dims10": adims10,
                "stage1/sigma_max": float(sig[0]),
                "stage1/sigma_min": sig_min,
                "stage1/z_norm_mean": div["z_norm_mean"],
                "stage1/z_norm_std": div["z_norm_std"],
                "stage1/z_norm_min": div["z_norm_min"],
                "stage1/z_norm_max": div["z_norm_max"],
                "stage1/cos_sim_mean": div["cos_sim_mean"],
                "stage1/cos_sim_std": div["cos_sim_std"],
                "stage1/cos_sim_max": div["cos_sim_max"],
                "stage1/lr": current_lr,
                "stage1/epoch": epoch,
            }
        )

        # The schedule reduces the rate when the loss stops falling. While the penalty ramps in
        # the loss is meant to rise, so the schedule would read the ramp as a plateau and cut
        # the rate away just as the compression is starting. The compressed runs keep it fixed.
        if lambda_vol == 0:
            scheduler.step(train_loss)

        in_warmup = lambda_vol > 0 and epoch < vol_warm_len
        if lambda_vol > 0 and epoch == vol_warm_len:
            # The objective only becomes comparable across epochs once the weight is constant,
            # so the search for the best epoch restarts here rather than at the beginning.
            best_train_loss = float("inf")
            patience_counter = 0

        # During the ramp every epoch is written: there is no meaningful best to keep, and the
        # last epoch is as good a restart point as any if the run is interrupted.
        save_now = False
        if in_warmup:
            save_now = True
        elif train_loss < best_train_loss:
            best_train_loss = train_loss
            patience_counter = 0
            save_now = True
        else:
            patience_counter += 1

        if save_now:
            ckpt = {
                "decoder": decoder.state_dict(),
                "latent_codes": latent_codes.state_dict(),
                "latent_dim": latent_codes.embedding_dim,
                "n_shapes": latent_codes.num_embeddings,
                "decoder_spectral": getattr(decoder, "spectral", False),
                "lipschitz_k": getattr(decoder, "out_scale", 1.0),
                "timestamp": TIMESTAMP,
            }
            latent_dim_s1 = config.autoencoder.latent_dim
            unsuffixed_s1 = MODEL_FOLDER / f"stage1_best{name_tag}_latentdim{latent_dim_s1}.pth"
            archived_s1 = MODEL_FOLDER / f"stage1_best{name_tag}_latentdim{latent_dim_s1}_{TIMESTAMP}.pth"
            torch.save(ckpt, unsuffixed_s1)
            torch.save(ckpt, archived_s1)
            tag = "warmup" if in_warmup else f"best train {train_loss:.5f}"
            print(f"  -> Saved {unsuffixed_s1.name} ({tag})")

        # Neither early stop applies to a compressed run. The reconstruction reaches its
        # threshold long before the penalty has done anything, so stopping on it would end the
        # run at the very point the compression is supposed to begin.
        if lambda_vol == 0 and train_recon <= recon_threshold:
            print(
                f"  Stage 1 stopped at epoch {epoch + 1}: "
                f"train_recon {train_recon:.5f} <= threshold {recon_threshold:.5f}"
            )
            break

        if lambda_vol == 0 and patience_counter >= patience:
            print(f"  Stage 1 early stopping at epoch {epoch + 1} (train-loss plateau)")
            break

    return best_train_loss


def train_stage2(
    encoder: BoxEncoder,
    z_targets: torch.Tensor,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    patience: int,
    run: "wandb.sdk.wandb_run.Run",
    encoder_input: str = "half",
    name_tag: str = "",
) -> None:
    """Train the encoder to reproduce the codes stage 1 learned.

    This is a plain regression: the decoder takes no part in it and the targets do not move, so
    the encoder has something definite to hit and cannot drift into a degenerate solution
    together with the decoder. Unlike stage 1 this stage does have a validation split, since the
    encoder is a function of the shape and can therefore be asked about a shape it has not seen.

    Args:
        z_targets: The frozen codes of stage 1, shape ``(N_shapes, latent_dim)``, indexed by the
            global shape index the loaders carry.
        encoder_input: ``surface`` feeds the surface points to the point cloud encoder,
            ``half`` feeds the half extents to the box encoder.
    """
    z_targets = z_targets.to(device)

    optimizer = torch.optim.Adam(encoder.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=7, min_lr=1e-6
    )

    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(epochs):
        encoder.train()
        t_loss = 0.0
        for idx, surf, _q, _s, half_extents in tqdm(
            train_loader, desc=f"S2 E{epoch + 1}", leave=False
        ):
            idx = idx.to(device)
            enc_in = (surf if encoder_input == "surface" else half_extents).to(device)

            z_pred = encoder(enc_in)
            z_target = z_targets[idx].detach()
            loss = F.mse_loss(z_pred, z_target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            t_loss += loss.item()

        encoder.eval()
        v_loss = 0.0
        val_z_preds: list[torch.Tensor] = []
        with torch.no_grad():
            for idx, surf, _q, _s, half_extents in val_loader:
                idx = idx.to(device)
                enc_in = (surf if encoder_input == "surface" else half_extents).to(device)
                z_pred = encoder(enc_in)
                z_target = z_targets[idx].detach()
                loss = F.mse_loss(z_pred, z_target)
                v_loss += loss.item()
                val_z_preds.append(z_pred.cpu())

        n, nv = len(train_loader), len(val_loader)
        val_loss = v_loss / nv

        # Measured on the encoder's own predictions, not on the targets: a low regression error
        # on average is still compatible with the encoder having flattened the codes together.
        z_pred_all = torch.cat(val_z_preds, dim=0)
        div = z_diversity_metrics(z_pred_all)
        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"S2 E{epoch + 1}/{epochs} | "
            f"train {t_loss/n:.6f} | val {val_loss:.6f} | "
            f"pred z_norm {div['z_norm_mean']:.3f}±{div['z_norm_std']:.3f} | "
            f"pred cos mean={div['cos_sim_mean']:.3f} max={div['cos_sim_max']:.3f} | "
            f"LR {current_lr:.2e}"
        )

        run.log(
            {
                "stage2/train_loss": t_loss / n,
                "stage2/val_loss": val_loss,
                "stage2/pred_z_norm_mean": div["z_norm_mean"],
                "stage2/pred_z_norm_std": div["z_norm_std"],
                "stage2/pred_z_norm_min": div["z_norm_min"],
                "stage2/pred_z_norm_max": div["z_norm_max"],
                "stage2/pred_cos_mean": div["cos_sim_mean"],
                "stage2/pred_cos_std": div["cos_sim_std"],
                "stage2/pred_cos_max": div["cos_sim_max"],
                "stage2/lr": current_lr,
                "stage2/epoch": epoch,
            }
        )

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            ckpt = {
                "encoder": encoder.state_dict(),
                "latent_dim": encoder.latent_dim,
                "encoder_type": "pointnet" if encoder_input == "surface" else "box",
                "encoder_spectral": getattr(encoder, "spectral", False),
                "timestamp": TIMESTAMP,
            }
            latent_dim_s2 = config.autoencoder.latent_dim
            unsuffixed_s2 = MODEL_FOLDER / f"stage2_best_encoder{name_tag}_latentdim{latent_dim_s2}.pth"
            archived_s2 = MODEL_FOLDER / f"stage2_best_encoder{name_tag}_latentdim{latent_dim_s2}_{TIMESTAMP}.pth"
            torch.save(ckpt, unsuffixed_s2)
            torch.save(ckpt, archived_s2)
            print(f"  -> Saved {unsuffixed_s2.name} + {archived_s2.name}  (val loss {val_loss:.6f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Stage 2 early stopping at epoch {epoch + 1}")
                break


def train_stage3(
    encoder: BoxEncoder,
    decoder: SDFDecoder,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    lambda_reg: float,
    delta: float,
    run: "wandb.sdk.wandb_run.Run",
    encoder_input: str = "half",
    name_tag: str = "",
    lambda_vol: float = 0.0,
    vol_eta: float = 0.1,
) -> None:
    """Train encoder and decoder together, at a small learning rate, and compress the code.

    The objective is the same reconstruction loss as in stage 1, but the code is now the output
    of the encoder rather than a free parameter. That is the difference that makes the volume
    penalty do what it is meant to: distinct shapes must receive distinct codes or they cannot
    be reconstructed, so the penalty can only remove what carries nothing. Applied in stage 1
    instead, where nothing links a shape to its code, the same penalty simply starves the codes.

    The penalty is evaluated on the codes of the current batch, so it estimates the spread from
    a sample; the batch has to be large enough for that estimate to be stable. The learning rate
    is kept constant when the penalty is on, and the epoch to keep is chosen on the objective
    including the penalty rather than on the reconstruction alone.

    Args:
        lambda_vol: Weight of the volume penalty. Zero disables it and restores the plateau
            schedule and a selection on the reconstruction alone.
        vol_eta: Offset of the volume penalty.
    """
    for p in decoder.parameters():
        p.requires_grad_(True)

    optimizer = torch.optim.Adam(
        [
            {"params": encoder.parameters(), "lr": lr},
            {"params": decoder.parameters(), "lr": lr},
        ]
    )
    # Compression is slow, and a schedule that cuts the rate on a plateau of the reconstruction
    # would stop it before it has finished. Only the runs without the penalty use one.
    scheduler = None if lambda_vol > 0 else torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-7
    )

    best_val_loss = float("inf")

    for epoch in range(epochs):
        encoder.train()
        decoder.train()
        t_total = t_recon = t_reg = t_vol = 0.0
        val_z_preds: list[torch.Tensor] = []

        for _idx, surf, query_pts, sdf_gt, half_extents in tqdm(
            train_loader, desc=f"S3 E{epoch + 1}", leave=False
        ):
            enc_in = (surf if encoder_input == "surface" else half_extents).to(device)
            query_pts = query_pts.to(device)
            sdf_gt = sdf_gt.to(device)

            z = encoder(enc_in)
            sdf_pred = decode_batch(decoder, z, query_pts)
            loss, recon, reg = sdf_reconstruction_loss(
                sdf_pred, sdf_gt, z, lambda_reg, delta
            )
            if lambda_vol > 0:
                # On the codes of this batch only. Unlike stage 1 there is no stored table to
                # take the spread from, since the codes are produced on the fly.
                vol = volume_penalty(z, vol_eta)
                loss = loss + lambda_vol * vol
                t_vol += float(vol.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            t_total += loss.item()
            t_recon += recon.item()
            t_reg += reg.item()

        encoder.eval()
        decoder.eval()
        v_total = 0.0
        with torch.no_grad():
            for _idx, surf, query_pts, sdf_gt, half_extents in val_loader:
                enc_in = (surf if encoder_input == "surface" else half_extents).to(device)
                query_pts = query_pts.to(device)
                sdf_gt = sdf_gt.to(device)
                z = encoder(enc_in)
                sdf_pred = decode_batch(decoder, z, query_pts)
                loss, _r, _g = sdf_reconstruction_loss(
                    sdf_pred, sdf_gt, z, lambda_reg, delta
                )
                v_total += loss.item()
                val_z_preds.append(z.cpu())

        n, nv = len(train_loader), len(val_loader)
        val_loss = v_total / nv
        val_z = torch.cat(val_z_preds, dim=0)
        div = z_diversity_metrics(val_z)
        current_lr = optimizer.param_groups[0]["lr"]

        sig = sigma_spectrum(val_z)
        adims = active_dims(val_z)
        adims10 = active_dims(val_z, frac=0.10)
        # Over the whole validation split rather than per batch, so the spread is not an
        # estimate here, and the epoch is selected on the objective that is actually minimised.
        # Selecting on the reconstruction alone would always prefer the least compressed epoch.
        val_vol = float(volume_penalty(val_z, vol_eta).item()) if lambda_vol > 0 else 0.0
        select_loss = val_loss + lambda_vol * val_vol
        vol_str = (f" | vol {t_vol/n:.4f} active {adims}/{adims10} sig_min {sig[-1]:.4f}"
                   if lambda_vol > 0 else "")

        print(
            f"S3 E{epoch + 1}/{epochs} | "
            f"train {t_total/n:.5f} (recon {t_recon/n:.5f}) | val {val_loss:.5f} | "
            f"z_norm {div['z_norm_mean']:.3f}±{div['z_norm_std']:.3f} | "
            f"cos mean={div['cos_sim_mean']:.3f} | LR {current_lr:.2e}{vol_str}"
        )

        run.log(
            {
                "stage3/train_total": t_total / n,
                "stage3/train_recon": t_recon / n,
                "stage3/train_vol": t_vol / n,
                "stage3/val_total": val_loss,
                "stage3/val_select": select_loss,
                "stage3/active_dims": adims,
                "stage3/active_dims10": adims10,
                "stage3/sigma_min": float(sig[-1]),
                "stage3/z_norm_mean": div["z_norm_mean"],
                "stage3/z_norm_std": div["z_norm_std"],
                "stage3/cos_sim_mean": div["cos_sim_mean"],
                "stage3/lr": current_lr,
                "stage3/epoch": epoch,
            }
        )

        if scheduler is not None:
            scheduler.step(val_loss)

        if select_loss < best_val_loss:
            best_val_loss = select_loss
            ckpt = {
                "encoder": encoder.state_dict(),
                "decoder": decoder.state_dict(),
                "latent_dim": encoder.latent_dim,
                "encoder_type": "pointnet" if encoder_input == "surface" else "box",
                "encoder_spectral": getattr(encoder, "spectral", False),
                "decoder_spectral": getattr(decoder, "spectral", False),
                "lipschitz_k": getattr(decoder, "out_scale", 1.0),
                "timestamp": TIMESTAMP,
            }
            latent_dim = config.autoencoder.latent_dim
            unsuffixed = MODEL_FOLDER / f"best_encoder_decoder{name_tag}_latentdim{latent_dim}.pth"
            archived = MODEL_FOLDER / f"best_encoder_decoder{name_tag}_latentdim{latent_dim}_{TIMESTAMP}.pth"
            torch.save(ckpt, unsuffixed)
            torch.save(ckpt, archived)
            print(
                f"  -> Saved {unsuffixed.name} + {archived.name}  (val {val_loss:.5f})"
            )


def main() -> None:
    """Read the settings, generate the shapes and run the stages that are requested."""
    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    import os
    general_shapes = bool(int(os.environ.get("GENERAL_SHAPES", "0")))
    ae_quick = bool(int(os.environ.get("AE_QUICK", "0")))
    canonical = bool(int(os.environ.get("AE_CANON_NORM", "0")))
    enc_input = "surface" if general_shapes else "half"
    # Everything below comes from the environment with the configuration as the fallback. The
    # sweep of the thesis varies these per job, and they have to differ between jobs that share
    # one checkout, so they cannot live in a file.
    lambda_vol = float(os.environ.get("LV_LAMBDA_VOL", str(config.training.lambda_vol)))
    vol_eta = float(os.environ.get("LV_VOL_ETA", str(config.training.vol_eta)))
    lv_spectral = bool(int(os.environ.get("LV_SPECTRAL", str(int(config.training.lv_spectral)))))
    lv_enc_spectral = bool(int(os.environ.get("LV_ENC_SPECTRAL", str(int(config.training.lv_enc_spectral)))))
    lipschitz_k = float(os.environ.get("LV_LIPSCHITZ_K", str(config.training.lipschitz_k)))
    # Start stage 1 from an already trained autoencoder instead of from noise: the decoder is
    # loaded and every code is set to what that encoder produces for its shape. Beginning from
    # a reconstruction that already works is what lets the penalty be resisted rather than
    # obeyed immediately.
    lv_warmstart = bool(int(os.environ.get("LV_WARMSTART", "0")))
    lv_warmstart_ckpt = os.environ.get("LV_WARMSTART_CKPT", "")
    # Skip stages 1 and 2 entirely and run only the compression on top of an existing
    # autoencoder. This is the path the sweep of the thesis takes, since only stage 3 differs
    # between its runs and re-training the first two stages per run would change nothing.
    lv_s3_only = bool(int(os.environ.get("LV_S3_ONLY", "0")))
    lv_s3_ckpt = os.environ.get("LV_S3_CKPT", "")
    # The tag is what separates one run's checkpoints from another's. Without it a compressed
    # run would overwrite the uncompressed autoencoder it was started from, and two jobs of the
    # sweep would overwrite each other.
    name_tag = "_general" if general_shapes else ""
    if canonical:
        name_tag += "_canon"
    if lambda_vol > 0:
        name_tag += "_lv"
    name_tag += os.environ.get("AE_RUN_TAG", "")
    lv_str = f"  [Least Volume lambda_vol={lambda_vol} eta={vol_eta}]" if lambda_vol > 0 else ""
    sn_str = ""
    if lv_spectral:
        sn_str = f"  [spectral dec K={lipschitz_k}" + ("+enc]" if lv_enc_spectral else "]")
    ws_str = "  [warm-start]" if lv_warmstart else ""
    print(f"Encoder: {'PointNet (surface points)' if general_shapes else 'BoxEncoder (half-extents)'}"
          f"{'  [AE_QUICK smoke]' if ae_quick else ''}"
          f"{'  [canonical norm]' if canonical else ''}"
          f"{lv_str}{sn_str}{ws_str}"
          f"  ckpt tag='{name_tag or '(none)'}'")

    # Hyperparameters from config
    n_shapes: int = config.autoencoder.n_shapes
    n_surface: int = config.autoencoder.n_surface
    n_query: int = config.autoencoder.n_query
    batch_size: int = config.autoencoder.batch_size
    latent_dim: int = config.autoencoder.latent_dim
    lambda_reg: float = config.training.lambda_reg
    delta: float = config.training.sdf_clamp_delta

    stage1_epochs: int = config.training.stage1_epochs
    stage1_lr: float = config.training.stage1_lr
    stage1_patience: int = config.training.stage1_patience
    # The stop is on the mean over all shapes, and the mean hides the hard ones: the curved
    # members are still far from converged when it is reached, so the threshold has to be low
    # enough that the stage keeps training for them.
    stage1_recon_threshold: float = float(os.environ.get("AE_S1_RECON_THRESH", "0.002"))
    stage2_epochs: int = config.training.stage2_epochs
    stage2_lr: float = config.training.stage2_lr
    stage2_patience: int = config.training.stage2_patience
    stage3_epochs: int = config.training.stage3_epochs
    stage3_lr: float = config.training.stage3_lr

    if ae_quick:
        n_shapes, batch_size = 300, 64
        stage1_epochs = stage2_epochs = 3
        stage3_epochs = 2
        stage1_patience = stage2_patience = 10**6
        print(f"AE_QUICK: n_shapes={n_shapes}, batch={batch_size}, stages 3/3/2 epochs")
    else:
        n_shapes = int(os.environ.get("AE_N_SHAPES", str(n_shapes)))
        # The peak memory of stage 1 is set by the product of these two, since the backward
        # pass keeps one activation per query point of the batch. They are overridable per run
        # so that a job can be fitted to whichever card it lands on.
        batch_size = int(os.environ.get("AE_BATCH", str(batch_size)))
        n_query = int(os.environ.get("AE_NQUERY", str(n_query)))
        stage1_epochs = int(os.environ.get("AE_STAGE1_EPOCHS", str(stage1_epochs)))
        stage2_epochs = int(os.environ.get("AE_STAGE2_EPOCHS", str(stage2_epochs)))
        stage3_epochs = int(os.environ.get("AE_STAGE3_EPOCHS", str(stage3_epochs)))
        stage1_lr = float(os.environ.get("AE_STAGE1_LR", str(stage1_lr)))

    run = wandb.init(
        project="train_autoencoder_twostage",
        config={
            "n_shapes": n_shapes,
            "n_surface": n_surface,
            "n_query": n_query,
            "batch_size": batch_size,
            "latent_dim": latent_dim,
            "lambda_reg": lambda_reg,
            "sdf_clamp_delta": delta,
            "lambda_vol": lambda_vol,
            "vol_eta": vol_eta,
            "lv_spectral": lv_spectral,
            "lv_enc_spectral": lv_enc_spectral,
            "lipschitz_k": lipschitz_k,
            "lv_warmstart": lv_warmstart,
            "stage1_epochs": stage1_epochs,
            "stage1_lr": stage1_lr,
            "stage1_recon_threshold": stage1_recon_threshold,
            "stage2_epochs": stage2_epochs,
            "stage2_lr": stage2_lr,
            "stage3_epochs": stage3_epochs,
            "stage3_lr": stage3_lr,
        },
    )

    # The shapes are generated even for the compression-only path, because the split has to be
    # the same one the autoencoder being compressed was trained under.
    print(f"\n{'='*60}")
    print(
        f"Generating {n_shapes} shapes ({n_surface} surface + {n_query} query pts)..."
    )
    print(f"{'='*60}")
    t0 = time.time()
    base_dataset = AutoEncoderDataset(
        n_shapes=n_shapes, n_surface=n_surface, n_query=n_query, canonical=canonical
    )
    indexed_dataset = IndexedDataset(base_dataset)
    print(f"Data generation: {time.time() - t0:.1f}s")

    if lv_s3_only:
        if not general_shapes:
            raise SystemExit("LV_S3_ONLY needs the general PointNet codec (set GENERAL_SHAPES=1).")
        from src.enc_pointnet import pointnet_encoder_from_ckpt
        from src.dec_sdf import sdf_decoder_from_ckpt
        ref_path = Path(lv_s3_ckpt) if lv_s3_ckpt else (
            MODEL_FOLDER / f"best_encoder_decoder_general_canon_latentdim{latent_dim}.pth")
        if not ref_path.is_absolute():
            ref_path = Path(__file__).parent.parent / ref_path
        if not ref_path.exists():
            raise SystemExit(f"LV_S3_ONLY checkpoint not found: {ref_path}")
        ck = torch.load(ref_path, map_location=device, weights_only=False)
        encoder = pointnet_encoder_from_ckpt(ck, device=device, latent_dim=latent_dim)
        if lv_spectral:
            # Only the encoder is carried over. A spectrally normalised decoder cannot inherit
            # the weights of a weight-normalised one, since the two re-parametrise the same
            # matrix differently, so it starts fresh from the loaded encoder's codes.
            decoder = SDFDecoder(latent_dim=latent_dim, spectral=True, lipschitz_k=lipschitz_k).to(device)
            print(f"S3-only: loaded encoder from {ref_path.name}; FRESH spectral decoder "
                  f"(K={lipschitz_k}) -> Stage-3 LV (true-AE + Lipschitz decoder)")
        else:
            decoder = sdf_decoder_from_ckpt(ck, device=device, latent_dim=latent_dim)
            print(f"S3-only: loaded codec {ref_path.name} -> running Stage-3 LV only")

        n_train = int(config.training.train_split * n_shapes)
        n_val = int(config.training.val_split * n_shapes)
        n_test = n_shapes - n_train - n_val
        # The same seeded split as the full run below, so that no shape the loaded autoencoder
        # trained on ends up in the validation set of the compression.
        generator = torch.Generator().manual_seed(config.general.random_seed or 42)
        train_ds, val_ds, _ = torch.utils.data.random_split(
            indexed_dataset, [n_train, n_val, n_test], generator=generator)
        train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size)
        s3_ep = stage3_epochs if stage3_epochs > 0 else 300
        train_stage3(
            encoder=encoder, decoder=decoder, train_loader=train_loader, val_loader=val_loader,
            device=device, epochs=s3_ep, lr=stage3_lr, lambda_reg=lambda_reg, delta=delta,
            run=run, encoder_input=enc_input, name_tag=name_tag,
            lambda_vol=lambda_vol, vol_eta=vol_eta,
        )
        run.finish()
        print("\nStage-3-only LV complete!")
        return

    stage1_loader = torch.utils.data.DataLoader(
        indexed_dataset, batch_size=batch_size, shuffle=True
    )
    print(f"Stage 1 loader: {len(indexed_dataset)} shapes (no val/test split)")

    print(f"\n{'='*60}")
    print(f"STAGE 1 — Auto-Decoder  (max {stage1_epochs} epochs)")
    print(f"{'='*60}")

    decoder = SDFDecoder(latent_dim=latent_dim, spectral=lv_spectral, lipschitz_k=lipschitz_k).to(device)
    latent_codes = nn.Embedding(n_shapes, latent_dim).to(device)
    # Initialised small but not at zero: identical codes would give every shape the same
    # gradient at the first step, which is the collapse the stage exists to avoid.
    nn.init.normal_(latent_codes.weight, mean=0.0, std=0.1)

    if lv_warmstart:
        if not general_shapes:
            raise SystemExit("LV_WARMSTART needs the general PointNet codec (set GENERAL_SHAPES=1).")
        from src.enc_pointnet import pointnet_encoder_from_ckpt
        ref_path = Path(lv_warmstart_ckpt) if lv_warmstart_ckpt else (
            MODEL_FOLDER / f"best_encoder_decoder_general_canon_latentdim{latent_dim}.pth")
        if not ref_path.is_absolute():
            ref_path = Path(__file__).parent.parent / ref_path
        if not ref_path.exists():
            raise SystemExit(f"LV_WARMSTART checkpoint not found: {ref_path}")
        ref_ck = torch.load(ref_path, map_location=device, weights_only=False)
        if lv_spectral or ref_ck.get("decoder_spectral", False):
            raise SystemExit("LV_WARMSTART expects a non-spectral reference + non-spectral run "
                             "(spectral parametrisation can't be warm-started from weight_norm).")
        decoder.load_state_dict(ref_ck["decoder"])
        # Every code is set to what the reference encoder produces for its shape, so the free
        # codes start where the loaded decoder already reconstructs well rather than at noise.
        ref_enc = pointnet_encoder_from_ckpt(ref_ck, device=device, latent_dim=latent_dim).eval()
        with torch.no_grad():
            # In blocks, because encoding all shapes at once would not fit on the card.
            for c in range(0, n_shapes, 256):
                surf = torch.stack([torch.from_numpy(base_dataset.surface_points[i])
                                    for i in range(c, min(c + 256, n_shapes))]).to(device)
                latent_codes.weight[c:min(c + 256, n_shapes)] = ref_enc(surf)
        zn = latent_codes.weight.norm(dim=1)
        print(f"Warm-started from {ref_path.name}: decoder loaded, "
              f"codes = encoder(surface)  z_norm {zn.mean():.3f}±{zn.std():.3f}")

    dec_params = sum(p.numel() for p in decoder.parameters())
    print(f"Decoder params:  {dec_params:,}")
    print(f"Latent codes:    {n_shapes * latent_dim:,}  ({n_shapes} × {latent_dim})")

    train_stage1(
        decoder=decoder,
        latent_codes=latent_codes,
        stage1_loader=stage1_loader,
        device=device,
        epochs=stage1_epochs,
        lr=stage1_lr,
        patience=stage1_patience,
        recon_threshold=stage1_recon_threshold,
        lambda_reg=lambda_reg,
        delta=delta,
        run=run,
        name_tag=name_tag,
        # Passed as zero regardless of the setting: stage 1 has no encoder, so the penalty has
        # nothing to press against and would simply shrink the codes. It belongs in stage 3.
        lambda_vol=0.0,
        vol_eta=vol_eta,
    )

    s1_ckpt = torch.load(
        MODEL_FOLDER / f"stage1_best{name_tag}_latentdim{latent_dim}.pth", map_location=device
    )
    decoder.load_state_dict(s1_ckpt["decoder"])
    latent_codes.load_state_dict(s1_ckpt["latent_codes"])

    all_z = latent_codes.weight.detach().cpu()
    div = z_diversity_metrics(all_z, n_pairs=2000)
    print(
        f"\nStage 1 final z diversity:\n"
        f"  norm:  mean={div['z_norm_mean']:.4f}  std={div['z_norm_std']:.4f}"
        f"  min={div['z_norm_min']:.4f}  max={div['z_norm_max']:.4f}\n"
        f"  cos:   mean={div['cos_sim_mean']:.4f}  std={div['cos_sim_std']:.4f}"
        f"  max={div['cos_sim_max']:.4f}"
    )
    run.log(
        {
            "stage1_final/z_norm_mean": div["z_norm_mean"],
            "stage1_final/z_norm_std": div["z_norm_std"],
            "stage1_final/cos_sim_mean": div["cos_sim_mean"],
            "stage1_final/cos_sim_max": div["cos_sim_max"],
        }
    )

    n_train = int(config.training.train_split * n_shapes)
    n_val = int(config.training.val_split * n_shapes)
    n_test = n_shapes - n_train - n_val

    # Seeded, because stage 2 looks its targets up by the global shape index. An unseeded split
    # would pair a shape with another shape's code and the run would silently learn nonsense.
    generator = torch.Generator().manual_seed(config.general.random_seed or 42)
    train_ds, val_ds, test_ds = torch.utils.data.random_split(
        indexed_dataset, [n_train, n_val, n_test], generator=generator
    )
    print(f"\nSplit for Stage 2/3: {n_train} train / {n_val} val / {n_test} test")

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size)

    print(f"\n{'='*60}")
    print(f"STAGE 2 — Encoder Distillation  (max {stage2_epochs} epochs)")
    print(f"{'='*60}")

    for p in decoder.parameters():
        p.requires_grad_(False)

    if general_shapes:
        from src.enc_pointnet import Autoencoder as PointNetEncoder
        encoder = PointNetEncoder(latent_dim=latent_dim, spectral=lv_enc_spectral).to(device)
    else:
        encoder = BoxEncoder(latent_dim=latent_dim).to(device)
    enc_params = sum(p.numel() for p in encoder.parameters())
    print(f"{'PointNet' if general_shapes else 'Box'}Encoder params: {enc_params:,}")

    # The codes are taken exactly as stage 1 left them, with no rescaling or centring: they are
    # the input the frozen decoder was fitted to, and any transformation would invalidate it.
    z_frozen = latent_codes.weight.detach().cpu()

    train_stage2(
        encoder=encoder,
        z_targets=z_frozen,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=stage2_epochs,
        lr=stage2_lr,
        patience=stage2_patience,
        run=run,
        encoder_input=enc_input,
        name_tag=name_tag,
    )

    s2_ckpt = torch.load(
        MODEL_FOLDER / f"stage2_best_encoder{name_tag}_latentdim{latent_dim}.pth",
        map_location=device,
    )
    encoder.load_state_dict(s2_ckpt["encoder"])

    if stage3_epochs > 0:
        print(f"\n{'='*60}")
        lv_note = f" + Least Volume (lambda_vol={lambda_vol}, eta={vol_eta})" if lambda_vol > 0 else ""
        print(
            f"STAGE 3 — Joint Fine-Tune  (max {stage3_epochs} epochs, LR={stage3_lr}){lv_note}"
        )
        print(f"{'='*60}")
        train_stage3(
            encoder=encoder,
            decoder=decoder,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            epochs=stage3_epochs,
            lr=stage3_lr,
            lambda_reg=lambda_reg,
            delta=delta,
            run=run,
            encoder_input=enc_input,
            name_tag=name_tag,
            lambda_vol=lambda_vol,
            vol_eta=vol_eta,
        )
        # Stage 3 writes the final checkpoint itself, on its best epoch.
    else:
        # Without stage 3 nothing has yet written a checkpoint holding both halves, so the
        # decoder of stage 1 and the encoder of stage 2 are saved together here.
        ckpt = {
            "encoder": encoder.state_dict(),
            "decoder": decoder.state_dict(),
            "latent_dim": latent_dim,
            "encoder_type": "pointnet" if general_shapes else "box",
            "encoder_spectral": getattr(encoder, "spectral", False),
            "decoder_spectral": getattr(decoder, "spectral", False),
            "lipschitz_k": getattr(decoder, "out_scale", 1.0),
            "timestamp": TIMESTAMP,
        }
        unsuffixed = MODEL_FOLDER / f"best_encoder_decoder{name_tag}_latentdim{latent_dim}.pth"
        archived = MODEL_FOLDER / f"best_encoder_decoder{name_tag}_latentdim{latent_dim}_{TIMESTAMP}.pth"
        torch.save(ckpt, unsuffixed)
        torch.save(ckpt, archived)
        print(f"\nFinal checkpoint saved: {unsuffixed} + {archived.name}")

    # A last check on shapes no stage was trained on: if the codes are distinct here, the
    # encoder has learned to describe form rather than to memorise the training shapes.
    encoder.eval()
    decoder.eval()
    test_z_preds: list[torch.Tensor] = []
    with torch.no_grad():
        for _idx, surf, _q, _s, half_extents in test_loader:
            enc_in = (surf if enc_input == "surface" else half_extents).to(device)
            test_z_preds.append(encoder(enc_in).cpu())

    test_z = torch.cat(test_z_preds, dim=0)
    div_test = z_diversity_metrics(test_z, n_pairs=2000)
    print(
        f"\nTest-set encoder z diversity:\n"
        f"  norm:  mean={div_test['z_norm_mean']:.4f}  std={div_test['z_norm_std']:.4f}"
        f"  min={div_test['z_norm_min']:.4f}  max={div_test['z_norm_max']:.4f}\n"
        f"  cos:   mean={div_test['cos_sim_mean']:.4f}  std={div_test['cos_sim_std']:.4f}"
        f"  max={div_test['cos_sim_max']:.4f}"
    )
    run.log(
        {
            "final/test_z_norm_mean": div_test["z_norm_mean"],
            "final/test_z_norm_std": div_test["z_norm_std"],
            "final/test_cos_mean": div_test["cos_sim_mean"],
            "final/test_cos_max": div_test["cos_sim_max"],
        }
    )

    run.finish()
    print("\nTraining complete!")


if __name__ == "__main__":
    main()
