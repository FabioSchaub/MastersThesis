"""Three-stage training that gives the box encoder a latent space with a geometric meaning.

Training an encoder and a decoder jointly from scratch invites the decoder to ignore the code
and reproduce an average shape, at which point the encoder is free to map every box to the same
vector. The three stages avoid that by fitting the codes first and the encoder second.

    Stage 1  Trains the auxiliary signed-distance decoder of ``src/dec_sdf.py`` together with a
             free code per training shape, held in an embedding table. The encoder takes no
             part. Because each code is an ordinary parameter with a gradient of its own, the
             codes cannot collapse onto one another, and reconstructing the field is what ties
             a code to a geometry.
    Stage 2  Freezes that decoder and the codes and trains the box encoder of
             ``src/enc_box.py`` alone, to reproduce the stage-1 code of a shape from its three
             half-extents.
    Stage 3  Unfreezes the auxiliary decoder and continues on both at a much smaller learning
             rate, against the reconstruction loss again. Skipped when
             ``config.training.stage3_epochs`` is zero.

The encoder is the artefact all of Part II depends on; the auxiliary decoder is scaffolding and
is not used again after training. The third network, the box decoder that maps a code back to
three half-extents and stands inside the repair loop, is a separate model trained afterwards by
``tools/train_box_decoder.py`` on codes of the frozen encoder, and is not trained here.

Stage 1 writes ``stage1_best_latentdim{N}.pth``, stage 2 writes
``stage2_best_encoder_latentdim{N}.pth``, and the final encoder and auxiliary decoder land in
``best_encoder_decoder_latentdim{N}.pth`` under ``config.autoencoder.autoencoder_folder``. That
last file is what ``src/gnn_dataset_preparation.py`` loads the encoder from. On the cluster the
entry point is ``autoencoder_train_twostage.slurm``.
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

MODEL_FOLDER: Path = (
    Path(__file__).parent.parent / config.autoencoder.autoencoder_folder
)
MODEL_FOLDER.mkdir(parents=True, exist_ok=True)

# Fixed once at import, so that every checkpoint of one run carries the same suffix. Each save
# writes twice: an unsuffixed file that downstream code always finds, and a timestamped archive
# so an earlier run is not overwritten by a later one.
TIMESTAMP: str = time.strftime("%Y%m%d-%H%M%S")


def set_seed(seed: int = config.general.random_seed or 42) -> None:
    """Seed every generator the run uses and put cuDNN into deterministic mode."""
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


class IndexedDataset(torch.utils.data.Dataset):
    """Wraps the shape dataset so that every sample also carries its position in the table.

    Stage 1 stores one free code per shape in an embedding, and stage 2 needs the code that
    belongs to the shape in front of it. Both look the code up by this index, so it has to
    survive shuffling and the train/validation split.

    Each item is ``(idx, surface_points, query_points, sdf_values, half_extents)``, where the
    half-extents are ``(3,)`` in metres and are the input the encoder is trained on.
    """

    def __init__(self, base_dataset: AutoEncoderDataset) -> None:
        self.base = base_dataset

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(
        self, idx: int
    ) -> tuple[int, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        surface, query, sdf = self.base[idx]
        half = torch.from_numpy(self.base.half_extents[idx])
        return idx, surface, query, sdf, half


def decode_batch(
    decoder: SDFDecoder,
    z: torch.Tensor,
    query_pts: torch.Tensor,
) -> torch.Tensor:
    """Evaluate the auxiliary decoder at every query point of every shape in a batch.

    Args:
        z: Latent codes, shape ``(B, latent_dim)``, one per shape.
        query_pts: Query coordinates, shape ``(B, N_query, 3)``, in the normalised frame the
            samples were generated in.

    Returns:
        Predicted signed distances, shape ``(B, N_query)``.
    """
    B, N_query, _ = query_pts.shape
    latent_dim = z.shape[1]
    # One code per shape is repeated across that shape's query points and flattened into a
    # single call, so the whole batch costs one forward pass instead of B of them.
    z_exp = z.unsqueeze(1).expand(B, N_query, latent_dim)
    inp = torch.cat([z_exp, query_pts], dim=-1).reshape(B * N_query, latent_dim + 3)
    return decoder.net(inp).squeeze(-1).reshape(B, N_query)


def sdf_reconstruction_loss(
    sdf_pred: torch.Tensor,
    sdf_gt: torch.Tensor,
    z: torch.Tensor,
    lambda_reg: float,
    delta: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Clamped absolute error on the signed distance, plus a penalty on the code norm.

    Args:
        sdf_pred, sdf_gt: Predicted and true signed distances, shape ``(B, N_query)``.
        z: The codes the prediction was conditioned on, shape ``(B, latent_dim)``.
        lambda_reg: Weight of the code penalty, ``config.training.lambda_reg``.
        delta: Half-width of the band the loss is computed in, in the units of the field.

    Returns:
        A tuple ``(total_loss, recon_loss, reg_loss)``, the first being the second plus
        ``lambda_reg`` times the third.
    """
    # Both sides are clamped before the comparison, so a point far inside or far outside the
    # shape contributes a constant and the loss concentrates on the band around the surface.
    # That band is where the geometry actually is; matching the field far away is easy and
    # would otherwise dominate the average.
    sdf_pred_c = torch.clamp(sdf_pred, -delta, delta)
    sdf_gt_c = torch.clamp(sdf_gt, -delta, delta)
    recon = F.l1_loss(sdf_pred_c, sdf_gt_c)
    # Stage 1 optimises the codes directly, so nothing else bounds them. Without this term they
    # can grow without limit and the space they span stops being usable.
    reg = torch.mean(z.norm(dim=1) ** 2)
    return recon + lambda_reg * reg, recon, reg


@torch.no_grad()
def z_diversity_metrics(
    z_matrix: torch.Tensor, n_pairs: int = 1000
) -> dict[str, float]:
    """Measure whether a set of codes has collapsed, from their norms and mutual angles.

    This is the diagnostic the three-stage design exists for. If the codes collapse, every
    shape ends up pointing the same way and the mean cosine similarity approaches one, so the
    quantity is watched at every epoch rather than only at the end.

    Args:
        z_matrix: Codes, shape ``(N, latent_dim)``.
        n_pairs: How many random pairs the cosine statistics are estimated from. All ``N``
            choose two pairs would be prohibitive for the shape counts used here.

    Returns:
        A dictionary with the mean, standard deviation, minimum and maximum of the code norms,
        and the mean, standard deviation and maximum of the pairwise cosine similarity.
    """
    norms = z_matrix.norm(dim=1)
    N = z_matrix.shape[0]
    idx_a = torch.randint(0, N, (n_pairs,))
    idx_b = torch.randint(0, N, (n_pairs,))
    # A code compared with itself has similarity one and would bias the mean upwards, which is
    # exactly the direction collapse would show up in.
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
) -> float:
    """Fit the auxiliary decoder and one free code per shape against the signed-distance field.

    The encoder does not exist yet at this point. Every code is a row of ``latent_codes`` and
    receives its own gradient, which is what makes the resulting set of codes diverse and gives
    each of them a geometric meaning to be distilled in stage 2.

    There is no validation split here: the objective is to fit a code to each of these
    particular shapes, not to generalise to unseen ones, so the stopping rule watches the
    training loss.

    Args:
        decoder: The auxiliary signed-distance decoder, trained in place.
        latent_codes: Embedding of shape ``(n_shapes, latent_dim)``, trained in place. The row
            index is the global shape index supplied by :class:`IndexedDataset`.
        epochs: Maximum epochs, ``config.training.stage1_epochs``.
        lr: Learning rate for the decoder, ``config.training.stage1_lr``.
        patience: Epochs without improvement before stopping, ``config.training.stage1_patience``.
        recon_threshold: Reconstruction error at which the stage stops satisfied, in the units
            of the signed-distance field.
        lambda_reg: Weight of the penalty on the code norms.
        delta: Half-width of the band the reconstruction loss is computed in.
        run: Weights and Biases run the per-epoch metrics are logged to.

    Returns:
        The best training loss reached.
    """
    optimizer = torch.optim.Adam(
        [
            {"params": decoder.parameters(), "lr": lr},
            {
                # The codes are moved faster than the decoder. They are the quantity this
                # stage is really producing, and a code that adapts more slowly than the
                # network reading it lets the decoder explain the data on its own instead.
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
        t_total = t_recon = t_reg = 0.0

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

            optimizer.zero_grad()
            loss.backward()
            # Clipped separately for the two parameter groups, because they are on different
            # learning rates. The sine variant of the decoder in particular produces large
            # gradients that would otherwise drive the code norms up.
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

        all_z = latent_codes.weight.detach().cpu()
        div = z_diversity_metrics(all_z)
        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"S1 E{epoch + 1}/{epochs} | "
            f"train {train_loss:.5f} (recon {train_recon:.5f}, reg {train_reg:.5f}) | "
            f"z_norm {div['z_norm_mean']:.3f}±{div['z_norm_std']:.3f} | "
            f"cos mean={div['cos_sim_mean']:.3f} max={div['cos_sim_max']:.3f} | "
            f"LR {current_lr:.2e}"
        )

        run.log(
            {
                "stage1/train_total": train_loss,
                "stage1/train_recon": train_recon,
                "stage1/train_reg": train_reg,
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

        scheduler.step(train_loss)

        if train_loss < best_train_loss:
            best_train_loss = train_loss
            patience_counter = 0
            ckpt = {
                "decoder": decoder.state_dict(),
                "latent_codes": latent_codes.state_dict(),
                "latent_dim": latent_codes.embedding_dim,
                "n_shapes": latent_codes.num_embeddings,
                "timestamp": TIMESTAMP,
            }
            latent_dim_s1 = config.autoencoder.latent_dim
            unsuffixed_s1 = MODEL_FOLDER / f"stage1_best_latentdim{latent_dim_s1}.pth"
            archived_s1 = MODEL_FOLDER / f"stage1_best_latentdim{latent_dim_s1}_{TIMESTAMP}.pth"
            torch.save(ckpt, unsuffixed_s1)
            torch.save(ckpt, archived_s1)
            print(f"  -> Saved {unsuffixed_s1.name} + {archived_s1.name}  (train {train_loss:.5f})")
        else:
            patience_counter += 1
        # A reconstruction this close is already better than the later stages can exploit, so
        # the stage stops rather than spending epochs refining a field nothing downstream reads.
        if train_recon <= recon_threshold:
            print(
                f"  Stage 1 stopped at epoch {epoch + 1}: "
                f"train_recon {train_recon:.5f} <= threshold {recon_threshold:.5f}"
            )
            break

        if patience_counter >= patience:
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
) -> None:
    """Train the box encoder alone to reproduce the codes stage 1 fitted.

    A plain regression: the encoder sees the three half-extents of a shape and must return the
    code stage 1 assigned to it. The auxiliary decoder is frozen and never enters the backward
    pass, so nothing here can move the target the encoder is aiming at. This is the stage that
    turns a lookup table of codes into a function of the geometry, which is what the surrogate
    and the repair need.

    Args:
        encoder: The box encoder, trained in place.
        z_targets: The frozen stage-1 codes, shape ``(n_shapes, latent_dim)``, indexed by the
            global shape index.
        epochs: Maximum epochs, ``config.training.stage2_epochs``.
        lr: Learning rate, ``config.training.stage2_lr``.
        patience: Epochs without improvement before stopping,
            ``config.training.stage2_patience``.
        run: Weights and Biases run the per-epoch metrics are logged to.
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
        for idx, _surf, _q, _s, half_extents in tqdm(
            train_loader, desc=f"S2 E{epoch + 1}", leave=False
        ):
            idx = idx.to(device)
            half_extents = half_extents.to(device)

            z_pred = encoder(half_extents)
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
            for idx, _surf, _q, _s, half_extents in val_loader:
                idx = idx.to(device)
                half_extents = half_extents.to(device)
                z_pred = encoder(half_extents)
                z_target = z_targets[idx].detach()
                loss = F.mse_loss(z_pred, z_target)
                v_loss += loss.item()
                val_z_preds.append(z_pred.cpu())

        n, nv = len(train_loader), len(val_loader)
        val_loss = v_loss / nv

        # Measured on what the encoder predicts, not on the stage-1 table: a low distillation
        # loss on average is still compatible with the encoder having smoothed the codes
        # towards each other.
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
                "encoder_type": "box",
                "timestamp": TIMESTAMP,
            }
            latent_dim_s2 = config.autoencoder.latent_dim
            unsuffixed_s2 = MODEL_FOLDER / f"stage2_best_encoder_latentdim{latent_dim_s2}.pth"
            archived_s2 = MODEL_FOLDER / f"stage2_best_encoder_latentdim{latent_dim_s2}_{TIMESTAMP}.pth"
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
) -> None:
    """Fine-tune the box encoder and the auxiliary decoder together, at a small learning rate.

    The objective is the reconstruction loss of stage 1 again, with the encoder supplying the
    code in place of the embedding table. Training this way from scratch is what collapses; it
    is safe here because the encoder already produces distinct codes and the learning rate is
    small enough that the two stay near where stages 1 and 2 left them.

    Both networks are saved to ``best_encoder_decoder_latentdim{N}.pth`` whenever the
    validation loss improves, which is the file the rest of Part II reads the encoder from.

    Args:
        encoder: The box encoder, trained in place.
        decoder: The auxiliary signed-distance decoder, unfrozen here and trained in place.
        epochs: Maximum epochs, ``config.training.stage3_epochs``. Zero skips the stage.
        lr: Learning rate for both networks, ``config.training.stage3_lr``.
        lambda_reg: Weight of the penalty on the code norms.
        delta: Half-width of the band the reconstruction loss is computed in.
        run: Weights and Biases run the per-epoch metrics are logged to.
    """
    # Stage 2 froze it; this is what makes the stage joint rather than another distillation.
    for p in decoder.parameters():
        p.requires_grad_(True)

    optimizer = torch.optim.Adam(
        [
            {"params": encoder.parameters(), "lr": lr},
            {"params": decoder.parameters(), "lr": lr},
        ]
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-7
    )

    best_val_loss = float("inf")

    for epoch in range(epochs):
        encoder.train()
        decoder.train()
        t_total = t_recon = t_reg = 0.0
        val_z_preds: list[torch.Tensor] = []

        for _idx, _surf, query_pts, sdf_gt, half_extents in tqdm(
            train_loader, desc=f"S3 E{epoch + 1}", leave=False
        ):
            half_extents = half_extents.to(device)
            query_pts = query_pts.to(device)
            sdf_gt = sdf_gt.to(device)

            z = encoder(half_extents)
            sdf_pred = decode_batch(decoder, z, query_pts)
            loss, recon, reg = sdf_reconstruction_loss(
                sdf_pred, sdf_gt, z, lambda_reg, delta
            )

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
            for _idx, _surf, query_pts, sdf_gt, half_extents in val_loader:
                half_extents = half_extents.to(device)
                query_pts = query_pts.to(device)
                sdf_gt = sdf_gt.to(device)
                z = encoder(half_extents)
                sdf_pred = decode_batch(decoder, z, query_pts)
                loss, _r, _g = sdf_reconstruction_loss(
                    sdf_pred, sdf_gt, z, lambda_reg, delta
                )
                v_total += loss.item()
                val_z_preds.append(z.cpu())

        n, nv = len(train_loader), len(val_loader)
        val_loss = v_total / nv
        div = z_diversity_metrics(torch.cat(val_z_preds, dim=0))
        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"S3 E{epoch + 1}/{epochs} | "
            f"train {t_total/n:.5f} (recon {t_recon/n:.5f}) | val {val_loss:.5f} | "
            f"z_norm {div['z_norm_mean']:.3f}±{div['z_norm_std']:.3f} | "
            f"cos mean={div['cos_sim_mean']:.3f} | LR {current_lr:.2e}"
        )

        run.log(
            {
                "stage3/train_total": t_total / n,
                "stage3/train_recon": t_recon / n,
                "stage3/val_total": val_loss,
                "stage3/z_norm_mean": div["z_norm_mean"],
                "stage3/z_norm_std": div["z_norm_std"],
                "stage3/cos_sim_mean": div["cos_sim_mean"],
                "stage3/lr": current_lr,
                "stage3/epoch": epoch,
            }
        )

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt = {
                "encoder": encoder.state_dict(),
                "decoder": decoder.state_dict(),
                "latent_dim": encoder.latent_dim,
                "encoder_type": "box",
                "timestamp": TIMESTAMP,
            }
            latent_dim = config.autoencoder.latent_dim
            unsuffixed = MODEL_FOLDER / f"best_encoder_decoder_latentdim{latent_dim}.pth"
            archived = MODEL_FOLDER / f"best_encoder_decoder_latentdim{latent_dim}_{TIMESTAMP}.pth"
            torch.save(ckpt, unsuffixed)
            torch.save(ckpt, archived)
            print(
                f"  -> Saved {unsuffixed.name} + {archived.name}  (val {val_loss:.5f})"
            )


def main() -> None:
    """Generate the shapes, run the three stages in order, and report the final code diversity."""
    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

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
    # Not in the configuration: this is a stopping rule for stage 1, not a quantity anything
    # downstream depends on.
    stage1_recon_threshold: float = 0.005
    stage2_epochs: int = config.training.stage2_epochs
    stage2_lr: float = config.training.stage2_lr
    stage2_patience: int = config.training.stage2_patience
    stage3_epochs: int = config.training.stage3_epochs
    stage3_lr: float = config.training.stage3_lr

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
            "stage1_epochs": stage1_epochs,
            "stage1_lr": stage1_lr,
            "stage1_recon_threshold": stage1_recon_threshold,
            "stage2_epochs": stage2_epochs,
            "stage2_lr": stage2_lr,
            "stage3_epochs": stage3_epochs,
            "stage3_lr": stage3_lr,
        },
    )

    print(f"\n{'='*60}")
    print(
        f"Generating {n_shapes} shapes ({n_surface} surface + {n_query} query pts)..."
    )
    print(f"{'='*60}")
    t0 = time.time()
    base_dataset = AutoEncoderDataset(
        n_shapes=n_shapes, n_surface=n_surface, n_query=n_query
    )
    indexed_dataset = IndexedDataset(base_dataset)
    print(f"Data generation: {time.time() - t0:.1f}s")

    # Stage 1 sees every shape. It fits one code per shape rather than a mapping, so holding
    # shapes back would only leave them without a code for stage 2 to distil.
    stage1_loader = torch.utils.data.DataLoader(
        indexed_dataset, batch_size=batch_size, shuffle=True
    )
    print(f"Stage 1 loader: {len(indexed_dataset)} shapes (no val/test split)")

    print(f"\n{'='*60}")
    print(f"STAGE 1 — Auto-Decoder  (max {stage1_epochs} epochs)")
    print(f"{'='*60}")

    decoder = SDFDecoder(latent_dim=latent_dim).to(device)
    latent_codes = nn.Embedding(n_shapes, latent_dim).to(device)
    # Small but non-zero: the codes must start apart, or the decoder receives the same input
    # for every shape on the first steps and learns to ignore it.
    nn.init.normal_(latent_codes.weight, mean=0.0, std=0.1)

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
    )

    # The stage may have ended on a worse epoch than its best, so the best checkpoint is read
    # back before the codes are handed to stage 2.
    s1_ckpt = torch.load(
        MODEL_FOLDER / f"stage1_best_latentdim{latent_dim}.pth", map_location=device
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

    # Seeded separately from the global seed so that the split is the same whatever stage 1 did
    # to the generator state. Stages 2 and 3 share it, so the encoder is never validated on a
    # shape it was fine-tuned on.
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

    encoder = BoxEncoder(latent_dim=latent_dim).to(device)
    enc_params = sum(p.numel() for p in encoder.parameters())
    print(f"BoxEncoder params: {enc_params:,}")

    # Taken verbatim, with no rescaling or renormalisation. Any transform applied here would
    # put the encoder in a different space from the one the decoder was trained against, and
    # stage 3 would then have to undo it.
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
    )

    s2_ckpt = torch.load(
        MODEL_FOLDER / f"stage2_best_encoder_latentdim{latent_dim}.pth",
        map_location=device,
    )
    encoder.load_state_dict(s2_ckpt["encoder"])

    if stage3_epochs > 0:
        print(f"\n{'='*60}")
        print(
            f"STAGE 3 — Joint Fine-Tune  (max {stage3_epochs} epochs, LR={stage3_lr})"
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
        )
        # Stage 3 writes the final checkpoint itself, on every validation improvement.
    else:
        # Without stage 3 the final checkpoint has to be assembled here, from the stage-2
        # encoder and the stage-1 decoder, so that downstream code finds the same filename
        # either way.
        ckpt = {
            "encoder": encoder.state_dict(),
            "decoder": decoder.state_dict(),
            "latent_dim": latent_dim,
            "encoder_type": "box",
            "timestamp": TIMESTAMP,
        }
        unsuffixed = MODEL_FOLDER / f"best_encoder_decoder_latentdim{latent_dim}.pth"
        archived = MODEL_FOLDER / f"best_encoder_decoder_latentdim{latent_dim}_{TIMESTAMP}.pth"
        torch.save(ckpt, unsuffixed)
        torch.save(ckpt, archived)
        print(f"\nFinal checkpoint saved: {unsuffixed} + {archived.name}")

    # The closing check is on the held-out split and on the encoder alone, since that is the
    # network the rest of Part II uses and the one whose codes must stay distinct on shapes it
    # was not trained on.
    encoder.eval()
    decoder.eval()
    test_z_preds: list[torch.Tensor] = []
    with torch.no_grad():
        for _idx, _surf, _q, _s, half_extents in test_loader:
            half_extents = half_extents.to(device)
            test_z_preds.append(encoder(half_extents).cpu())

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
