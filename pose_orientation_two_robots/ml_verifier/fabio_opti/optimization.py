"""Fabio's GNN-driven repair optimization for woodworking assembly.

Called from model_interface.py::refine_full_yaml_model_a(input_csv) -> output_csv.

Latent-space branch: GNN consumes the frozen-BoxEncoder latent z (8d)
instead of raw sizes, so node features are [z_0..z_7, type_0, type_1]
(node_dim=10). The Adam repair optimises (z_active, pos_active) and
maps z back to sizes for hard caps, snap geometry and CSV output via
a frozen BoxDecoder.

Pipeline (single CSV string in, single CSV string out):
  1. Symmetric-shrink 3D-penetration resolution (analytical, all pairs).
  2. Snap-to-contact for connected pairs (analytical, BFS from largest).
  3. GNN-driven bottom-up layered Adam repair (per-pair, parent frozen).
     - Hard caps (Claire-style PGD): pos box, tangential-grow / contact-shrink
       size box — applied via decode -> clamp -> re-encode roundtrip.
     - Drift penalty lives in size-space (decode(z) - decode(z_orig)).
     - Periodic analytical snap to contact face + floor correction every K steps.
     - Final analytical snap after the loop, regardless of early-stop.
  4. Screw placement in each pair's repaired tangential overlap rectangle
     with global collision avoidance.

Model files expected at:
  <this_file's_dir>/../fabio_model/gnn*.pth                              # GNN
  <this_file's_dir>/../fabio_model/best_encoder_decoder_latentdim8.pth  # BoxEncoder
  <this_file's_dir>/../fabio_model/best_box_decoder_latentdim8.pth      # BoxDecoder
"""

from __future__ import annotations

import io
import math
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch, Data
from torch_geometric.nn import GATv2Conv

# =========================================================
# PATHS
# =========================================================
_HERE = Path(__file__).resolve().parent
_MODEL_DIR = _HERE.parent / "fabio_model"

# =========================================================
# HARDCODED CONFIG (10mm-overlap branch)
# =========================================================
# Feasibility thresholds (raw metres)
THRESH_OVERLAP_M: float = 0.010
THRESH_THICKNESS_M: float = 0.020

# Adam repair
NUM_STEPS: int = 1000
LR: float = 0.003
LAMBDA_SIZE: float = 5.0
LAMBDA_POS: float = 1.0
HINGE_SCALE: float = 1e6  # raw-m^2 -> mm^2 magnitude
HINGE_W_OV: float = 1.0
HINGE_W_TH: float = 1.0
SIZE_MIN_M: float = 0.008
SIZE_MAX_M: float = 0.250
EARLY_STOP_INTERVAL: int = 20
EARLY_STOP_P_BINARY: float = (
    0.9  # early-stop when p_binary > this AND overlap/thickness in range
)

# Hard caps (Claire-style projected gradient descent)
POS_DRIFT_BOX_M: float = 0.020
SIZE_ALPHA: float = 0.3  # tangential growth budget (+30%)
CONTACT_SHRINK_ALPHA: float = 0.9  # contact axis may shrink to 10% of orig
ASYMMETRIC_TANGENTIAL: bool = True

# Periodic + final analytical snap
ANALYTICAL_SNAP_INTERVAL: int = 50
ANALYTICAL_FLOOR_CORRECTION: bool = True
RESET_ADAM_STATE_ON_SNAP: bool = True

# Per-sample pos-freeze when GNN already predicts overlap >= threshold
FREEZE_POS_IF_OVERLAP_OK: bool = True

POST_SHRINK_M: float = 0.001  # contact-axis safety margin after Adam

# Pipeline
SCALE_FACTOR: float = 1.0  # raw metres
FLOOR_TOLERANCE_M: float = 1e-3
OVERLAP_LOG_EPS: float = 1e-5  # mirror dataset preparation

# Screws
SCREW_LENGTH_M: float = 0.022
SCREW_SAFETY_M: float = 0.005
SCREW_SEPARATION_M: float = 0.010


# =========================================================
# CODEC (BoxEncoder + BoxDecoder) — single-file definitions
# =========================================================
# Must match src/enc_box.py and src/dec_box.py architectures exactly.
LATENT_DIM: int = 8
_BOX_ENCODER_CKPT = "best_encoder_decoder_latentdim8.pth"
_BOX_DECODER_CKPT = "best_box_decoder_latentdim8.pth"


class _BoxEncoder(nn.Module):
    """MLP encoder mapping box half-extents (B, 3) -> latent z (B, latent_dim).

    Mirrors src/enc_box.py: hidden_dim=128, num_layers=4, ReLU activations,
    final Linear with no activation.
    """

    def __init__(self, latent_dim: int = LATENT_DIM, hidden_dim: int = 128, num_layers: int = 4):
        super().__init__()
        self.latent_dim = latent_dim
        layers: list[nn.Module] = []
        in_dim = 3
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, latent_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _BoxDecoder(nn.Module):
    """MLP decoder mapping z (B, latent_dim) -> half_extents (B, 3) via exp.

    Mirrors src/dec_box.py: hidden_dim=256, num_layers=6, GELU activations,
    final Linear feeding through exp() so reconstruction lives in metres
    with relative-error balance across the log-uniform training range.
    """

    def __init__(self, latent_dim: int = LATENT_DIM, hidden_dim: int = 256, num_layers: int = 6):
        super().__init__()
        self.latent_dim = latent_dim
        layers: list[nn.Module] = []
        in_dim = latent_dim
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.GELU())
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, 3))
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return torch.exp(self.net(z))


def _encode_size(encoder: _BoxEncoder, size: torch.Tensor) -> torch.Tensor:
    """Map full size (..., 3) -> latent z (..., latent_dim). Encoder expects half_extents."""
    return encoder(size / 2.0)


def _decode_z(decoder: _BoxDecoder, z: torch.Tensor) -> torch.Tensor:
    """Map latent z (..., latent_dim) -> full size (..., 3) in metres."""
    return decoder(z) * 2.0


# =========================================================
# MODEL ARCHITECTURE (must match training: late-fork GNN)
# =========================================================
class _GNN(nn.Module):
    """Late-fork GNN with z + node-type as 10d node features, 6d edge features.

    Inputs (latent-space branch):
        node = [z_0..z_7, type_0, type_1]                            (10 dims)
        edge = [Δx, Δy, Δz, dist_xy, dist_xz, dist_yz]                (6 dims)

    Outputs (3-tuple): (reg_out, binary_logit, mode_logits)
        reg_out[:, 0] = overlap_log_std, reg_out[:, 1] = thickness_std
    """

    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        hidden_dim: int,
        heads: int,
        head_hidden: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.dropout_p = dropout

        self.gat1 = GATv2Conv(
            node_dim,
            hidden_dim // heads,
            heads=heads,
            edge_dim=edge_dim,
            dropout=dropout,
        )
        self.gat2 = GATv2Conv(
            hidden_dim,
            hidden_dim // heads,
            heads=heads,
            edge_dim=edge_dim,
            dropout=dropout,
        )
        self.gat3_reg = GATv2Conv(
            hidden_dim,
            hidden_dim // heads,
            heads=heads,
            edge_dim=edge_dim,
            dropout=dropout,
        )
        self.gat3_cls = GATv2Conv(
            hidden_dim,
            hidden_dim // heads,
            heads=heads,
            edge_dim=edge_dim,
            dropout=dropout,
        )

        self.mlp_reg = nn.Sequential(
            nn.Linear(hidden_dim, head_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, head_hidden // 2),
            nn.GELU(),
            nn.Linear(head_hidden // 2, 2),
        )
        self.mlp_binary = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )
        self.mlp_modes = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 2),
        )

    def forward(self, x, edge_index, edge_attr):
        x = F.gelu(self.gat1(x, edge_index, edge_attr))
        x = F.dropout(x, p=self.dropout_p, training=self.training)
        x = F.gelu(self.gat2(x, edge_index, edge_attr))
        x = F.dropout(x, p=self.dropout_p, training=self.training)
        x_reg = F.gelu(self.gat3_reg(x, edge_index, edge_attr))
        x_cls = F.gelu(self.gat3_cls(x, edge_index, edge_attr))
        # Block1 (active) only — odd indices in batched 2-node graphs
        x_reg_b1 = x_reg[1::2]
        x_cls_b1 = x_cls[1::2]
        return (
            self.mlp_reg(x_reg_b1),
            self.mlp_binary(x_cls_b1),
            self.mlp_modes(x_cls_b1),
        )


# =========================================================
# LAZY MODEL LOADER
# =========================================================
_loaded: dict = {}


def _load_model() -> dict:
    """Load the GNN + BoxEncoder + BoxDecoder from fabio_model/ (cached)."""
    if _loaded:
        return _loaded

    # --- GNN: pick the most recent gnn*.pth (not encoder/decoder ckpts) ---
    candidates = sorted(
        [
            p
            for p in _MODEL_DIR.glob("gnn*.pth")
            if "encoder" not in p.name and "decoder" not in p.name
        ],
        key=lambda p: p.stat().st_mtime,
    )
    if not candidates:
        raise FileNotFoundError(
            f"No gnn*.pth in {_MODEL_DIR}. Drop the trained GNN checkpoint there."
        )
    ckpt_path = candidates[-1]
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = dict(ckpt.get("model", ckpt))

    target_mean = state.pop("target_mean")
    target_std = state.pop("target_std")

    # Infer architecture from weight shapes
    node_dim = int(state["gat1.lin_l.weight"].shape[1])
    edge_dim = int(state["gat1.lin_edge.weight"].shape[1])
    heads = int(state["gat1.att"].shape[1])
    hidden_per_head = int(state["gat1.att"].shape[2])
    hidden_dim = heads * hidden_per_head
    head_hidden = int(state["mlp_reg.0.weight"].shape[0])

    expected_node_dim = LATENT_DIM + 2  # 8 z + 2 type
    if node_dim != expected_node_dim:
        raise RuntimeError(
            f"GNN checkpoint {ckpt_path.name} has node_dim={node_dim}, "
            f"but the latent-space pipeline expects node_dim={expected_node_dim} "
            f"(latent_dim={LATENT_DIM} + 2 type). Checkpoint mismatch — "
            f"is this a parameter-branch GNN?"
        )

    model = _GNN(
        node_dim=node_dim,
        edge_dim=edge_dim,
        hidden_dim=hidden_dim,
        heads=heads,
        head_hidden=head_hidden,
        dropout=0.0,
    )
    model.register_buffer("target_mean", target_mean.clone())
    model.register_buffer("target_std", target_std.clone())
    model.load_state_dict(state, strict=False)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    # --- BoxEncoder ---
    enc_path = _MODEL_DIR / _BOX_ENCODER_CKPT
    if not enc_path.exists():
        raise FileNotFoundError(
            f"BoxEncoder checkpoint not found: {enc_path}. "
            f"Copy from project_root/encoder_decoder_model/."
        )
    enc_ckpt = torch.load(enc_path, map_location="cpu", weights_only=False)
    encoder = _BoxEncoder(latent_dim=LATENT_DIM)
    encoder.load_state_dict(enc_ckpt["encoder"])
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False

    # --- BoxDecoder ---
    dec_path = _MODEL_DIR / _BOX_DECODER_CKPT
    if not dec_path.exists():
        raise FileNotFoundError(
            f"BoxDecoder checkpoint not found: {dec_path}. "
            f"Train it via tools/train_box_decoder.py and copy from "
            f"project_root/encoder_decoder_model/."
        )
    dec_ckpt = torch.load(dec_path, map_location="cpu", weights_only=False)
    decoder = _BoxDecoder(latent_dim=LATENT_DIM)
    decoder.load_state_dict(dec_ckpt["decoder"])
    decoder.eval()
    for p in decoder.parameters():
        p.requires_grad = False

    _loaded.update(
        {
            "model": model,
            "encoder": encoder,
            "decoder": decoder,
            "ckpt_name": ckpt_path.name,
            "node_dim": node_dim,
            "edge_dim": edge_dim,
            "hidden_dim": hidden_dim,
        }
    )
    return _loaded


# =========================================================
# ANALYTICAL HELPERS (face inference + contact snap)
# =========================================================
def _infer_face(
    pos_p: np.ndarray,
    he_p: np.ndarray,
    pos_c: np.ndarray,
    he_c: np.ndarray,
) -> int:
    """Face code 0..5 from the axis with largest |Δ|/(he_p+he_c)."""
    delta = pos_c - pos_p
    ratio = np.abs(delta) / (he_p + he_c + 1e-12)
    ax = int(np.argmax(ratio))
    return 2 * ax + (0 if delta[ax] >= 0 else 1)


def _snap_to_contact_face(
    pos_p: np.ndarray,
    he_p: np.ndarray,
    pos_c: np.ndarray,
    he_c: np.ndarray,
    contact_axis: int,
) -> np.ndarray:
    """Translate curr along the contact axis so its face touches prev exactly.
    Returns the snapped curr-centre (other axes unchanged).
    """
    delta = pos_c - pos_p
    sign = 1.0 if delta[contact_axis] >= 0 else -1.0
    target = pos_p[contact_axis] + sign * (he_p[contact_axis] + he_c[contact_axis])
    out = pos_c.copy()
    out[contact_axis] = target
    return out


# =========================================================
# GRAPH CONSTRUCTION (batched 2-node graphs)
# =========================================================
def _edge_features(
    pos_anchor: torch.Tensor,
    pos_active: torch.Tensor,
) -> torch.Tensor:
    """(2, 6) edge tensor: fwd + reverse with [Δxyz, dist_xy, dist_xz, dist_yz]."""
    delta = pos_active - pos_anchor
    dx, dy, dz = delta[0], delta[1], delta[2]
    eps = 1e-8
    dist_xy = torch.sqrt(dx**2 + dy**2 + eps)
    dist_xz = torch.sqrt(dx**2 + dz**2 + eps)
    dist_yz = torch.sqrt(dy**2 + dz**2 + eps)
    fwd = torch.stack([dx, dy, dz, dist_xy, dist_xz, dist_yz])
    rev = torch.stack([-dx, -dy, -dz, dist_xy, dist_xz, dist_yz])
    return torch.stack([fwd, rev])


def _build_batch(
    z_anchor_all: torch.Tensor,   # (N, latent_dim)
    z_active_all: torch.Tensor,   # (N, latent_dim)
    pos_anchor_all: torch.Tensor, # (N, 3) scaled
    pos_active_all: torch.Tensor, # (N, 3) scaled
) -> Batch:
    """Build a PyG batch of 2-node graphs with z + type as node features."""
    device = z_anchor_all.device
    N = z_anchor_all.shape[0]
    type_0 = torch.tensor([1.0, 0.0], device=device)
    type_1 = torch.tensor([0.0, 1.0], device=device)
    edge_index_template = torch.tensor(
        [[0, 1], [1, 0]], dtype=torch.long, device=device
    )
    graphs = []
    for i in range(N):
        x_i = torch.stack(
            [
                torch.cat([z_anchor_all[i], type_0]),
                torch.cat([z_active_all[i], type_1]),
            ]
        )
        ea_i = _edge_features(pos_anchor_all[i], pos_active_all[i])
        graphs.append(Data(x=x_i, edge_index=edge_index_template, edge_attr=ea_i))
    return Batch.from_data_list(graphs)


def _destandardize(
    reg_pred_std: torch.Tensor,
    target_mean: torch.Tensor,
    target_std: torch.Tensor,
) -> torch.Tensor:
    """De-standardise GNN regression outputs to raw metres.
    Column 0 (overlap) is log-standardised: invert via exp(...) - eps.
    """
    out = reg_pred_std * target_std + target_mean
    if target_mean[0].item() < 0.0:
        out = out.clone()
        out[..., 0] = torch.exp(out[..., 0]) - OVERLAP_LOG_EPS
    return out


# =========================================================
# ANALYTICAL SNAP (periodic + final, in scaled domain)
# =========================================================
def _apply_analytical_snap(
    pos_anchor_all: torch.Tensor,
    size_anchor_all: torch.Tensor,
    pos_active: nn.Parameter,
    size_active_decoded: torch.Tensor,  # (N, 3) scaled — read-only here
    repaired_mask: torch.Tensor,
    contact_axes: torch.Tensor,
    scale_factor: float,
    floor_correction: bool,
) -> None:
    """Snap each not-yet-repaired pair to its contact face (resolves both
    penetration and gap on the contact axis). Optionally lift the block
    above the floor (z_bottom >= 0). Modifies pos_active.data in place.
    """
    sf = float(scale_factor)
    pos_p_m = (pos_anchor_all / sf).detach().cpu().numpy()
    he_p_m = (size_anchor_all / (2.0 * sf)).detach().cpu().numpy()
    pos_c_m = (pos_active.data / sf).detach().cpu().numpy()
    he_c_m = (size_active_decoded / (2.0 * sf)).detach().cpu().numpy()
    mask_np = repaired_mask.detach().cpu().numpy()
    axes_np = contact_axes.detach().cpu().numpy()

    for i in range(pos_p_m.shape[0]):
        if mask_np[i]:
            continue
        pos_c_m[i] = _snap_to_contact_face(
            pos_p_m[i],
            he_p_m[i],
            pos_c_m[i],
            he_c_m[i],
            contact_axis=int(axes_np[i]),
        )

    if floor_correction:
        bottom = pos_c_m[:, 2] - he_c_m[:, 2]
        below = bottom < 0.0
        if below.any():
            pos_c_m[below, 2] = he_c_m[below, 2]

    pos_active.data = torch.from_numpy(pos_c_m * sf).to(
        device=pos_active.device,
        dtype=pos_active.dtype,
    )


# =========================================================
# BATCHED ADAM REPAIR (the core gradient-based step)
# =========================================================
def _project_z_via_size_clamp(
    z_active: nn.Parameter,
    encoder: _BoxEncoder,
    decoder: _BoxDecoder,
    size_lo: torch.Tensor,           # (N, 3) scaled
    size_hi: torch.Tensor,           # (N, 3) scaled
    size_min_abs: float,
    size_max_abs: float,
) -> None:
    """Hard-cap projection in size-space: decode -> clamp -> re-encode.

    Operates in no_grad; modifies z_active.data in place. Only re-encodes
    samples that actually moved, to avoid roundtrip drift on already-legal z.
    """
    with torch.no_grad():
        size_dec = _decode_z(decoder, z_active)
        size_clamped = size_dec.clamp(min=size_min_abs, max=size_max_abs)
        size_clamped = torch.minimum(
            torch.maximum(size_clamped, size_lo), size_hi
        )
        moved = (size_clamped - size_dec).abs().sum(dim=1) > 1e-9
        if moved.any():
            z_new = _encode_size(encoder, size_clamped[moved])
            z_active.data[moved] = z_new


def _adam_repair_batched(
    size_anchor_list: list[torch.Tensor],
    size_active_list: list[torch.Tensor],
    pos_anchor_list: list[torch.Tensor],
    pos_active_list: list[torch.Tensor],
    gnn_model: _GNN,
    encoder: _BoxEncoder,
    decoder: _BoxDecoder,
    scale_factor: float,
) -> dict:
    """Latent-space Adam batched repair on (z_active, pos_active).

    Inputs / outputs are still sizes (raw metres) so callers don't change.
    Internally:
        - Encode initial sizes -> z once (frozen).
        - Adam optimises z + pos.
        - GNN forward uses z as node feature.
        - Drift penalty in size-space via decode(z) - decode(z_orig).
        - Hard caps applied as decode -> clamp -> re-encode roundtrip.
        - Periodic + final analytical snap operate on decoded sizes; snap
          only modifies pos so z (and hence the snapped sample's size)
          remains consistent.

    Returns dict with: size_active_out (N,3) decoded, pos_active_out (N,3),
    reg_out_m (N,2), p_binary (N,), success_mask (N,).
    """
    device = size_anchor_list[0].device
    N = len(size_anchor_list)
    size_min_scaled = float(SIZE_MIN_M * scale_factor)
    size_max_scaled = float(SIZE_MAX_M * scale_factor)

    size_anchor_all = torch.stack([s.clone().detach() for s in size_anchor_list])
    pos_anchor_all = torch.stack([p.clone().detach() for p in pos_anchor_list])
    size_active_start = torch.stack([s.clone().detach() for s in size_active_list])
    pos_active_start = torch.stack([p.clone().detach() for p in pos_active_list])
    size_active_orig = size_active_start.clone()  # for size-space drift penalty
    pos_active_orig = pos_active_start.clone()

    # Encode initial sizes to latent (frozen, no_grad).
    with torch.no_grad():
        z_anchor_all = _encode_size(encoder, size_anchor_all)         # (N, D)
        z_active_start = _encode_size(encoder, size_active_start)     # (N, D)

    z_active = nn.Parameter(z_active_start.clone())
    pos_active = nn.Parameter(pos_active_start.clone())

    optimizer = torch.optim.Adam([z_active, pos_active], lr=LR)
    gnn_model.eval()
    target_mean = gnn_model.target_mean
    target_std = gnn_model.target_std

    repaired_mask = torch.zeros(N, dtype=torch.bool, device=device)
    pos_box_scaled = float(POS_DRIFT_BOX_M) * scale_factor

    # Infer contact axis per sample analytically (used by hard cap + snap)
    contact_axes = torch.zeros(N, dtype=torch.long, device=device)
    sf = float(scale_factor)
    for i in range(N):
        pos_p_m = pos_anchor_all[i].cpu().numpy() / sf
        he_p_m = (size_anchor_all[i] / 2.0).cpu().numpy() / sf
        pos_c_m = pos_active_orig[i].cpu().numpy() / sf
        he_c_m = (size_active_orig[i] / 2.0).cpu().numpy() / sf
        face = _infer_face(pos_p_m, he_p_m, pos_c_m, he_c_m)
        contact_axes[i] = face // 2

    # Size box: upper bound symmetric; lower bound asymmetric.
    size_box_hi = size_active_orig * (1.0 + SIZE_ALPHA)
    if ASYMMETRIC_TANGENTIAL:
        contact_mask = F.one_hot(contact_axes, num_classes=3).bool()
        contact_lo = size_active_orig * (1.0 - CONTACT_SHRINK_ALPHA)
        size_box_lo = torch.where(contact_mask, contact_lo, size_active_orig)
    else:
        size_box_lo = size_active_orig * (1.0 - SIZE_ALPHA)
    size_lo_final = torch.clamp(size_box_lo, min=size_min_scaled)
    size_hi_final = torch.clamp(size_box_hi, max=size_max_scaled)

    # Per-sample pos-freeze: probe GNN; samples whose predicted overlap
    # already passes the threshold get pos.grad zeroed for the entire run.
    if FREEZE_POS_IF_OVERLAP_OK:
        with torch.no_grad():
            batch_probe = _build_batch(
                z_anchor_all, z_active, pos_anchor_all, pos_active,
            )
            reg_probe_std, _, _ = gnn_model(
                batch_probe.x, batch_probe.edge_index, batch_probe.edge_attr,
            )
            reg_probe = _destandardize(reg_probe_std, target_mean, target_std)
            pos_freeze_mask = (reg_probe[:, 0] >= THRESH_OVERLAP_M).to(device)
    else:
        pos_freeze_mask = None

    for step in range(NUM_STEPS):
        progress = step / max(NUM_STEPS - 1, 1)
        relax = 1.0 + (1.0 - progress)
        thresh_ov = THRESH_OVERLAP_M / relax
        thresh_th = THRESH_THICKNESS_M * relax

        batch = _build_batch(z_anchor_all, z_active, pos_anchor_all, pos_active)
        reg_std, bin_logit, _ = gnn_model(batch.x, batch.edge_index, batch.edge_attr)
        reg_m = _destandardize(reg_std, target_mean, target_std)
        ov_m = reg_m[:, 0]
        th_m = reg_m[:, 1]

        active_mask = (~repaired_mask).float()
        n_active = active_mask.sum().clamp(min=1.0)

        binary_loss = (
            F.relu(2.0 - bin_logit.squeeze(1)) ** 2 * active_mask
        ).sum() / n_active
        hinge_ov = F.relu(thresh_ov - ov_m) ** 2
        hinge_th = F.relu(th_m - thresh_th) ** 2
        reg_loss = (
            HINGE_SCALE
            * ((HINGE_W_OV * hinge_ov + HINGE_W_TH * hinge_th) * active_mask).sum()
            / n_active
        )
        # Drift in SIZE-space (decoded), mirroring the parameter branch.
        size_active_decoded = _decode_z(decoder, z_active)
        size_drift = (
            (size_active_decoded - size_active_orig).pow(2).sum(dim=1) * active_mask
        ).sum() / n_active
        pos_drift = (
            (pos_active - pos_active_orig).pow(2).sum(dim=1) * active_mask
        ).sum() / n_active
        loss = (
            binary_loss + reg_loss + LAMBDA_SIZE * size_drift + LAMBDA_POS * pos_drift
        )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_([z_active, pos_active], max_norm=1.0)
        if pos_freeze_mask is not None and pos_active.grad is not None:
            pos_active.grad[pos_freeze_mask] = 0
        optimizer.step()

        # Post-step projection: size-space hard caps via decode -> clamp -> re-encode.
        with torch.no_grad():
            _project_z_via_size_clamp(
                z_active, encoder, decoder,
                size_lo_final, size_hi_final, size_min_scaled, size_max_scaled,
            )
            # Pos drift box
            pos_diff = pos_active.data - pos_active_orig
            pos_active.data = pos_active_orig + pos_diff.clamp(
                -pos_box_scaled, pos_box_scaled
            )

            # Periodic analytical snap
            if (
                ANALYTICAL_SNAP_INTERVAL > 0
                and (step + 1) % ANALYTICAL_SNAP_INTERVAL == 0
                and not bool(repaired_mask.all())
            ):
                size_active_now = _decode_z(decoder, z_active)  # no_grad
                _apply_analytical_snap(
                    pos_anchor_all,
                    size_anchor_all,
                    pos_active,
                    size_active_now,
                    repaired_mask,
                    contact_axes,
                    scale_factor,
                    floor_correction=ANALYTICAL_FLOOR_CORRECTION,
                )
                pos_diff = pos_active.data - pos_active_orig
                pos_active.data = pos_active_orig + pos_diff.clamp(
                    -pos_box_scaled, pos_box_scaled
                )
                if RESET_ADAM_STATE_ON_SNAP:
                    optimizer.state.clear()

        # Early stop
        if (step + 1) % EARLY_STOP_INTERVAL == 0:
            with torch.no_grad():
                batch_es = _build_batch(
                    z_anchor_all, z_active, pos_anchor_all, pos_active
                )
                reg_es_std, bin_es, _ = gnn_model(
                    batch_es.x, batch_es.edge_index, batch_es.edge_attr
                )
                reg_es = _destandardize(reg_es_std, target_mean, target_std)
                p_es = torch.sigmoid(bin_es).squeeze(1).cpu().numpy()
                r_es = reg_es.cpu().numpy()
                ov_ok = r_es[:, 0] > THRESH_OVERLAP_M
                th_ok = r_es[:, 1] < THRESH_THICKNESS_M
                bn_ok = p_es > EARLY_STOP_P_BINARY
                all_ok = ov_ok & th_ok & bn_ok
                repaired_mask = torch.from_numpy(all_ok).to(device)
                if all_ok.all():
                    break

    # Final analytical snap (all pairs, no pos-box re-apply)
    if ANALYTICAL_SNAP_INTERVAL > 0:
        with torch.no_grad():
            size_active_now = _decode_z(decoder, z_active)
            _apply_analytical_snap(
                pos_anchor_all,
                size_anchor_all,
                pos_active,
                size_active_now,
                repaired_mask=torch.zeros(N, dtype=torch.bool, device=device),
                contact_axes=contact_axes,
                scale_factor=scale_factor,
                floor_correction=ANALYTICAL_FLOOR_CORRECTION,
            )

    # Final forward on snapped geometry
    with torch.no_grad():
        batch_f = _build_batch(z_anchor_all, z_active, pos_anchor_all, pos_active)
        reg_f_std, bin_f, _ = gnn_model(
            batch_f.x, batch_f.edge_index, batch_f.edge_attr
        )
        reg_f = _destandardize(reg_f_std, target_mean, target_std)
        p_binary = torch.sigmoid(bin_f).squeeze(1).cpu().numpy()
        reg_out_m = reg_f.cpu().numpy()
        size_active_out = _decode_z(decoder, z_active).detach()

    success_mask = (
        (reg_out_m[:, 0] >= THRESH_OVERLAP_M)
        & (reg_out_m[:, 1] <= THRESH_THICKNESS_M)
        & (p_binary >= 0.5)
    )
    return {
        "size_active_out": size_active_out,
        "pos_active_out": pos_active.detach(),
        "reg_out_m": reg_out_m,
        "p_binary": p_binary,
        "success_mask": success_mask,
    }


# =========================================================
# CSV PARSING
# =========================================================
def _detect_n_blocks(df: pd.DataFrame) -> int:
    n = 0
    while f"Block{n}_PosX" in df.columns:
        n += 1
    return n


def _parse_blocks(row: pd.Series, n: int) -> dict[int, dict]:
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


def _parse_connections(row: pd.Series, n: int) -> list[tuple[int, int]]:
    conns: list[tuple[int, int]] = []
    for i in range(n):
        for j in range(i + 1, n):
            col = f"Block{i}_ConnectsTo_Block{j}"
            if col in row.index and int(row[col]) == 1:
                conns.append((i, j))
    return conns


def _parse_original_screws(
    row: pd.Series,
    n_blocks: int,
    n_screws: int,
) -> dict[int, list[tuple[int, np.ndarray]]]:
    """{block_id: [(slot, np.array([x, y, z])), ...]} for every non-NaN screw."""
    out: dict[int, list[tuple[int, np.ndarray]]] = {i: [] for i in range(n_blocks)}
    for i in range(n_blocks):
        for s in range(n_screws):
            cols = [f"Block{i}_Screw{s}_{ax}" for ax in "XYZ"]
            if not all(c in row.index for c in cols):
                continue
            try:
                vals = np.array([float(row[c]) for c in cols], dtype=float)
            except (TypeError, ValueError):
                continue
            if np.isnan(vals).any():
                continue
            out[i].append((s, vals))
    return out


# =========================================================
# STAGE 1: PENETRATION SHRINK (analytical)
# =========================================================
def _aabb_overlap_xyz(b_i: dict, b_j: dict) -> np.ndarray:
    delta = np.abs(b_i["pos"] - b_j["pos"])
    return (b_i["size"] + b_j["size"]) / 2.0 - delta


def _resolve_penetrations(blocks: dict[int, dict]) -> dict[int, dict]:
    """Iteratively shrink the two blocks of each 3D-penetrating pair on
    the cheapest axis. Hard cap of 50 iterations.
    """
    new = {
        k: {"pos": v["pos"].copy(), "size": v["size"].copy()} for k, v in blocks.items()
    }
    keys = sorted(new.keys())
    for _ in range(50):
        candidates: list[tuple[int, int, np.ndarray]] = []
        for ii, i in enumerate(keys):
            for j in keys[ii + 1 :]:
                ov = _aabb_overlap_xyz(new[i], new[j])
                if (ov > 1e-9).all():
                    candidates.append((i, j, ov))
        if not candidates:
            break
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


# =========================================================
# STAGE 2: SNAP-TO-CONTACT BFS (analytical)
# =========================================================
def _apply_snaps_bfs(
    blocks: dict[int, dict],
    conns: list[tuple[int, int]],
) -> dict[int, dict]:
    """BFS from the largest-volume block; for each edge snap child to parent's face."""
    if not blocks:
        return blocks
    largest = max(blocks, key=lambda i: float(np.prod(blocks[i]["size"])))
    adj: dict[int, set[int]] = {i: set() for i in blocks}
    for i, j in conns:
        adj[i].add(j)
        adj[j].add(i)
    visited = {largest}
    queue: deque[int] = deque([largest])
    edges_order: list[tuple[int, int]] = []
    while queue:
        cur = queue.popleft()
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
        he_p = p["size"] / 2.0
        he_c = c["size"] / 2.0
        ax = _infer_face(p["pos"], he_p, c["pos"], he_c) // 2
        snapped[child]["pos"] = _snap_to_contact_face(
            p["pos"], he_p, c["pos"], he_c, contact_axis=ax
        )
    return snapped


# =========================================================
# STAGE 3: GNN-DRIVEN LAYERED REPAIR
# =========================================================
def _gnn_repair_layered(
    blocks: dict[int, dict],
    conns: list[tuple[int, int]],
    gnn_model: _GNN,
    encoder: _BoxEncoder,
    decoder: _BoxDecoder,
    device: torch.device,
) -> dict[int, dict]:
    """Bottom-up layered repair: floor anchors -> children BFS via _adam_repair_batched."""
    sf = SCALE_FACTOR
    adj: dict[int, set[int]] = {i: set() for i in blocks}
    for i, j in conns:
        adj[i].add(j)
        adj[j].add(i)

    pos_t: dict[int, torch.Tensor] = {}
    size_t: dict[int, torch.Tensor] = {}
    for i, b in blocks.items():
        pos_t[i] = torch.from_numpy((b["pos"] * sf).astype(np.float32)).to(device)
        size_t[i] = torch.from_numpy((b["size"] * sf).astype(np.float32)).to(device)

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
                # Skip if already committed OR already queued in this layer
                # by another parent (would only overwrite with cumulative drift).
                if child in committed or child in next_layer:
                    continue

                # Pre-Adam snap: re-snap child to its (already-repaired) parent
                # on the inferred contact axis. If the parent shrunk in a
                # previous layer, the child was left with a stale gap that
                # the GNN reads as "no contact" -> wrong overlap prediction
                # -> freeze doesn't fire and Adam wastes effort on pos.
                pos_p_m = pos_t[parent].cpu().numpy() / sf
                he_p_m = size_t[parent].cpu().numpy() / (2.0 * sf)
                pos_c_m = pos_t[child].cpu().numpy() / sf
                he_c_m = size_t[child].cpu().numpy() / (2.0 * sf)
                face = _infer_face(pos_p_m, he_p_m, pos_c_m, he_c_m)
                contact_ax = face // 2
                pos_c_snapped = _snap_to_contact_face(
                    pos_p_m,
                    he_p_m,
                    pos_c_m,
                    he_c_m,
                    contact_axis=contact_ax,
                )
                pos_t[child] = torch.from_numpy(
                    (pos_c_snapped * sf).astype(np.float32)
                ).to(device)

                # Initial GNN probe — skip Adam if already feasible
                # (mirrors repair_chain_single in the dashboard pipeline).
                with torch.no_grad():
                    z_parent = _encode_size(encoder, size_t[parent].unsqueeze(0))
                    z_child = _encode_size(encoder, size_t[child].unsqueeze(0))
                    batch_probe = _build_batch(
                        z_parent,
                        z_child,
                        pos_t[parent].unsqueeze(0),
                        pos_t[child].unsqueeze(0),
                    )
                    reg_std_probe, bin_probe, _ = gnn_model(
                        batch_probe.x,
                        batch_probe.edge_index,
                        batch_probe.edge_attr,
                    )
                    reg_probe = _destandardize(
                        reg_std_probe,
                        gnn_model.target_mean,
                        gnn_model.target_std,
                    )
                    p_probe = float(torch.sigmoid(bin_probe.squeeze()).item())
                    ov_probe = float(reg_probe[0, 0].item())
                    th_probe = float(reg_probe[0, 1].item())

                initially_feasible = (
                    ov_probe >= THRESH_OVERLAP_M
                    and th_probe <= THRESH_THICKNESS_M
                    and p_probe >= 0.5
                )

                if not initially_feasible:
                    out = _adam_repair_batched(
                        [size_t[parent]],
                        [size_t[child]],
                        [pos_t[parent]],
                        [pos_t[child]],
                        gnn_model,
                        encoder,
                        decoder,
                        sf,
                    )
                    size_t[child] = out["size_active_out"][0].detach()
                    if bool(out["success_mask"][0]):
                        # Post-Adam contact snap (mirrors _apply_contact_snap
                        # in repair_chain_single). Tangential pos preserved,
                        # contact axis snapped to parent face.
                        pos_c_after = out["pos_active_out"][0].cpu().numpy() / sf
                        pc_m_after = _snap_to_contact_face(
                            pos_t[parent].cpu().numpy() / sf,
                            size_t[parent].cpu().numpy() / (2.0 * sf),
                            pos_c_after,
                            size_t[child].cpu().numpy() / (2.0 * sf),
                            contact_axis=contact_ax,
                        )
                        pos_t[child] = torch.from_numpy(
                            (pc_m_after * sf).astype(np.float32)
                        ).to(device)
                    else:
                        pos_t[child] = out["pos_active_out"][0].detach()

                if POST_SHRINK_M > 0:
                    sz_m = size_t[child].cpu().numpy() / sf
                    sz_m[contact_ax] = max(sz_m[contact_ax] - POST_SHRINK_M, SIZE_MIN_M)
                    pc_m = _snap_to_contact_face(
                        pos_t[parent].cpu().numpy() / sf,
                        size_t[parent].cpu().numpy() / (2.0 * sf),
                        pos_t[child].cpu().numpy() / sf,
                        sz_m / 2.0,
                        contact_axis=contact_ax,
                    )
                    pos_t[child] = torch.from_numpy((pc_m * sf).astype(np.float32)).to(
                        device
                    )
                    size_t[child] = torch.from_numpy((sz_m * sf).astype(np.float32)).to(
                        device
                    )
                next_layer.add(child)
        committed |= next_layer
        current_layer = next_layer

    return {
        i: {
            "pos": (pos_t[i].cpu().numpy() / sf).astype(float),
            "size": (size_t[i].cpu().numpy() / sf).astype(float),
        }
        for i in blocks.keys()
    }


# =========================================================
# STAGE 4: SCREW PLACEMENT
# =========================================================
def _distribute_in_rect(
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    n: int,
) -> list[tuple[float, float]]:
    if n <= 0:
        return []
    if n == 1:
        return [((x_min + x_max) / 2.0, (y_min + y_max) / 2.0)]
    if n == 2:
        if (x_max - x_min) >= (y_max - y_min):
            return [(x_min, (y_min + y_max) / 2.0), (x_max, (y_min + y_max) / 2.0)]
        return [((x_min + x_max) / 2.0, y_min), ((x_min + x_max) / 2.0, y_max)]
    if n <= 4:
        return [(x_min, y_min), (x_max, y_min), (x_min, y_max), (x_max, y_max)][:n]
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


def _update_screws_for_repair(
    orig_screws: dict[int, list[tuple[int, np.ndarray]]],
    orig_blocks: dict[int, dict],
    repaired_blocks: dict[int, dict],
    conns: list[tuple[int, int]],
) -> dict[int, list[tuple[int, np.ndarray]]]:
    """Place each CSV screw in the repaired pair's tangential overlap rectangle
    (5mm safety inset). Multiple screws per pair are distributed; global XY+Z
    collision avoidance.
    """

    def _xy_dist_to_rect(px, py, xn, xx, yn, yx):
        dx = max(xn - px, 0.0, px - xx)
        dy = max(yn - py, 0.0, py - yx)
        return float(np.hypot(dx, dy))

    pair_to_screws: dict[tuple[int, int], list[tuple[int, int, np.ndarray]]] = {}
    for stored_idx, screw_list in orig_screws.items():
        for slot, orig_xyz in screw_list:
            best_pair = None
            best_score = float("inf")
            for a, b in conns:
                if orig_blocks[a]["pos"][2] >= orig_blocks[b]["pos"][2]:
                    up_o, lo_o = a, b
                else:
                    up_o, lo_o = b, a
                up_top = (
                    orig_blocks[up_o]["pos"][2] + orig_blocks[up_o]["size"][2] / 2.0
                )
                z_dist = abs(float(orig_xyz[2]) - up_top)
                lo, up = orig_blocks[lo_o], orig_blocks[up_o]
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
                    best_pair = (lo_o, up_o)
            if best_pair is None:
                continue
            pair_to_screws.setdefault(best_pair, []).append(
                (stored_idx, slot, orig_xyz)
            )

    new_screws: dict[int, list[tuple[int, np.ndarray]]] = {
        i: [] for i in orig_blocks.keys() | repaired_blocks.keys()
    }
    all_placed: list[np.ndarray] = []

    for (lo_idx, up_idx), screws_in_pair in pair_to_screws.items():
        if lo_idx not in repaired_blocks or up_idx not in repaired_blocks:
            for stored_idx, slot, orig_xyz in screws_in_pair:
                new_screws[stored_idx].append((slot, orig_xyz.copy()))
            continue
        lo = repaired_blocks[lo_idx]
        up = repaired_blocks[up_idx]
        ox_min = max(
            lo["pos"][0] - lo["size"][0] / 2.0, up["pos"][0] - up["size"][0] / 2.0
        )
        ox_max = min(
            lo["pos"][0] + lo["size"][0] / 2.0, up["pos"][0] + up["size"][0] / 2.0
        )
        oy_min = max(
            lo["pos"][1] - lo["size"][1] / 2.0, up["pos"][1] - up["size"][1] / 2.0
        )
        oy_max = min(
            lo["pos"][1] + lo["size"][1] / 2.0, up["pos"][1] + up["size"][1] / 2.0
        )
        sx_min, sx_max = ox_min + SCREW_SAFETY_M, ox_max - SCREW_SAFETY_M
        sy_min, sy_max = oy_min + SCREW_SAFETY_M, oy_max - SCREW_SAFETY_M
        if sx_min > sx_max:
            sx_min = sx_max = (ox_min + ox_max) / 2.0
        if sy_min > sy_max:
            sy_min = sy_max = (oy_min + oy_max) / 2.0
        up_top_new = up["pos"][2] + up["size"][2] / 2.0

        n_screws = len(screws_in_pair)
        positions = _distribute_in_rect(sx_min, sx_max, sy_min, sy_max, n_screws)
        orig_order = sorted(
            range(n_screws),
            key=lambda k: (
                float(screws_in_pair[k][2][0]),
                float(screws_in_pair[k][2][1]),
            ),
        )
        pos_sorted = sorted(positions, key=lambda p: (p[0], p[1]))

        for idx, (px, py) in zip(orig_order, pos_sorted):
            stored_idx, slot, _ = screws_in_pair[idx]
            pos = np.array([px, py, up_top_new], dtype=float)
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


# =========================================================
# MAIN ENTRY POINT
# =========================================================
def optimization(input_csv: str) -> str:
    """Run the full GNN-driven repair pipeline on a single-row CSV string
    and return the repaired CSV string with the SAME schema.
    """
    df = pd.read_csv(io.StringIO(input_csv))
    row = df.iloc[0]
    n = _detect_n_blocks(df)
    blocks = _parse_blocks(row, n)
    conns = _parse_connections(row, n)

    # Stage 1: analytical penetration shrink
    resolved = _resolve_penetrations(blocks)
    # Stage 2: analytical snap-to-contact BFS
    snapped = _apply_snaps_bfs(resolved, conns)
    # Stage 3: GNN-driven layered repair
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaded = _load_model()
    gnn_model = loaded["model"].to(device)
    encoder = loaded["encoder"].to(device)
    decoder = loaded["decoder"].to(device)
    repaired = _gnn_repair_layered(snapped, conns, gnn_model, encoder, decoder, device)

    # Stage 4: screw placement on the repaired overlap rectangles
    orig_screws = _parse_original_screws(row, n, n_screws=n)
    new_screws = _update_screws_for_repair(orig_screws, blocks, repaired, conns)

    out_row = row.copy()
    for bi, b in repaired.items():
        for k, ax in enumerate("XYZ"):
            out_row[f"Block{bi}_Pos{ax}"] = float(b["pos"][k])
            out_row[f"Block{bi}_Size{ax}"] = float(b["size"][k])
    # Clear all screw slots, then write the new ones
    for bi in range(n):
        for slot in range(n):
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


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 3:
        print("Usage: python optimization.py <input.csv> <output.csv>")
        sys.exit(1)
    in_path, out_path = Path(sys.argv[1]), Path(sys.argv[2])
    out_csv = optimization(in_path.read_text())
    out_path.write_text(out_csv)
    print(f"Wrote {out_path}")
