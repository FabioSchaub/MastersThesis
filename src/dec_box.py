"""Decoder from a latent code back to the three half-extents of a box.

This is the decoder that stands in the repair loop, not the auxiliary signed-distance decoder
of stage 1. It is trained afterwards by ``tools/train_box_decoder.py`` on codes produced by the
frozen :class:`~src.enc_box.BoxEncoder`, so it inverts exactly the mapping the surrogate was
trained under. The repair keeps its Adam parameters in latent space and calls this decoder
whenever it needs a size: for the hard caps on the block dimensions, for the analytical snap,
for the drift term, and for the sizes written back to the design.

The network predicts the logarithm of the half-extents and the forward exponentiates, so the
error stays relative rather than absolute across the log-uniform range the encoder was trained
on. It is wider and deeper than the encoder because inverting ``latent_dim`` numbers back to
three is the harder direction.
"""

import sys
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).parent.parent))
from config.config import config

_DECODER_DIR = Path(__file__).parent.parent / config.autoencoder.autoencoder_folder
_LATENT_DIM = int(config.autoencoder.latent_dim)
_DECODER_CKPT = _DECODER_DIR / f"best_box_decoder_latentdim{_LATENT_DIM}.pth"

# Cached frozen decoder, loaded lazily on first call.
_BOX_DECODER: "BoxDecoder | None" = None


def get_box_decoder(device: str = "cpu") -> "BoxDecoder":
    """Return the frozen decoder, loading and caching it on first call.

    The repair calls this once per process and reuses the instance for every step of every
    pair, so the checkpoint is read from disk only once. The returned module is in evaluation
    mode with all parameters detached from the graph: the repair differentiates through it
    with respect to the code, never with respect to its weights.

    Args:
        device: Torch device the module is moved to on the first call. Later calls return the
            cached instance and ignore this argument.

    Raises:
        SystemExit: If the checkpoint is missing or does not carry a ``decoder`` state dict.
            Stopping here is deliberate: without the decoder a repair could still run in
            latent space but could never be turned back into a design.
    """
    global _BOX_DECODER
    if _BOX_DECODER is None:
        if not _DECODER_CKPT.exists():
            raise SystemExit(
                f"BoxDecoder checkpoint not found: {_DECODER_CKPT}\n"
                "Train the decoder first: python tools/train_box_decoder.py"
            )
        ckpt = torch.load(_DECODER_CKPT, map_location="cpu", weights_only=False)
        if "decoder" not in ckpt:
            raise SystemExit(
                f"BoxDecoder checkpoint {_DECODER_CKPT.name} has no 'decoder' key; "
                f"available keys: {list(ckpt.keys())}"
            )
        decoder = BoxDecoder(latent_dim=_LATENT_DIM)
        decoder.load_state_dict(ckpt["decoder"])
        decoder.eval()
        for p in decoder.parameters():
            p.requires_grad_(False)
        _BOX_DECODER = decoder.to(device)
        print(
            f" Loaded frozen BoxDecoder: {_DECODER_CKPT.name}  "
            f"(latent_dim={_LATENT_DIM}, val_mae_mm="
            f"{ckpt.get('val_mae_mm', 'n/a')})"
        )
    return _BOX_DECODER


class BoxDecoder(nn.Module):
    """MLP mapping a latent code back to box half-extents.

    Linear layers with GELU in between. The output layer produces the logarithm of the three
    half-extents and :meth:`forward` exponentiates it, which also makes a negative extent
    unreachable by construction.
    """

    def __init__(
        self,
        latent_dim: int = config.autoencoder.latent_dim,
        hidden_dim: int = 256,
        num_layers: int = 6,
    ):
        """Build the MLP.

        Args:
            latent_dim: Width of the code, which must match the encoder that produced it.
            hidden_dim: Width of every hidden layer.
            num_layers: Number of linear layers including the output layer.

        Raises:
            ValueError: If ``num_layers`` is smaller than one.
        """
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")

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
        """Decode a batch of codes into half-extents.

        Args:
            z: Latent codes, shape ``(B, latent_dim)``.

        Returns:
            Half-extents in metres, shape ``(B, 3)``, in the order x, y, z. A full edge length
            is twice this.
        """
        log_half = self.net(z)
        return torch.exp(log_half)

    def forward_log(self, z: torch.Tensor) -> torch.Tensor:
        """Return the logarithm of the half-extents without exponentiating.

        The training loss is formulated in log space, so it uses this rather than taking the
        logarithm of :meth:`forward` and losing precision on the way.
        """
        return self.net(z)
