"""Point cloud encoder that maps a surface sample of a shape to its latent code.

The architecture is the one of PointNet: a multilayer perceptron applied to every point
separately, a maximum over the points, and a second perceptron down to the code. The maximum
is what makes the result invariant to the order of the points and to how many of them there
are, which is what lets a sampled surface stand in for the shape itself.

The encoder sees shapes in the canonical frame only, centred and uniformly scaled, so the code
describes the form of a part and never its size. Size is carried separately, as a bounding box.
With the widths of the configuration and a code of 32 entries the module has 333,472
parameters.
"""

from torch import nn
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
from config.config import config


class Autoencoder(nn.Module):
    """PointNet-style encoder from a surface point cloud to a latent code.

    Despite the name the class is an encoder only; the matching decoder is
    :class:`src.dec_sdf.SDFDecoder`. The name and the ``latent_dim`` attribute mirror
    :class:`src.enc_box.BoxEncoder`, so that the training and checkpoint code can treat the two
    interchangeably.

    Attributes:
        spectral: Whether every linear layer is spectrally normalised, which bounds the
            Lipschitz constant of the encoder. This is the optional bi-Lipschitz variant used
            with the Least Volume compression, not the default.
        latent_dim: Width of the code the encoder produces.
    """

    def __init__(
        self,
        latent_dim: int = config.autoencoder.latent_dim,
        point_mlp_dims: list[int] | None = None,
        fc_dims: list[int] | None = None,
        spectral: bool = False,  # Lipschitz-bound the encoder via spectral norm (bi-Lipschitz extra)
    ):
        """Build the per-point perceptron and the perceptron after the pooling.

        Args:
            point_mlp_dims: Widths of the per-point layers. The last one is the width of the
                pooled feature. ``None`` takes the value from the configuration.
            fc_dims: Widths of the layers between the pooled feature and the code, without the
                final projection onto ``latent_dim``. ``None`` takes the value from the
                configuration.
            spectral: Spectrally normalise every linear layer.

        Raises:
            ValueError: If ``point_mlp_dims`` is empty, in which case there is nothing to pool.
        """
        super().__init__()
        self.spectral = bool(spectral)
        point_mlp_dims = (
            point_mlp_dims
            if point_mlp_dims is not None
            else list(config.autoencoder.encoder_point_mlp_dims)
        )
        fc_dims = (
            fc_dims if fc_dims is not None else list(config.autoencoder.encoder_fc_dims)
        )

        if len(point_mlp_dims) == 0:
            raise ValueError("point_mlp_dims must contain at least one layer width")

        self.latent_dim = latent_dim

        encoder_layers: list[nn.Module] = []
        in_dim = 3
        for out_dim in point_mlp_dims:
            encoder_layers.append(nn.Linear(in_dim, out_dim))
            encoder_layers.append(nn.ReLU())
            in_dim = out_dim
        # The activation after the last per-point layer is dropped on purpose, so that the
        # pooled feature is not restricted to non-negative values.
        encoder_layers = encoder_layers[:-1]
        self.encoder = nn.Sequential(*encoder_layers)

        fc_layers: list[nn.Module] = []
        in_dim = point_mlp_dims[-1]
        for out_dim in fc_dims:
            fc_layers.append(nn.Linear(in_dim, out_dim))
            fc_layers.append(nn.ReLU())
            in_dim = out_dim
        fc_layers.append(nn.Linear(in_dim, latent_dim))
        self.fc = nn.Sequential(*fc_layers)

        if self.spectral:
            from src.least_volume import apply_spectral_norm
            apply_spectral_norm(self.encoder)
            apply_spectral_norm(self.fc)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode a batch of surface point clouds.

        Args:
            x: Surface points in the canonical frame, shape ``(B, N, 3)``. ``N`` may differ
               between calls; the pooling makes the result independent of it.

        Returns:
            The latent codes, shape ``(B, latent_dim)``.
        """
        features: torch.Tensor = self.encoder(x)
        global_feat: torch.Tensor = features.max(dim=1)[0]
        return self.fc(global_feat)


def pointnet_encoder_from_ckpt(ck: dict, device="cpu", latent_dim: int | None = None) -> "Autoencoder":
    """Build an encoder that matches the checkpoint's normalisation, then load the weights.

    Spectral normalisation re-parametrises a linear layer and therefore changes the keys of the
    state dictionary. Constructing the module by hand and calling ``load_state_dict`` only works
    if the flag happens to be guessed right, so every caller should go through this function.
    A checkpoint without the flag is read as a plain encoder.

    Args:
        ck: A loaded checkpoint, expected to hold ``encoder`` and ``latent_dim``.
        latent_dim: Overrides the code width recorded in the checkpoint.
    """
    ld = latent_dim if latent_dim is not None else int(ck["latent_dim"])
    enc = Autoencoder(latent_dim=ld, spectral=bool(ck.get("encoder_spectral", False)))
    enc.load_state_dict(ck["encoder"])
    return enc.to(device)
