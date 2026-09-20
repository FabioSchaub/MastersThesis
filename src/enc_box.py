"""Encoder that turns the three half-extents of a box into the latent code the surrogate reads.

An axis-aligned box is fully described by three numbers, so the encoder is a plain MLP rather
than a point-cloud network: it takes the half-extents directly and is therefore a function, not
an estimate from a sampled surface. What makes the code more than a relabelling of the input is
where its target comes from. Stage 1 of ``src/enc_dec_training.py`` fits one free code per
training shape against the auxiliary signed-distance decoder, stage 2 distils this encoder onto
those codes, and stage 3 fine-tunes the two jointly.

This is the encoder used everywhere downstream: it builds the node features of the latent graph
and it defines the space the repair optimises in. Its inverse is a separate network,
:class:`~src.dec_box.BoxDecoder`.
"""

import sys
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).parent.parent))
from config.config import config


class BoxEncoder(nn.Module):
    """MLP mapping box half-extents to a latent code.

    Linear layers with ReLU in between and no activation on the output, so the code is
    unbounded. The shapes it is trained on are drawn log-uniformly between
    ``config.data.sampling_min`` and ``config.data.sampling_max`` metres per axis; outside that
    band the code is an extrapolation.
    """

    def __init__(
        self,
        latent_dim: int = config.autoencoder.latent_dim,
        hidden_dim: int = config.box_encoder.hidden_dim,
        num_layers: int = config.box_encoder.num_layers,
    ):
        """Build the MLP.

        Args:
            latent_dim: Width of the code, which is also the number of latent entries in a
                node feature of the surrogate.
            hidden_dim: Width of every hidden layer.
            num_layers: Number of linear layers including the output layer, so the number of
                hidden layers is one less.

        Raises:
            ValueError: If ``num_layers`` is smaller than one.
        """
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")

        # Kept as an attribute so that checkpoints can record the width without inspecting the
        # layer shapes.
        self.latent_dim = latent_dim

        layers: list[nn.Module] = []
        in_dim = 3
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim
        # No activation on the output: the code must be free to take any sign and magnitude,
        # since stage 1 fits it without constraining its range.
        layers.append(nn.Linear(in_dim, latent_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode a batch of boxes.

        Args:
            x: Half-extents in metres, shape ``(B, 3)``, in the order x, y, z.

        Returns:
            Latent codes of shape ``(B, latent_dim)``.
        """
        return self.net(x)
