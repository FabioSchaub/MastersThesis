"""Encoder for the box vocabulary: a perceptron from three half extents to a latent code.

An axis-aligned box is described exactly by its three half extents, so the encoder can read the
geometry directly instead of inferring it from a sampled surface. This is the encoder of the
earlier parts of the thesis, where the vocabulary held nothing but boxes.

On this branch the shapes are general and the encoder in use is
:class:`src.enc_pointnet.Autoencoder`; this module is selected only when the training is run in
its box configuration, and is kept so that the branches share one training file. The two
encoders present the same interface, so which one is built is a single decision in
:mod:`src.enc_dec_training`.
"""

import sys
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).parent.parent))
from config.config import config


class BoxEncoder(nn.Module):
    """Perceptron mapping the half extents of a box to a latent code.

    Every hidden layer is followed by a rectifier; the output layer is linear, so the code is
    not constrained in sign or magnitude.

    Attributes:
        latent_dim: Width of the code, recorded here so that the checkpoint code does not have
            to reach into the layers.
    """

    def __init__(
        self,
        latent_dim: int = config.autoencoder.latent_dim,
        hidden_dim: int = config.box_encoder.hidden_dim,
        num_layers: int = config.box_encoder.num_layers,
    ):
        """Build the perceptron.

        Args:
            num_layers: Total number of linear layers, the output layer included, so a value of
                one gives a single linear map from the half extents to the code.

        Raises:
            ValueError: If ``num_layers`` is smaller than one.
        """
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")

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
        """Encode a batch of boxes.

        Args:
            x: Half extents, shape ``(B, 3)``, in the same units as the shapes were generated
                in.

        Returns:
            The latent codes, shape ``(B, latent_dim)``.
        """
        return self.net(x)
