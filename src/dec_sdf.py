"""Auxiliary decoder that reads a latent code and a point and returns a signed distance.

This is the first of the two decoders and the one that does not appear in the repair. It exists
only for stage 1 of ``src/enc_dec_training.py``, where it is trained jointly with one free code
per training shape: the code has to carry enough about the box for this network to reproduce
its signed-distance field, which is what gives the code a geometric meaning instead of an
arbitrary one. Once the codes exist, stage 2 distils the box encoder onto them and this decoder
is set aside. The decoder that turns a code back into half-extents is
:class:`~src.dec_box.BoxDecoder`.

The shape is conditioned on by concatenation: the code is repeated over the query points and
appended to the coordinates, so a single MLP represents the whole family of fields.
"""

import torch
from torch import nn
from torch.nn.utils.parametrizations import weight_norm
from pathlib import Path
import numpy as np
import sys

# Add parent directory to path to import config
sys.path.insert(0, str(Path(__file__).parent.parent))
from config.config import config


class SirenLayer(nn.Module):
    """A linear layer followed by a sine activation, as in a sinusoidal representation network.

    Args:
        in_dim: Input width.
        out_dim: Output width.
        omega: Frequency the pre-activation is scaled by before the sine.
        is_first: Whether this is the input layer, which is initialised differently.
    """

    def __init__(self, in_dim, out_dim, omega=30.0, is_first=False):
        super().__init__()
        self.omega = omega
        self.linear = nn.Linear(in_dim, out_dim)

        # The initialisation of the reference formulation. The first layer spans one period of
        # the sine over the input range; every later layer keeps the pre-activation variance
        # constant despite the multiplication by omega.
        with torch.no_grad():
            if is_first:
                self.linear.weight.uniform_(-1 / in_dim, 1 / in_dim)
            else:
                self.linear.weight.uniform_(
                    -np.sqrt(6 / in_dim) / omega,
                    np.sqrt(6 / in_dim) / omega,
                )

    def forward(self, x):
        return torch.sin(self.omega * self.linear(x))


class SDFDecoder(nn.Module):
    """MLP that predicts the signed distance at a point, conditioned on a shape code.

    Two variants of the same network, selected by ``use_siren``: weight-normalised linear
    layers with ReLU, or sine layers. The default in the configuration is the ReLU variant;
    boxes have no high-frequency detail that would justify the sine one.

    Args:
        latent_dim: Width of the shape code.
        hidden_dim: Width of every hidden layer.
        num_layers: Number of layers including the input and the output layer.
        point_dim: Width of a query coordinate, three.
        output_dim: Width of the prediction, one signed distance.
        use_siren: Whether to build sine layers instead of weight-normalised ReLU layers.
        omega: Frequency of the sine layers, ignored when ``use_siren`` is false.
    """

    def __init__(
        self,
        latent_dim=config.sdf_decoder.latent_dim,  # Dimensionality of the input latent code
        hidden_dim=config.sdf_decoder.hidden_dim,  # Number of hidden units in each layer of the MLP
        num_layers=config.sdf_decoder.num_layers,  # Total number of layers in the MLP (including output layer)
        point_dim=config.sdf_decoder.point_dim,  # Dimensionality of the input point coordinates (e.g., 3 for 3D)
        output_dim=config.sdf_decoder.output_dim,  # Dimensionality of the output SDF value (usually 1 for scalar)
        use_siren=config.sdf_decoder.use_siren,  # Toggle for using SIREN activations in the MLP decoder
        omega=config.sdf_decoder.omega,  # Omega parameter for SIREN activations (30.0)
    ):
        super().__init__()

        if use_siren:
            layers = [
                SirenLayer(latent_dim + point_dim, hidden_dim, omega, is_first=True)
            ]
            for _ in range(num_layers - 2):
                layers.append(SirenLayer(hidden_dim, hidden_dim, omega))
            layers.append(nn.Linear(hidden_dim, output_dim))
        else:
            layers = [
                weight_norm(nn.Linear(latent_dim + point_dim, hidden_dim)),
                nn.ReLU(),
            ]
            for _ in range(num_layers - 2):
                layers += [weight_norm(nn.Linear(hidden_dim, hidden_dim)), nn.ReLU()]
            layers += [nn.Linear(hidden_dim, output_dim)]

        self.net = nn.Sequential(*layers)

    def forward(self, z, x):
        """Predict the signed distance at every query point.

        Args:
            z: Shape codes broadcast to one per query point, shape ``(..., latent_dim)``.
            x: Query coordinates, shape ``(..., point_dim)``, in the normalised frame the
                training samples were generated in.

        Returns:
            Signed distances with the trailing width dropped, shape ``(...,)``.
        """
        inp = torch.cat([z, x], dim=-1)
        return self.net(inp).squeeze(-1)
