"""Decoder that turns a latent code into a shape, as a signed distance field.

The decoder is queried point by point: given a code and a coordinate it returns the signed
distance of that coordinate to the surface of the encoded shape. A shape is therefore not
produced as a mesh but as a function, and a mesh is obtained where one is needed by marching
cubes over a grid of queries. That is also what makes the decoder differentiable in the code,
which is the property the repair of Part III relies on.

Three variants of the same multilayer perceptron are available: weight normalisation, which is
the default, sinusoidal activations, and spectral normalisation, which bounds the Lipschitz
constant of the decoder and is what makes the Least Volume compression meaningful. Only one is
active at a time, and which one it was is recorded in the checkpoint.
"""

import torch
from torch import nn
from torch.nn.utils.parametrizations import weight_norm
from pathlib import Path
import numpy as np
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from config.config import config
from src.least_volume import apply_spectral_norm


class SirenLayer(nn.Module):
    """Linear layer followed by a sine activation, as in a sinusoidal representation network.

    The initialisation is the one the sine activation requires: it keeps the pre-activations in
    a range over which the sine is still informative, and it differs for the first layer, which
    sees raw coordinates rather than the output of another sine.
    """

    def __init__(self, in_dim, out_dim, omega=30.0, is_first=False):
        """Build the layer and initialise its weights for the sine.

        Args:
            omega: Frequency the pre-activation is multiplied by before the sine.
            is_first: Whether the layer receives the raw input of the network.
        """
        super().__init__()
        self.omega = omega
        self.linear = nn.Linear(in_dim, out_dim)

        # Only the weights are re-initialised; the bias keeps the default of nn.Linear.
        with torch.no_grad():
            if is_first:
                self.linear.weight.uniform_(-1 / in_dim, 1 / in_dim)
            else:
                self.linear.weight.uniform_(
                    -np.sqrt(6 / in_dim) / omega,
                    np.sqrt(6 / in_dim) / omega,
                )

    def forward(self, x):
        """One layer: a linear map followed by a sine, scaled by the frequency factor."""
        return torch.sin(self.omega * self.linear(x))


class SDFDecoder(nn.Module):
    """Multilayer perceptron mapping a latent code and a query point to a signed distance.

    The code and the coordinate are simply concatenated at the input, so the same network is
    evaluated once per query point; a batch of points of one shape repeats the same code.

    The three variants are mutually exclusive and are tried in the order sinusoidal, spectral,
    weight-normalised, so passing ``use_siren`` overrides ``spectral``.

    Attributes:
        spectral: Whether every linear layer is spectrally normalised. Combined with the
            one-Lipschitz activations this bounds the Lipschitz constant of the network by one,
            which is what stops the Least Volume penalty from being satisfied trivially: without
            the bound the encoder can shrink the code arbitrarily and the decoder can amplify it
            straight back.
        out_scale: Factor applied to the output of the spectral variant. A one-Lipschitz network
            cannot cover the range of a signed distance field on its own, so the output is scaled
            back up and the network as a whole becomes Lipschitz with this constant.
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
        spectral: bool = False,  # Lipschitz-bound the decoder via spectral norm (Least Volume)
        lipschitz_k: float = 1.0,  # output scale -> overall K-Lipschitz (K>1 restores SDF range)
    ):
        """Build the perceptron in whichever of the three variants is requested.

        Args:
            num_layers: Total number of linear layers, the output layer included.
            point_dim: Width of a query coordinate, three.
            output_dim: Width of the output, one.
            spectral: Spectrally normalise every linear layer.
            lipschitz_k: Output scale of the spectral variant. It has no effect otherwise.
        """
        super().__init__()
        self.spectral = bool(spectral)
        self.out_scale = float(lipschitz_k)

        if use_siren:
            layers = [
                SirenLayer(latent_dim + point_dim, hidden_dim, omega, is_first=True)
            ]
            for _ in range(num_layers - 2):
                layers.append(SirenLayer(hidden_dim, hidden_dim, omega))
            layers.append(nn.Linear(hidden_dim, output_dim))
        elif self.spectral:
            # The layers are left plain so that the spectral normalisation applied below is the
            # only re-parametrisation. The rectifier in between is itself one-Lipschitz, so the
            # bound on the individual layers carries over to the whole network.
            layers = [nn.Linear(latent_dim + point_dim, hidden_dim), nn.ReLU()]
            for _ in range(num_layers - 2):
                layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
            layers += [nn.Linear(hidden_dim, output_dim)]
        else:
            layers = [
                weight_norm(nn.Linear(latent_dim + point_dim, hidden_dim)),
                nn.ReLU(),
            ]
            for _ in range(num_layers - 2):
                layers += [weight_norm(nn.Linear(hidden_dim, hidden_dim)), nn.ReLU()]
            layers += [nn.Linear(hidden_dim, output_dim)]

        self.net = nn.Sequential(*layers)
        if self.spectral:
            apply_spectral_norm(self.net)

    def forward(self, z, x):
        """Evaluate the signed distance of the query points under the given codes.

        Args:
            z: Latent codes, shape ``(..., latent_dim)``.
            x: Query coordinates, shape ``(..., point_dim)``, broadcast against ``z`` over the
               leading axes. One code per query point, so a single shape has its code repeated.

        Returns:
            The signed distances, of the shared leading shape of ``z`` and ``x``.
        """
        inp = torch.cat([z, x], dim=-1)
        out = self.net(inp).squeeze(-1)
        if self.spectral and self.out_scale != 1.0:
            out = out * self.out_scale
        return out


def sdf_decoder_from_ckpt(ck: dict, device="cpu", latent_dim: int | None = None) -> "SDFDecoder":
    """Build a decoder that matches the checkpoint's normalisation, then load the weights.

    Weight normalisation and spectral normalisation produce different keys in the state
    dictionary, so the variant has to be reconstructed before loading. A checkpoint without the
    flags is read as the weight-normalised default.

    Args:
        ck: A loaded checkpoint, expected to hold ``decoder`` and ``latent_dim``.
        latent_dim: Overrides the code width recorded in the checkpoint.
    """
    ld = latent_dim if latent_dim is not None else int(ck["latent_dim"])
    dec = SDFDecoder(latent_dim=ld,
                     spectral=bool(ck.get("decoder_spectral", False)),
                     lipschitz_k=float(ck.get("lipschitz_k", 1.0)))
    dec.load_state_dict(ck["decoder"])
    return dec.to(device)
