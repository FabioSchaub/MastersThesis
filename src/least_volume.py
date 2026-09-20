"""Building blocks of the Least Volume regulariser, after Chen and Fuge.

The idea is to make an autoencoder describe its shapes with as few of the entries of the latent
code as it can, without being told in advance how many that is. Two pieces are needed and both
are here:

    the volume penalty     the supplemented geometric mean of the standard deviations of the
                           code entries over a set of shapes. It is the volume of the box that
                           encloses the codes, so pushing it down flattens the box against as
                           many axes as the reconstruction can spare, and the entries that are
                           flattened stop carrying information.
    a Lipschitz bound      on the decoder. Without it the penalty is satisfied for free: the
                           encoder shrinks every entry towards zero and the decoder scales the
                           code straight back up, so the box has no volume and nothing has been
                           compressed. Spectral normalisation of every linear layer removes
                           that escape.

The remaining functions read the compression off a set of codes rather than enforcing it. The
wiring into the training, which stage the penalty acts in and where the settings come from, is
in :mod:`src.enc_dec_training`.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn.utils.parametrizations import spectral_norm


def latent_std(z: torch.Tensor) -> torch.Tensor:
    """Standard deviation of each code entry over a set of codes.

    Args:
        z: Latent codes, shape ``(N, D)``.

    Returns:
        One standard deviation per entry, shape ``(D,)``.
    """
    return z.std(dim=0)


def volume_penalty(z: torch.Tensor, eta: float = 1.0) -> torch.Tensor:
    """Supplemented geometric mean of the standard deviations of the code entries.

    It is evaluated through logarithms, which keeps it finite when an entry has collapsed. The
    offset is what makes the penalty differentiable there in the first place, and it also sets
    the character of the gradient: a small offset presses hardest on the entries that are
    already small and so drives them to zero, a large one weighs all entries about equally and
    the penalty degenerates into a plain sum of the standard deviations.

    Args:
        z: Latent codes, shape ``(N, D)``. During training these are the codes of one batch,
            so the estimate is as noisy as the batch is small.
        eta: The offset added to every standard deviation.
    """
    sigma = latent_std(z)
    return torch.exp(torch.log(sigma + eta).mean())


def sigma_spectrum(z: torch.Tensor) -> np.ndarray:
    """Standard deviations of the code entries, sorted from largest to smallest.

    A few large values followed by a drop towards zero is what compression looks like, and the
    position of the drop is the number of entries the data actually occupies.
    """
    return np.sort(latent_std(z).detach().cpu().numpy())[::-1].copy()


def active_dims(z: torch.Tensor, frac: float = 0.01) -> int:
    """Count the code entries whose standard deviation exceeds a fraction of the largest one.

    The count is a reading of the spectrum, not a property of the model: it depends on where
    the fraction is placed, and the compressed codes are evaluated at both one and ten per cent
    because the drop is gradual rather than sharp.
    """
    s = sigma_spectrum(z)
    if s.size == 0 or s[0] == 0:
        return 0
    return int((s > frac * s[0]).sum())


def apply_spectral_norm(net: nn.Module, n_power_iterations: int = 1) -> nn.Module:
    """Spectrally normalise every linear layer of a module in place.

    This must run before a state dictionary is loaded: it re-parametrises the weights and
    therefore changes the keys.

    Args:
        n_power_iterations: Number of power iterations used to estimate the largest singular
            value at every forward pass.
    """
    for m in net.modules():
        if isinstance(m, nn.Linear):
            spectral_norm(m, n_power_iterations=n_power_iterations)
    return net
