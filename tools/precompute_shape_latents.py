"""Compute the latent code of each shape of the vocabulary once, as a table.

The encoder is queried in the canonical frame, so it returns the same code for a shape whatever
size that shape is spawned at. The code therefore depends only on which of the fourteen shapes a
part is, and can be computed once instead of per configuration. The surrogate looks each part's
code up in this table and reads its size from the bounding box beside it.

The codes are produced from the exported meshes rather than from the analytic distance fields,
by sampling points on them: that is the same path a part takes through the simulator, so the
code stored here belongs to the object the simulator actually spawns.

The table is checked after it is written. If two shapes received codes pointing the same way,
the surrogate could not tell them apart and everything downstream would be measuring something
else, so the pairwise similarities and the number of varying entries are reported.

Run:  python -m tools.precompute_shape_latents

Writes:
    ``encoder_decoder_model/shape_latents_lv_v14_lam005.pt``, mapping mesh name to code.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch

_CODE = Path(__file__).resolve().parents[1]
if str(_CODE) not in sys.path:
    sys.path.insert(0, str(_CODE))

from src.enc_dec_dataset_generation import CANON_EXTENT  # noqa: E402
from src.enc_pointnet import pointnet_encoder_from_ckpt  # noqa: E402

# The compressed autoencoder selected for the thesis, named explicitly rather than taken from a
# pointer, so that the table cannot silently be rebuilt from a different one.
_CKPT = _CODE / "encoder_decoder_model" / "best_encoder_decoder_general_canon_lv_v14_lam005_latentdim32_active9.pth"
_OUT = _CODE / "encoder_decoder_model" / "shape_latents_lv_v14_lam005.pt"
_N_SURF = 4096
_RES = 96


def _load_shapes():
    """Load the shape definitions and the meshing from the export tool, by file path.

    Imported this way rather than as a module, so that the tool works from any directory. The
    definitions must be the same ones the simulator was given.
    """
    p = _CODE / "tools" / "export_shape_objs.py"
    spec = importlib.util.spec_from_file_location("export_shape_objs", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _to_canonical(surf: np.ndarray) -> np.ndarray:
    """Centre and scale a sampled surface into the encoder's frame, as the training does."""
    surf = np.asarray(surf, np.float64)
    centred = surf - surf.mean(axis=0)
    scale = CANON_EXTENT / (float(np.ptp(centred, axis=0).max()) + 1e-12)
    return (centred * scale).astype(np.float32)


@torch.no_grad()
def main() -> None:
    """Encode every shape of the vocabulary, write the table, and check that the codes differ."""
    if not _CKPT.exists():
        raise SystemExit(f"LV checkpoint not found: {_CKPT}")
    ck = torch.load(_CKPT, map_location="cpu", weights_only=False)
    latent_dim = int(ck.get("latent_dim", 32)) if isinstance(ck, dict) else 32
    enc = pointnet_encoder_from_ckpt(ck, latent_dim=latent_dim)
    enc.eval()
    print(f"loaded LV encoder: {_CKPT.name} (latent_dim={latent_dim})")

    ex = _load_shapes()
    shapes = ex._shapes()

    lut: dict[str, torch.Tensor] = {}
    for name, (sdf_fn, half) in shapes.items():
        # Meshed and then sampled, rather than sampled from the field directly: this is the
        # object the simulator spawns, meshing artefacts included.
        mesh = ex._mesh_from_sdf(sdf_fn, half, _RES)
        surf = np.asarray(mesh.sample(_N_SURF), dtype=np.float32)
        x = torch.from_numpy(_to_canonical(surf)).unsqueeze(0)
        z = enc(x).squeeze(0).cpu()
        # Keyed by the name the simulator uses, which is how the dataset resolves a row.
        lut[f"{name}.usd"] = z

    torch.save(lut, _OUT)
    print(f"saved {len(lut)} shape latents -> {_OUT.name}")

    names = list(lut)
    Z = torch.stack([lut[n] for n in names])
    Zn = torch.nn.functional.normalize(Z, dim=1)
    cos = Zn @ Zn.T
    off = cos[~torch.eye(len(names), dtype=torch.bool)]
    print(f"\nz-norm: min {Z.norm(dim=1).min():.3f} max {Z.norm(dim=1).max():.3f}")
    print(f"pairwise cos (off-diag): mean {off.mean():.3f} max {off.max():.3f}  (max<1 => all distinct)")
    active = (Z.std(dim=0) > 1e-3).sum().item()
    print(f"active latent dims (std>1e-3 across shapes): {active}/{Z.shape[1]}")
    # The closest pair is named, because a similarity near one is only a problem if it is
    # between two shapes that ought to be told apart.
    c2 = cos.clone(); c2.fill_diagonal_(-1)
    i = int(c2.argmax()); a, b = divmod(i, len(names))
    print(f"most similar pair: {names[a]} ~ {names[b]} (cos {c2.max():.3f})")


if __name__ == "__main__":
    main()
