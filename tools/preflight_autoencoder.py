"""Answer, on a laptop and in seconds, the two questions a cluster job answers in a day.

The first is whether the current configuration runs at all. The encoder and the decoder are
built from it and pushed through one forward and one backward pass on a tiny batch, which
catches the mismatched widths and unsupported settings that otherwise fail in the first seconds
of a submitted job.

The second is whether the training fits in the memory of the card. The backward pass of the
first stage stores one activation per query point of the batch, and that term dominates
everything else, so the peak grows with the batch size times the query count. The activations
are measured here by attaching hooks to a small forward pass and scaled up, with a factor that
depends on the activation function, since the sinusoidal variant has to keep more state than the
rectifier. A sweep over batch sizes is printed against the memory budget.

The estimate is on the pessimistic side by design: being told a run is too large when it would
have fitted costs a re-run, while the opposite costs a job that fails hours in.

Run:
    python -m tools.preflight_autoencoder
    python -m tools.preflight_autoencoder --vram 24 --smoke

Writes nothing, except with ``--smoke``, which runs the whole training briefly on the processor
and deletes the checkpoints it produced afterwards.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import torch
from torch import nn

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from config.config import config
from src.dec_sdf import SDFDecoder
from src.enc_pointnet import Autoencoder as PointNetEncoder

# Calibrated against runs that did and did not fit on the cards available. The first is the
# fixed cost of the context, the parameters and the state of the optimiser; the other two say
# how much more than the forward activations the backward pass has to keep, which is more for
# the sine, whose derivative needs the pre-activation, than for the rectifier.
OVERHEAD_GB = 1.5
BWD_FACTOR_SIREN = 2.0
BWD_FACTOR_RELU = 1.4


def _measure_act_floats_per_unit(module: nn.Module, run_forward, probe_units: int) -> float:
    """Activation elements produced by one forward pass, per query point or per shape."""
    totals: list[int] = []
    handles = []

    def hook(_m, _inp, out):
        """Record the size of one layer's output; non-tensor outputs hold no activations."""
        if isinstance(out, torch.Tensor):
            totals.append(out.numel())

    for m in module.modules():
        # Leaves only: a container reports the output of its last child, which is already
        # counted, and including both would double the estimate.
        if len(list(m.children())) == 0:
            handles.append(m.register_forward_hook(hook))
    try:
        run_forward()
    finally:
        for h in handles:
            h.remove()
    return sum(totals) / max(probe_units, 1)


def estimate(batch_size: int, n_query: int, n_surface: int,
             dec_floats_per_pt: float, enc_floats_per_shape: float,
             use_siren: bool) -> dict:
    """Peak memory in gigabytes for the two stages that train the decoder.

    Stage 1 has no encoder, so only the decoder's activations count; stage 3 adds the encoder's.
    Stage 2 is not estimated: it trains the encoder alone and is far cheaper than either.
    """
    bwd = BWD_FACTOR_SIREN if use_siren else BWD_FACTOR_RELU
    # Four bytes per element, the decoder being evaluated once per query point of every shape
    # in the batch and the encoder once per shape.
    dec_bytes = dec_floats_per_pt * batch_size * n_query * 4
    enc_bytes = enc_floats_per_shape * batch_size * 4
    stage1 = OVERHEAD_GB + dec_bytes * bwd / 1e9
    stage3 = OVERHEAD_GB + (dec_bytes + enc_bytes) * bwd / 1e9
    return {"stage1": stage1, "stage3": stage3}


def main() -> int:
    """Build the models, estimate the memory, and optionally run the whole training briefly.

    Returns:
        Zero, or the exit code of the brief training run when one was requested.
    """
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--vram", type=float, default=10.57, help="GPU budget in GB (Euler default 10.57).")
    p.add_argument("--smoke", action="store_true", help="Also run the full AE_QUICK CPU smoke.")
    args = p.parse_args()

    dev = torch.device("cpu")
    ld = int(config.autoencoder.latent_dim)
    nq = int(config.autoencoder.n_query)
    ns = int(config.autoencoder.n_surface)
    bs = int(config.autoencoder.batch_size)
    use_siren = bool(config.sdf_decoder.use_siren)

    print("Config:")
    print(f"  latent_dim={ld}  n_query={nq}  n_surface={ns}  batch_size={bs}")
    print(f"  decoder: hidden={config.sdf_decoder.hidden_dim} layers={config.sdf_decoder.num_layers} "
          f"siren={use_siren}  clamp_delta={config.training.sdf_clamp_delta}")

    print("\n[1/2] Build + forward/backward on a tiny CPU batch ...")
    decoder = SDFDecoder(latent_dim=ld).to(dev)
    encoder = PointNetEncoder(latent_dim=ld).to(dev)
    # A handful of shapes is enough: what is checked is that the shapes of the tensors agree,
    # and the memory is extrapolated from a rate rather than from an absolute measurement.
    pb = 4
    z = torch.randn(pb, ld, requires_grad=True)
    q = torch.randn(pb, nq, 3)

    def dec_forward():
        """The same evaluation the training performs, so the same activations are produced."""
        zexp = z.unsqueeze(1).expand(pb, nq, ld)
        inp = torch.cat([zexp, q], dim=-1).reshape(pb * nq, ld + 3)
        return decoder.net(inp).squeeze(-1).reshape(pb, nq)

    surf = torch.randn(pb, ns, 3)
    sdf_pred = dec_forward()
    z_enc = encoder(surf)
    loss = sdf_pred.abs().mean() + z_enc.abs().mean()
    loss.backward()
    print("      OK - model builds, forward + backward run, no shape/SIREN errors.")

    print("\n[2/2] GPU memory estimate (calibrated to observed OOMs) ...")
    dec_fpp = _measure_act_floats_per_unit(decoder, dec_forward, pb * nq)
    enc_fps = _measure_act_floats_per_unit(encoder, lambda: encoder(surf), pb)
    bwd = BWD_FACTOR_SIREN if use_siren else BWD_FACTOR_RELU
    print(f"      decoder ~{dec_fpp:.0f} act-floats/point   encoder ~{enc_fps:.0f} act-floats/shape   "
          f"bwd_factor={bwd}")

    est = estimate(bs, nq, ns, dec_fpp, enc_fps, use_siren)
    verdict = "PASS" if max(est["stage1"], est["stage3"]) <= args.vram else "FAIL — will OOM"
    print(f"\n  Current batch_size={bs}:  Stage-1 ~{est['stage1']:.1f} GB   "
          f"Stage-3 ~{est['stage3']:.1f} GB   (budget {args.vram} GB)  ->  {verdict}")

    print(f"\n  Batch-size sweep (peak = max(stage1, stage3) GB):")
    print(f"  {'batch':>6} {'stage1':>8} {'stage3':>8}  fits {args.vram}GB?")
    for b in (256, 192, 128, 96, 64, 48, 32):
        e = estimate(b, nq, ns, dec_fpp, enc_fps, use_siren)
        peak = max(e["stage1"], e["stage3"])
        mark = "yes" if peak <= args.vram else "NO"
        star = "  <- current" if b == bs else ""
        print(f"  {b:>6} {e['stage1']:>7.1f} {e['stage3']:>7.1f}      {mark:>3}{star}")

    if max(est["stage1"], est["stage3"]) > args.vram:
        print("\n  >> Lower autoencoder.batch_size (or n_query) until it fits, then re-run this preflight.")

    if args.smoke:
        print("\n[smoke] Running full pipeline on CPU (AE_QUICK=1, ~1 min) ...")
        # The brief run trains on a few hundred shapes and writes checkpoints under the same
        # names a real run would. The folder is listed before and after and only the files that
        # appeared are removed, so a worthless checkpoint can never be mistaken for a real one
        # while an existing real one is left untouched.
        model_dir = ROOT / config.autoencoder.autoencoder_folder
        before = set(model_dir.glob(f"*latentdim{ld}*.pth"))
        # The card is hidden from the subprocess: the point of this run is to exercise the code
        # path the cluster takes, and it must not silently use a device that is not there.
        env = {**os.environ, "AE_QUICK": "1", "GENERAL_SHAPES": "1",
               "WANDB_MODE": "offline", "CUDA_VISIBLE_DEVICES": ""}
        r = subprocess.run([sys.executable, str(ROOT / "src" / "enc_dec_training.py")],
                           env=env, cwd=str(ROOT))
        new_files = set(model_dir.glob(f"*latentdim{ld}*.pth")) - before
        for f in new_files:
            f.unlink()
        print(f"[smoke] exit code {r.returncode}  ({'OK' if r.returncode == 0 else 'FAILED'})  "
              f"cleaned {len(new_files)} smoke checkpoint(s)")
        return r.returncode

    return 0


if __name__ == "__main__":
    sys.exit(main())
