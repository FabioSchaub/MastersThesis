"""Check the trained shape autoencoder against parts it was not trained on.

The autoencoder is fitted on shapes this repository generates. The question this module answers
is whether it also represents the parts of an externally generated assembly, or whether it has
merely learned its own training distribution. The parts are read as point clouds and the answer
is a reconstruction error, per part and grouped by shape type, so that a failure attributable to
one type stands out immediately rather than being averaged away.

The error is the magnitude of the decoded distance field evaluated at the input surface points.
A faithful reconstruction has a surface passing through those points, so the field is zero
there. Measuring it this way needs no mesh and no correspondence, which is why it is the primary
figure. It is reported both in the units of the canonical frame, where it is comparable with the
training loss, and converted back to millimetres at the part's real size.

Beyond the check, this module supplies the canonical normalisation and the two evaluation
helpers that the tools of the vocabulary reuse; :func:`to_canonical` in particular has to agree
with the normalisation the training applies, or the encoder is queried in a frame it never saw.

Run:
    python -m src.validate_shape_encoder
    python -m src.validate_shape_encoder --candidate candidate_02 --no_html
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
from config.config import config
from src.dec_sdf import SDFDecoder
from src.enc_pointnet import Autoencoder as PointNetEncoder

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_FIGURE = "a_cactus_with_a_round_cylinder_trunk_and"


def load_models(checkpoint_path: Path, device: torch.device):
    """Load the encoder and the decoder of a checkpoint, in evaluation mode.

    Both are built through the helpers that read the normalisation flags of the checkpoint,
    since a compressed autoencoder is parametrised differently and would not load otherwise.

    Returns:
        The encoder, the decoder, and the width of the latent code.
    """
    ck = torch.load(checkpoint_path, map_location=device, weights_only=False)
    latent_dim = int(ck["latent_dim"])
    if ck.get("encoder_type", "pointnet") != "pointnet":
        print(f"WARNING: checkpoint encoder_type={ck.get('encoder_type')!r}, expected 'pointnet'.")
    from src.enc_pointnet import pointnet_encoder_from_ckpt
    from src.dec_sdf import sdf_decoder_from_ckpt
    encoder = pointnet_encoder_from_ckpt(ck, device=device)
    decoder = sdf_decoder_from_ckpt(ck, device=device)
    encoder.eval()
    decoder.eval()
    print(f"Loaded {checkpoint_path.name}  (latent_dim={latent_dim}, "
          f"type={ck.get('encoder_type')}, ts={ck.get('timestamp')})")
    return encoder, decoder, latent_dim


def to_canonical(cloud_m: np.ndarray, canon_extent: float) -> tuple[np.ndarray, float, np.ndarray]:
    """Centre and uniformly scale a point cloud into the frame the encoder was trained in.

    The scaling is uniform, so the proportions of the part are preserved and only its size is
    removed. That is the separation the whole representation rests on: the code describes form,
    and size is carried alongside it as a bounding box.

    This has to agree with the normalisation applied when the training shapes are generated,
    including the use of the centroid of the points rather than the centre of the bounding box.

    Args:
        cloud_m: The point cloud in metres, shape ``(N, 3)``.
        canon_extent: Largest extent the cloud is scaled to.

    Returns:
        The normalised cloud, the scale factor that was applied, and the centre that was
        subtracted. A distance of one in the normalised frame is the reciprocal of the scale
        factor in metres.
    """
    centre = cloud_m.mean(axis=0)
    centred = cloud_m - centre
    max_extent = float(np.ptp(centred, axis=0).max()) + 1e-12
    s = canon_extent / max_extent
    return (centred * s).astype(np.float32), s, centre


@torch.no_grad()
def encode(encoder, cloud_canon: np.ndarray, device) -> torch.Tensor:
    """Encode one cloud already in the canonical frame, shape ``(N, 3)``, to its code."""
    x = torch.from_numpy(cloud_canon).unsqueeze(0).to(device)
    return encoder(x).squeeze(0)


@torch.no_grad()
def decode_sdf_at(decoder, z: torch.Tensor, points: np.ndarray, device, batch: int = 65536) -> np.ndarray:
    """Evaluate the decoded distance field of one code at arbitrary points.

    Args:
        points: Query points in the canonical frame, shape ``(N, 3)``.
        batch: Query points per forward pass. The code is repeated once per point, so a whole
            marching-cubes grid at once would not fit.

    Returns:
        The signed distances, shape ``(N,)``, in the units of the canonical frame.
    """
    pts = torch.from_numpy(points.astype(np.float32)).to(device)
    z_exp = z.unsqueeze(0).expand(pts.shape[0], -1)
    out = []
    for s in range(0, pts.shape[0], batch):
        out.append(decoder(z_exp[s:s + batch], pts[s:s + batch]).cpu().numpy())
    return np.concatenate(out)


@torch.no_grad()
def decode_grid(decoder, z: torch.Tensor, resolution: int, bounds: float, device) -> np.ndarray:
    """Sample the decoded field on a cubic grid over ``[-bounds, bounds]``, ready for meshing."""
    lin = np.linspace(-bounds, bounds, resolution)
    xx, yy, zz = np.meshgrid(lin, lin, lin, indexing="ij")
    grid = np.stack([xx, yy, zz], axis=-1).reshape(-1, 3)
    sdf = decode_sdf_at(decoder, z, grid, device)
    return sdf.reshape(resolution, resolution, resolution)


def validate_block(encoder, decoder, cloud_m, canon_extent, device):
    """Encode one part and measure how far the decoded surface is from its real one.

    Args:
        cloud_m: The part's surface points in metres, shape ``(N, 3)``.

    Returns:
        The normalised cloud and the error at each of its points, the size of the code, and the
        mean and worst error both in the canonical frame and in millimetres at the part's own
        size.
    """
    cloud_c, scale, _ = to_canonical(cloud_m, canon_extent)
    z = encode(encoder, cloud_c, device)
    # The field is evaluated at the input points themselves: a faithful reconstruction has its
    # surface passing through them, so the value there is zero. No mesh is needed for this.
    sdf_at_surface = decode_sdf_at(decoder, z, cloud_c, device)
    err = np.abs(sdf_at_surface)
    # Undoing the normalisation, so the error is expressed at the part's real size and is
    # comparable across parts of different sizes.
    inv_mm = 1000.0 / scale
    return {
        "cloud_canon": cloud_c,
        "sdf_err": err,
        "z_norm": float(z.norm().item()),
        "mean_err_norm": float(err.mean()),
        "max_err_norm": float(err.max()),
        "mean_err_mm": float(err.mean() * inv_mm),
        "max_err_mm": float(err.max() * inv_mm),
        "z": z.cpu().numpy(),
    }


def try_build_html(results, blocks, decoder, device, resolution, bounds, out_html):
    """Write an interactive page showing each reconstruction next to the points it was fitted to.

    Purely for inspection, and skipped without failing if the plotting and meshing libraries are
    absent, so that the measured results are produced even in an environment that lacks them.
    """
    try:
        import plotly.graph_objects as go
        from skimage.measure import marching_cubes
    except ImportError as exc:
        print(f"[viz] skipping HTML ({exc}); quantitative results still written.")
        return False

    traces, buttons = [], []
    for i, (res, blk) in enumerate(zip(results, blocks)):
        grid = decode_grid(decoder, torch.from_numpy(res["z"]).to(device), resolution, bounds, device)
        mesh_x = mesh_y = mesh_z = None
        try:
            verts, faces, _, _ = marching_cubes(grid, level=0.0,
                                                spacing=(2 * bounds / (resolution - 1),) * 3)
            verts = verts - bounds
            mesh = go.Mesh3d(x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
                             i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
                             color="#bdbdbd", opacity=0.45, name="recon", visible=(i == 0))
        except (ValueError, RuntimeError):
            # A code the decoder maps to a field of one sign has no surface to extract. That is
            # itself a result worth seeing, so the part is labelled rather than dropped.
            mesh = go.Scatter3d(x=[0], y=[0], z=[0], mode="text",
                                text=["mesh extraction failed (empty iso-surface)"],
                                visible=(i == 0), name="recon")
        pc = res["cloud_canon"]
        pts = go.Scatter3d(
            x=pc[:, 0], y=pc[:, 1], z=pc[:, 2], mode="markers",
            marker=dict(size=1.8, color=res["sdf_err"], colorscale="RdYlBu_r",
                        cmin=0.0, cmax=float(np.percentile(res["sdf_err"], 95)) or 1e-6,
                        colorbar=dict(title="|SDF|", thickness=12, len=0.6)),
            name="input pts", visible=(i == 0))
        traces += [mesh, pts]
        vis = [False] * (2 * len(results))
        vis[2 * i] = vis[2 * i + 1] = True
        buttons.append(dict(
            label=f"{i}:{blk['shape_type']} ({res['mean_err_mm']:.1f}mm)",
            method="update",
            args=[{"visible": vis},
                  {"title": f"block {i} — {blk['shape_type']} "
                            f"local={tuple(round(x, 3) for x in blk['local_size_m'])}m  "
                            f"mean|SDF|={res['mean_err_mm']:.2f}mm max={res['max_err_mm']:.2f}mm"}]))

    fig = go.Figure(data=traces)
    fig.update_layout(
        updatemenus=[dict(active=0, buttons=buttons, x=0.0, y=1.12, xanchor="left")],
        scene=dict(xaxis=dict(range=[-bounds, bounds]), yaxis=dict(range=[-bounds, bounds]),
                   zaxis=dict(range=[-bounds, bounds]), aspectmode="cube"),
        title=f"block 0 — {blocks[0]['shape_type']}", margin=dict(l=0, r=0, t=80, b=0))
    out_html.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out_html))
    print(f"[viz] wrote {out_html}")
    return True


def main() -> int:
    """Validate every part of one assembly and write the summary.

    Returns:
        Zero on success, one if the checkpoint is missing.
    """
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--figure", default=DEFAULT_FIGURE, help="Figure subdir in data/shape_dataset/.")
    p.add_argument("--candidate", default="candidate_04", help="Candidate stem (e.g. candidate_04).")
    p.add_argument("--checkpoint", default=None, help="Override autoencoder checkpoint path.")
    p.add_argument("--canon_extent", type=float, default=0.9,
                   help="Largest normalised extent in the canonical frame (default 0.9).")
    p.add_argument("--resolution", type=int, default=96, help="Marching-cubes grid resolution.")
    p.add_argument("--bounds", type=float, default=1.0, help="SDF grid half-span.")
    p.add_argument("--no_html", action="store_true", help="Skip the interactive HTML.")
    p.add_argument("--out_dir", default=str(BASE_DIR / "results" / "shape_validation"))
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = (Path(args.checkpoint) if args.checkpoint else
            BASE_DIR / config.autoencoder.autoencoder_folder /
            f"best_encoder_decoder_general_latentdim{config.autoencoder.latent_dim}.pth")
    if not ckpt.is_absolute():
        ckpt = BASE_DIR / ckpt
    if not ckpt.exists():
        print(f"ERROR: checkpoint not found: {ckpt}")
        return 1
    encoder, decoder, _ = load_models(ckpt, device)

    fig_dir = BASE_DIR / "data" / "shape_dataset" / args.figure
    meta = json.loads((fig_dir / f"{args.candidate}.json").read_text())
    blocks = meta["blocks"]
    print(f"\nFigure: {args.figure}/{args.candidate}  "
          f"({meta['n_blocks']} blocks, {meta['shape_counts']})\n")

    results = []
    for blk in blocks:
        cloud = np.load(fig_dir / blk["pointcloud_file"]).astype(np.float32)
        results.append(validate_block(encoder, decoder, cloud, args.canon_extent, device))

    hdr = f"{'idx':>3} {'shape':>9} {'local_size (m)':>22} {'||z||':>7} {'mean|SDF|mm':>12} {'max|SDF|mm':>11}"
    print(hdr)
    print("-" * len(hdr))
    for blk, r in zip(blocks, results):
        ls = tuple(round(x, 3) for x in blk["local_size_m"])
        print(f"{blk['index']:>3} {blk['shape_type']:>9} {str(ls):>22} "
              f"{r['z_norm']:>7.3f} {r['mean_err_mm']:>12.3f} {r['max_err_mm']:>11.3f}")

    # Grouped by type, because a failure here is almost always a property of one shape and not
    # of the autoencoder as a whole, and the overall mean would hide it.
    print("\nBy shape type (mean over blocks):")
    by_type: dict[str, list[float]] = {}
    for blk, r in zip(blocks, results):
        by_type.setdefault(blk["shape_type"], []).append(r["mean_err_mm"])
    for st, errs in sorted(by_type.items()):
        print(f"  {st:>9}: n={len(errs):>2}  mean|SDF|={np.mean(errs):7.3f} mm  "
              f"worst={np.max(errs):7.3f} mm")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "figure": args.figure, "candidate": args.candidate, "checkpoint": ckpt.name,
        "canon_extent": args.canon_extent,
        "blocks": [{"index": b["index"], "shape_type": b["shape_type"],
                    "local_size_m": b["local_size_m"], "z_norm": r["z_norm"],
                    "mean_err_mm": r["mean_err_mm"], "max_err_mm": r["max_err_mm"]}
                   for b, r in zip(blocks, results)],
        "by_shape_type": {st: {"n": len(e), "mean_err_mm": float(np.mean(e)),
                               "worst_err_mm": float(np.max(e))} for st, e in by_type.items()},
    }
    out_json = out_dir / f"{args.figure}__{args.candidate}.json"
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nWrote {out_json}")

    if not args.no_html:
        try_build_html(results, blocks, decoder, device, args.resolution, args.bounds,
                       out_dir / f"{args.figure}__{args.candidate}.html")

    return 0


if __name__ == "__main__":
    sys.exit(main())
