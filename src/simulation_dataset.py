"""Reading the dataset file, and the shape-sampling helpers that go with it.

Every module that touches the labelled table goes through :func:`read_txt_file` and
:func:`manipulate_data` here, so that a boolean column means the same thing everywhere. The
block prefixes are detected from the column names rather than assumed, which is what lets the
same reader serve a table with two blocks and one with more.

The remaining functions sample points and signed distances on a box. They are not part of the
repair of Part I, where a block is described by three edge lengths and needs no sampled
surface; they are what the latent formulation builds its training data from, and are kept so
that the dataset reader and the sampler stay in one place.

Run directly, the file reads the configured dataset and reports the scale factor:

    python src/simulation_dataset.py
"""

from pathlib import Path
import pandas as pd
import numpy as np
import trimesh
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from config.config import config


DATA_FOLDER = Path(__file__).parent.parent / config.data.data_folder
DATA_FOLDER.mkdir(exist_ok=True)


def read_txt_file(file_path: Path, file_name: str) -> pd.DataFrame:
    """Read the comma-separated dataset file.

    Args:
        file_path: Directory holding the file.
        file_name: Name of the file.

    Returns:
        The table, or an empty one if the file could not be read. Callers that cannot work
        with an empty table check for it, since no exception is raised.
    """
    try:
        df = pd.read_csv(
            file_path / file_name, sep=","
        )
        print(f"Successfully read {file_name} with shape {df.shape}")
        return df
    except Exception as e:
        print(f"Error reading {file_path/file_name}: {e}")
        return pd.DataFrame()  # Return an empty DataFrame in case of error


def manipulate_data(df: pd.DataFrame) -> pd.DataFrame:
    """Map booleans to zero and one so the label columns are numeric everywhere."""
    df = df.replace({True: 1, False: 0})
    print("Replaced True/False with 1/0 in the DataFrame")
    return df


def get_block_prefixes(columns) -> list[str]:
    """Block prefixes present in a set of column names, in block order.

    Detected rather than assumed, so the same reader serves a two-block table and a longer
    one. Accepts anything iterable over strings: table columns or the index of one row.
    """
    prefixes: set[str] = set()
    for col in columns:
        if col.startswith("Block") and "_Size" in col:
            prefix = col.split("_Size")[0]
            prefixes.add(prefix)
    return sorted(prefixes, key=lambda p: int(p.replace("Block", "")))


def get_scale_factor(df: pd.DataFrame) -> float:
    """Factor that maps the largest half-extent in the table onto the sampling range.

    Used by the shape sampling below, where a box has to fit inside the unit cube the decoder
    is defined on. The surrogate of Part I does not go through this: it is fed raw metres.

    Raises:
        ValueError: If no size columns are present, or the largest half-extent is not
            positive.
    """
    prefixes = get_block_prefixes(df.columns)
    if not prefixes:
        raise ValueError("No Block{i}_Size* columns found in DataFrame")

    cols: list[str] = []
    for prefix in prefixes:
        for axis in ("X", "Y", "Z"):
            col = f"{prefix}_Size{axis}"
            if col in df.columns:
                cols.append(col)

    if not cols:
        raise ValueError(f"No block size columns found. Prefixes detected: {prefixes}")

    halves: pd.DataFrame = df[cols].astype(float) / 2.0

    max_half: float = float(halves.max().max())
    if max_half <= 0:
        raise ValueError("Computed max half-extent is non-positive")

    scale_factor: float = float(config.data.sampling_max) / max_half

    print(
        f"Computed scale factor: {scale_factor:.4f} (max half-extent: {max_half:.4f}, "
        f"{len(prefixes)} block type(s): {prefixes})"
    )
    return float(scale_factor)


def get_unique_shapes_combined(df: pd.DataFrame) -> pd.DataFrame:
    """Every distinct triple of edge lengths in the table, pooled over all blocks.

    Returns:
        A table with the columns ``SizeX``, ``SizeY`` and ``SizeZ``, one row per distinct box.
    """
    block_prefixes: list[str] = [
        c.split("_SizeX")[0] for c in df.columns if c.endswith("_SizeX")
    ]

    dfs: list[pd.DataFrame] = []
    for prefix in block_prefixes:
        cols = [f"{prefix}_SizeX", f"{prefix}_SizeY", f"{prefix}_SizeZ"]
        sub = df[cols].copy()
        sub.columns = ["SizeX", "SizeY", "SizeZ"]
        dfs.append(sub)

    all_shapes: pd.DataFrame = pd.concat(dfs).drop_duplicates().reset_index(drop=True)
    print(f"Found {len(block_prefixes)} block types, {len(all_shapes)} unique shapes")
    return all_shapes


def box_sdf(points: np.ndarray, half_extents: np.ndarray) -> np.ndarray:
    """Signed distance to an axis-aligned box centred at the origin.

    Args:
        points: Query points, shape ``(N, 3)``.
        half_extents: Half edge lengths of the box, shape ``(3,)``.

    Returns:
        One distance per point, shape ``(N,)``, negative inside the box.
    """
    q = np.abs(points) - half_extents
    outside = np.linalg.norm(np.maximum(q, 0), axis=1)
    inside = np.minimum(np.max(q, axis=1), 0)
    return outside + inside


def sample_sdf_for_shape(
    size: np.ndarray,
    n_uniform: int = config.data.n_uniform,
    n_surface: int = config.data.n_surface,
    sigma: float = config.data.sigma,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Sample a box and evaluate its signed distance field.

    Two populations are drawn: points spread through the cube, which teach a decoder where the
    shape is not, and points on the surface perturbed by noise, which is where the field has
    to be accurate.

    Args:
        size: Edge lengths of the box, shape ``(3,)``.
        n_uniform: Points drawn uniformly in the cube.
        n_surface: Points drawn on the surface.
        sigma: Standard deviation of the noise added to the surface points.

    Returns:
        The points of shape ``(N, 3)``, their signed distances of shape ``(N,)``, and the
        factor the box was divided by, which is needed to undo the normalisation later.
    """
    half = np.array(size) / 2.0

    # The box is normalised by its own diagonal, so that boxes of very different absolute size
    # are presented at a comparable scale and the field stays numerically well behaved.
    scale = np.linalg.norm(half)
    half_norm = half / scale

    pts_uniform = np.random.uniform(-1, 1, (n_uniform, 3))

    mesh = trimesh.creation.box(
        extents=2 * half_norm
    )
    pts_near, _ = trimesh.sample.sample_surface(mesh, n_surface)
    pts_near += np.random.normal(0, sigma, pts_near.shape)

    points = np.concatenate([pts_uniform, pts_near])
    sdf_vals = box_sdf(points, half_norm)
    return points.astype(np.float32), sdf_vals.astype(np.float32), scale


if __name__ == "__main__":

    data: pd.DataFrame = read_txt_file(DATA_FOLDER, config.data.data_file)

    data = manipulate_data(data)

    scale_factor = get_scale_factor(data)

    print(f"Scale factor: {scale_factor}")
