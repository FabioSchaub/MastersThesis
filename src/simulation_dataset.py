"""Reading layer for the labelled design tables, and the conventions their columns follow.

A design table is a comma-separated file with one assembly per row and one group of columns per
block, named by the prefixes ``Block0``, ``Block1`` and so on. This module turns such a file
into a DataFrame, normalises its boolean columns, and derives the two things the rest of the
pipeline needs from the column names alone: which blocks a table contains, and the factor
between the metres it is written in and the range the encoder was trained on.

Four of its functions carry the whole pipeline: ``read_txt_file``, ``manipulate_data``,
``get_block_prefixes`` and ``get_scale_factor`` are what the dataset preparation, the repair and
the diagnostics import. The remaining shape-sampling helpers below are not called anywhere on
this branch; the samples stage 1 trains on come from ``src/enc_dec_dataset_generation.py``.
"""

from pathlib import Path
import pandas as pd
import numpy as np
import trimesh
import sys

# Add parent directory to path to import config
sys.path.insert(0, str(Path(__file__).parent.parent))
from config.config import config


DATA_FOLDER = Path(__file__).parent.parent / config.data.data_folder
DATA_FOLDER.mkdir(exist_ok=True)


def read_txt_file(file_path: Path, file_name: str) -> pd.DataFrame:
    """Read one design table into a DataFrame.

    The extension is ``.txt`` but the contents are comma-separated, which is why the separator
    is given explicitly rather than inferred.

    Returns:
        The table, or an empty DataFrame if the file could not be read. A read failure is
        reported and swallowed rather than raised, so callers must check for emptiness.
    """
    try:
        df = pd.read_csv(
            file_path / file_name, sep=","
        )
        print(f"Successfully read {file_name} with shape {df.shape}")
        return df
    except Exception as e:
        print(f"Error reading {file_path/file_name}: {e}")
        return pd.DataFrame()


def manipulate_data(df: pd.DataFrame) -> pd.DataFrame:
    """Replace every boolean in the table with 1 or 0.

    The feasibility columns are written as ``True`` and ``False``. Converting them once here
    means every consumer can treat them as numbers, and in particular that a label can be cast
    to a tensor without a per-column special case.
    """
    df = df.replace({True: 1, False: 0})
    print("Replaced True/False with 1/0 in the DataFrame")
    return df


def get_block_prefixes(columns) -> list[str]:
    """List the block prefixes a table contains, in block order.

    The number of blocks is not recorded anywhere in the file; it has to be read off the column
    names. Sorting by the trailing integer rather than alphabetically is what makes the result
    the chain order the repair walks, and it is why ten or more blocks would still order
    correctly.

    Args:
        columns: Any iterable of column names, so this accepts a DataFrame's columns as well as
            a single row's index.

    Returns:
        Prefixes such as ``['Block0', 'Block1']``, one per block.
    """
    prefixes: set[str] = set()
    for col in columns:
        if col.startswith("Block") and "_Size" in col:
            prefix = col.split("_Size")[0]
            prefixes.add(prefix)
    return sorted(prefixes, key=lambda p: int(p.replace("Block", "")))


def get_scale_factor(df: pd.DataFrame) -> float:
    """Compute the factor that maps the largest block in the table onto the encoder's range.

    The encoder was trained on half-extents drawn between ``config.data.sampling_min`` and
    ``config.data.sampling_max`` metres, so a table whose blocks are larger than that would be
    encoded outside the range the codes mean anything in. Scaling the whole table by one factor
    keeps every proportion intact while bringing the largest half-extent onto the top of that
    range.

    All blocks are pooled rather than only the first two, so a table of any length gives the
    same factor for every block in it.

    Returns:
        The factor by which a length in metres is multiplied to reach the scaled frame.

    Raises:
        ValueError: If the table has no size columns, or if the largest half-extent is not
            positive. Both mean the table is not a design table and no factor is meaningful.
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

    # The table stores full edge lengths while the encoder is defined on half-extents, so the
    # factor has to be derived from halves or it would be out by two.
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
    """Collect every distinct block size in the table, pooled over all block positions.

    The same size may occur in several rows and at several positions; pooling first and
    deduplicating afterwards means a shape is counted once however often it is used.

    Returns:
        A table with the columns ``SizeX``, ``SizeY`` and ``SizeZ``, full edge lengths in
        metres, one row per distinct size.
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
    """Signed distance from points to an axis-aligned box centred at the origin.

    Args:
        points: Query coordinates, shape ``(N, 3)``.
        half_extents: Half side lengths, shape ``(3,)``, same units as the points.

    Returns:
        Signed distances of shape ``(N,)``, negative inside the box and positive outside.
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
    """Sample query points around one box and evaluate its signed distance at them.

    Half the work is a uniform fill of the cube and half is a band around the surface, because
    the zero level set is what a decoder has to get right and uniform points alone would spend
    most of their budget far from it.

    Args:
        size: Full edge lengths, shape ``(3,)``, in metres.
        n_uniform: Points drawn uniformly in the unit cube.
        n_surface: Points drawn on the surface before noise is added.
        sigma: Standard deviation of the noise that spreads the surface points into a band.

    Returns:
        A tuple ``(points, sdf_values, scale)``: coordinates of shape ``(N, 3)`` and distances
        of shape ``(N,)``, both in the normalised frame, and the divisor in metres that was
        used to reach it.
    """
    half = np.array(size) / 2.0

    # Every shape is divided by the length of its own half-extent vector, so the box always
    # fills a comparable part of the unit cube regardless of how large the real part is. Only
    # the ratio between the three axes survives, and the divisor is returned so the caller can
    # undo it.
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


# Running the module directly reports the scale factor of the configured table and nothing
# else. It writes no files.
if __name__ == "__main__":

    data: pd.DataFrame = read_txt_file(DATA_FOLDER, config.data.data_file)

    data = manipulate_data(data)

    scale_factor = get_scale_factor(data)

    print(f"Scale factor: {scale_factor}")
