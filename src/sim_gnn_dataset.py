"""Turn the labelled configurations of the simulator into the graphs the surrogate reads.

One row of the simulator's output is one configuration of two parts, and it becomes a graph of
two nodes joined in both directions:

    node    the latent code of the part, its measured bounding box in metres, and a two-entry
            indicator of its role. Node 0 is the frozen base, node 1 is the part under test,
            and that order is what the read-out of the surrogate depends on.
    edge    the offset from one centre to the other, and the three distances in the coordinate
            planes. In metres, and antisymmetric in the offset, symmetric in the distances.
    label   the three failure modes, and separately the overall verdict.

The code of a part is looked up by name rather than computed. The encoder is queried in the
canonical frame, so it returns the same code for every instance of a shape whatever its size;
the size is carried by the bounding box instead. A row naming a shape that has no entry in the
lookup is skipped rather than guessed at.

Four flags are recorded by the simulator but only three are used. The fourth, which reports a
gap between the parts, is not a function of the positions and sizes in the file: the simulator
derives it from state that is not exported, so no model reading these columns could predict it.

Two readers are provided for the same content, one for the simulator's own comma-separated
output and one for the reduced whitespace file that is what is actually copied to the cluster.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Data

_TARGET_FLAGS = [
    "ObjNew_OVERLAP_INSUFFICIENT",
    "ObjNew_THICKNESS_EXCEEDED",
    "ObjNew_TIPPING",
]


def load_shape_latents(paths) -> dict[str, torch.Tensor]:
    """Load the table that maps the name of a mesh to its latent code.

    Several tables can be given and are merged in order, a later one overriding an earlier one
    on a shared name. That is how the fourteen base shapes and the perturbed variants derived
    from them are made available together: each mesh the simulator can spawn, whether it is a
    base shape or a variant of one, has to resolve to its own code.

    Args:
        paths: One path, or several to be merged.

    Returns:
        The codes by mesh name, each of shape ``(latent_dim,)``.
    """
    if isinstance(paths, (str, Path)):
        paths = [paths]
    lut: dict[str, torch.Tensor] = {}
    for p in paths:
        d = torch.load(Path(p), map_location="cpu", weights_only=False)
        lut.update({k: v.float() for k, v in d.items()})
    return lut


def _edge_features(p_src: np.ndarray, p_dst: np.ndarray) -> list[float]:
    """Offset from source to destination in metres, then the distance in each coordinate plane."""
    d = np.asarray(p_dst, float) - np.asarray(p_src, float)
    return [
        float(d[0]), float(d[1]), float(d[2]),
        float(np.hypot(d[0], d[1])),
        float(np.hypot(d[0], d[2])),
        float(np.hypot(d[1], d[2])),
    ]


def _truthy(v: str) -> float:
    """Read a flag that the simulator may write as a word or as a number."""
    return 1.0 if str(v).strip().lower() in ("true", "1", "1.0") else 0.0


def csv_to_graphs(
    csv_path: str | Path,
    latents: dict[str, torch.Tensor],
    limit: int | None = None,
) -> tuple[list[Data], dict]:
    """Read the simulator's comma-separated output into graphs.

    Args:
        latents: The code table; rows naming a mesh that is absent from it are skipped.
        limit: Stop after this many rows have been read, skipped rows included.

    Returns:
        The graphs, and a count of rows read, rows skipped and graphs kept.
    """
    # The simulator may prefix its file with a separator declaration, which is not a row.
    lines = Path(csv_path).read_text(encoding="utf-8-sig").splitlines()
    start = 1 if lines and lines[0].lower().startswith("sep=") else 0
    rdr = csv.reader(lines[start:])
    header = next(rdr)
    col = {n: i for i, n in enumerate(header)}

    def g(row, name):
        """Value of a column by name, so the order of the columns need not be assumed."""
        return row[col[name]]

    graphs: list[Data] = []
    stats = {"rows": 0, "skipped_prism": 0, "kept": 0}
    for row in rdr:
        stats["rows"] += 1
        if limit is not None and stats["rows"] > limit:
            stats["rows"] -= 1
            break
        n0, n1 = g(row, "Obj0_UsdName"), g(row, "Obj1_UsdName")
        if n0 not in latents or n1 not in latents:
            stats["skipped_prism"] += 1
            continue

        z0, z1 = latents[n0], latents[n1]
        bbox0 = torch.tensor([float(g(row, f"Obj0_Size{a}")) for a in "XYZ"])
        bbox1 = torch.tensor([float(g(row, f"Obj1_Size{a}")) for a in "XYZ"])
        t_base = torch.tensor([1.0, 0.0])
        t_new = torch.tensor([0.0, 1.0])
        # The order is load-bearing: the surrogate reads its prediction off node 1, and the
        # labels of the row describe that part.
        x = torch.stack([
            torch.cat([z0, bbox0, t_base]),
            torch.cat([z1, bbox1, t_new]),
        ])

        p0 = [float(g(row, f"Obj0_Pos{a}")) for a in "XYZ"]
        p1 = [float(g(row, f"Obj1_Pos{a}")) for a in "XYZ"]
        edge_attr = torch.tensor([_edge_features(p0, p1), _edge_features(p1, p0)])
        edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)

        y = torch.tensor([[_truthy(g(row, f)) for f in _TARGET_FLAGS]])
        y_good = torch.tensor([[_truthy(g(row, "Assembly_Good?"))]])

        graphs.append(Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, y_good=y_good))
        stats["kept"] += 1

    return graphs, stats


def _build_graph(n0, s0, n1, s1, p0, p1, labels, good, latents) -> Data:
    """Assemble one graph from already parsed fields; node 0 is the base, node 1 the part under
    test.
    """
    z0, z1 = latents[n0], latents[n1]
    x = torch.stack([
        torch.cat([z0, torch.tensor(s0), torch.tensor([1.0, 0.0])]),
        torch.cat([z1, torch.tensor(s1), torch.tensor([0.0, 1.0])]),
    ])
    edge_attr = torch.tensor([_edge_features(p0, p1), _edge_features(p1, p0)])
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    return Data(
        x=x, edge_index=edge_index, edge_attr=edge_attr,
        y=torch.tensor([labels], dtype=torch.float),
        y_good=torch.tensor([[float(good)]], dtype=torch.float),
    )


def txt_to_graphs(
    txt_path: str | Path,
    latents: dict[str, torch.Tensor],
    limit: int | None = None,
) -> tuple[list[Data], dict]:
    """Read the reduced whitespace file into graphs; this is the reader training uses.

    Same content and same conventions as :func:`csv_to_graphs`, but the flags have already been
    reduced to zero and one by the conversion, so they are read as numbers.
    """
    graphs: list[Data] = []
    stats = {"rows": 0, "skipped": 0, "kept": 0}
    with open(txt_path) as f:
        header = f.readline().split()
        ci = {n: i for i, n in enumerate(header)}
        for line in f:
            stats["rows"] += 1
            if limit is not None and stats["rows"] > limit:
                stats["rows"] -= 1
                break
            r = line.split()
            n0, n1 = r[ci["Obj0_UsdName"]], r[ci["Obj1_UsdName"]]
            if n0 not in latents or n1 not in latents:
                stats["skipped"] += 1
                continue
            s0 = [float(r[ci[f"Obj0_Size{a}"]]) for a in "XYZ"]
            s1 = [float(r[ci[f"Obj1_Size{a}"]]) for a in "XYZ"]
            p0 = [float(r[ci[f"Obj0_Pos{a}"]]) for a in "XYZ"]
            p1 = [float(r[ci[f"Obj1_Pos{a}"]]) for a in "XYZ"]
            labels = [float(r[ci[f]]) for f in _TARGET_FLAGS]
            good = float(r[ci["Assembly_Good?"]])
            graphs.append(_build_graph(n0, s0, n1, s1, p0, p1, labels, good, latents))
            stats["kept"] += 1
    return graphs, stats


def node_dim(latents: dict[str, torch.Tensor]) -> int:
    """Width of a node feature: the latent code, the three box entries and the role indicator.

    Derived from the code table rather than fixed, so that changing the width of the code does
    not require the surrogate to be edited.
    """
    return next(iter(latents.values())).shape[0] + 3 + 2


if __name__ == "__main__":
    # Run directly to inspect a dataset file: it reports how many rows survived the lookup, the
    # width of a node, and the rate of each label, which is what sets the class weights later.
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--latents", default="encoder_decoder_model/shape_latents_lv_v14_lam005.pt")
    ap.add_argument("--limit", type=int, default=8000)
    args = ap.parse_args()

    lut = load_shape_latents(args.latents)
    graphs, stats = csv_to_graphs(args.csv, lut, limit=args.limit)
    print(f"stats: {stats}")
    print(f"node_dim: {node_dim(lut)}  (z {node_dim(lut) - 5} + bbox 3 + type 2)")
    if graphs:
        d = graphs[0]
        print(f"graph0: x{tuple(d.x.shape)} edge_attr{tuple(d.edge_attr.shape)} y{tuple(d.y.shape)} y_good{tuple(d.y_good.shape)}")
        Y = torch.cat([g.y for g in graphs])
        good = torch.cat([g.y_good for g in graphs])
        print(f"label rates over {len(graphs)} graphs:")
        for i, f in enumerate(_TARGET_FLAGS):
            print(f"  {f.replace('ObjNew_',''):<22} {Y[:, i].mean() * 100:5.1f}%")
        print(f"  Assembly_Good           {good.mean() * 100:5.1f}%")
