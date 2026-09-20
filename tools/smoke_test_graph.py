"""Check that one latent graph can be built end to end, from dataset row to node features.

Loads the dataset, forces the frozen box encoder to load, converts the first row into a
PyTorch Geometric graph and prints its node features, edge list, edge features and targets.
Two assertions guard the substitution this branch is about: the node width must be
``config.autoencoder.latent_dim + 2`` — the code plus the two-entry role indicator — and that
number must agree with ``config.gnn.node_dim``, which is what the surrogate is built with. A
mismatch between the two configuration keys is otherwise only discovered inside a training run.

Writes nothing. Exits with 2 after printing the traceback if anything raises.

Run:
    python tools/smoke_test_graph.py
"""

import sys
import traceback
from pathlib import Path

import torch

# The script is started as a file, not as a module, so the repository root is not on the path
# and `config` and `src` would not resolve.
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

print("python exe:", sys.executable)
print("torch version:", torch.__version__)

try:
    from config.config import config
    from src.gnn_dataset_preparation import (
        build_graph_from_row,
        get_box_encoder,
        prepare_simulation_dataset,
    )

    df, scale = prepare_simulation_dataset()
    print("Dataset rows:", len(df), "scale=", scale)

    # Loaded up front so that a missing or mismatched encoder checkpoint is reported here
    # rather than from inside the graph construction, where it would look like a data problem.
    encoder = get_box_encoder(device="cpu")
    print("Encoder loaded — latent_dim =", encoder.latent_dim)

    row = df.iloc[0]
    graph = build_graph_from_row(row, scale, device="cpu")

    print("Graph built:")
    print(" x.shape =", graph.x.shape, " (expect (2, latent_dim+2))")
    print(" x =\n", graph.x)
    print(" edge_index =", graph.edge_index)
    print(" edge_attr.shape =", graph.edge_attr.shape)
    print(" y.shape =", graph.y.shape, " y =", graph.y)
    print(" y_good =", graph.y_good)

    expected_node_dim = config.autoencoder.latent_dim + 2
    assert graph.x.shape == (2, expected_node_dim), (
        f"node dim mismatch: got {tuple(graph.x.shape)}, expected (2, {expected_node_dim})"
    )
    assert graph.x.shape[1] == config.gnn.node_dim, (
        f"node dim does not match config.gnn.node_dim ({config.gnn.node_dim})"
    )

except Exception:
    print("Exception during graph build:")
    traceback.print_exc()
    sys.exit(2)

print("Single-graph smoke test completed successfully")
