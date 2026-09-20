"""Smallest check that the environment is intact and the training dataset can be read.

Prints the interpreter, the torch version and whether CUDA is visible, then imports the
configuration and the graph preparation module and calls ``prepare_simulation_dataset`` on the
file named by ``config.data.data_file``, reporting the row count, the scale factor and the
first columns. No checkpoint is touched, because ``prepare_simulation_dataset`` loads the
frozen box encoder only when a graph is actually built. That is the point of splitting it from
``tools/smoke_test_graph.py``: if this one passes and that one fails, the problem is the
encoder checkpoint and not the installation or the dataset.

Writes nothing. Exits with 2 after printing the traceback if anything raises.

Run:
    python tools/smoke_test_imports.py
"""

import sys, traceback
import torch
from pathlib import Path

print("python exe:", sys.executable)
print("torch version:", torch.__version__)
print("cuda available:", torch.cuda.is_available())

# The script is started as a file, not as a module, so the repository root is not on the path
# and `config` and `src` would not resolve.
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

try:
    from config.config import config
    from src.gnn_dataset_preparation import prepare_simulation_dataset

    print("Imported config and gnn_dataset_preparation OK")

    df, scale = prepare_simulation_dataset()
    print("Loaded df rows:", len(df), "scale=", scale)
    print("First columns:", list(df.columns)[:30])
except Exception:
    print("Exception during import or dataset load:")
    traceback.print_exc()
    sys.exit(2)

print("Smoke import + dataset load completed successfully")
