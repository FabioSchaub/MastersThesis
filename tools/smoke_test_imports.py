"""Check that the environment and the dataset are in place, before anything long is started.

Reports the interpreter, the version of the tensor library and whether a device is available,
then imports the configuration and loads the dataset it points at. Failing here is cheap;
failing an hour into a training run because a file is missing or a package resolved to the
wrong build is not.

    python tools/smoke_test_imports.py

Prints what it found and exits non-zero if the import or the load failed.
"""

import sys, traceback
import torch
from pathlib import Path

print("python exe:", sys.executable)
print("torch version:", torch.__version__)
print("cuda available:", torch.cuda.is_available())

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
