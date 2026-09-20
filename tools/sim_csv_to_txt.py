"""Reduce the simulator's output to the columns the surrogate needs.

The file the simulator writes carries far more per configuration than the surrogate reads, and
it has to be moved to the cluster. This keeps the names, the sizes, the positions and the label
flags, drops every row naming a shape that has no code, and writes one row per configuration
separated by spaces. The flags are reduced to zero and one here, so nothing downstream has to
know how the simulator spelled them.

The file keeps a header line naming its columns, so it stays readable on its own and the reader
need not assume an order.

Run:
    python -m tools.sim_csv_to_txt <input csv> [--out <output txt>] [--latents <code table>]

Writes:
    the reduced file, next to the input with a ``.txt`` suffix unless told otherwise.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch

# The order defines the file. The five label columns come last, which the conversion relies on
# when it decides which values are flags; the fourth of them, the gap, is carried through even
# though the surrogate does not learn it, so that the file does not lose information.
_COLS = (
    ["Obj0_UsdName", "Obj0_SizeX", "Obj0_SizeY", "Obj0_SizeZ"]
    + ["Obj1_UsdName", "Obj1_SizeX", "Obj1_SizeY", "Obj1_SizeZ"]
    + ["Obj0_PosX", "Obj0_PosY", "Obj0_PosZ", "Obj1_PosX", "Obj1_PosY", "Obj1_PosZ"]
    + ["ObjNew_OBJECT_GAP", "ObjNew_OVERLAP_INSUFFICIENT",
       "ObjNew_THICKNESS_EXCEEDED", "ObjNew_TIPPING", "Assembly_Good?"]
)


def _b(v: str) -> str:
    """Reduce a flag the simulator may write as a word or a number to a single character."""
    return "1" if str(v).strip().lower() in ("true", "1", "1.0") else "0"


def main() -> None:
    """Convert one file, reporting how many rows were read, skipped and written."""
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--out", default=None)
    ap.add_argument("--latents", default="encoder_decoder_model/shape_latents_lv_v14_lam005.pt")
    args = ap.parse_args()

    # Only the names are needed: the codes themselves are looked up when the graphs are built.
    keep = set(torch.load(args.latents, map_location="cpu", weights_only=False).keys())
    out = Path(args.out) if args.out else Path(args.csv).with_suffix(".txt")

    # The simulator may prefix its file with a separator declaration, which is not a row.
    lines = Path(args.csv).read_text(encoding="utf-8-sig").splitlines()
    start = 1 if lines and lines[0].lower().startswith("sep=") else 0
    rdr = csv.reader(lines[start:])
    header = next(rdr)
    ci = {n: i for i, n in enumerate(header)}
    flags = _COLS[14:]

    n_in = n_out = n_skip = 0
    with open(out, "w", newline="") as f:
        f.write(" ".join(_COLS) + "\n")
        for row in rdr:
            n_in += 1
            if row[ci["Obj0_UsdName"]] not in keep or row[ci["Obj1_UsdName"]] not in keep:
                n_skip += 1
                continue
            vals = []
            for cname in _COLS:
                v = row[ci[cname]]
                vals.append(_b(v) if cname in flags else v)
            f.write(" ".join(vals) + "\n")
            n_out += 1

    print(f"read {n_in} rows, skipped {n_skip} (prisms), wrote {n_out} -> {out}")
    print(f"columns ({len(_COLS)}): {' '.join(_COLS)}")


if __name__ == "__main__":
    main()
