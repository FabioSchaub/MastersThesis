"""Mix the simulated training set with the design-derived one into a single training file.

The two sources cover different things. The simulated set is dense but drawn from its own
sampling distribution; the set built by ``tools/build_finetune_dataset.py`` follows the aspect
ratios that the generative stage actually produces, which the simulated set covers thinly. A
surrogate trained on either alone is weak on the other, so this script draws a stratified
subset of the first, appends all of the second, shuffles, and writes one file that both are
represented in.

Reads the two named files from ``data/`` and writes the mixed set to ``data/`` as well; the
result is what ``config.data.data_file`` points at. Column sets must match, otherwise the
script stops.

Run:
    python tools/build_mixed_dataset.py
    python tools/build_mixed_dataset.py --isaac-samples 100000 --pos-ratio 0.3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"

# All three defaults belong to the 10 mm overlap rule (config.gnn.thresh_overlap_min); the
# fine-tune file has to have been regenerated under the same threshold or the labels disagree.
DEFAULT_ISAAC = "isaac_sim_dataset_20260513_0823.txt"
DEFAULT_FINETUNE = "finetune_dataset_10mm.txt"
DEFAULT_OUTPUT = "mixed_dataset_small_10mm.txt"


def stratified_sample(df: pd.DataFrame, n: int, pos_ratio: float,
                      label_col: str, rng: np.random.Generator) -> pd.DataFrame:
    """Draw ``n`` rows holding the share of feasible ones near ``pos_ratio``, without replacement."""
    n_pos = int(round(n * pos_ratio))
    n_neg = n - n_pos
    pos_idx = df.index[df[label_col] == True].to_numpy()
    neg_idx = df.index[df[label_col] == False].to_numpy()
    if n_pos > len(pos_idx):
        print(f"  WARN: requested {n_pos} positives but only {len(pos_idx)} available "
              f"-> taking all positives, padding with negatives")
        n_pos = len(pos_idx)
        n_neg = n - n_pos
    if n_neg > len(neg_idx):
        print(f"  WARN: requested {n_neg} negatives but only {len(neg_idx)} available "
              f"-> taking all negatives")
        n_neg = len(neg_idx)
    pos_sample = rng.choice(pos_idx, size=n_pos, replace=False)
    neg_sample = rng.choice(neg_idx, size=n_neg, replace=False)
    keep = np.concatenate([pos_sample, neg_sample])
    rng.shuffle(keep)
    return df.loc[keep].reset_index(drop=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--isaac-file", type=str, default=DEFAULT_ISAAC,
                        help="Original isaac dataset filename (in data/)")
    parser.add_argument("--finetune-file", type=str, default=DEFAULT_FINETUNE,
                        help="Finetune dataset filename (in data/)")
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT,
                        help="Output mixed dataset filename (in data/)")
    parser.add_argument("--isaac-samples", type=int, default=100_000,
                        help="Number of rows to sample from the isaac dataset")
    parser.add_argument("--pos-ratio", type=float, default=0.30,
                        help="Target positive (Assembly_Good?=True) ratio in the isaac subset")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    label_col = "Assembly_Good?"

    isaac_path = DATA_DIR / args.isaac_file
    finetune_path = DATA_DIR / args.finetune_file
    output_path = DATA_DIR / args.output

    print(f"Loading isaac: {isaac_path}")
    isaac = pd.read_csv(isaac_path)
    print(f"  shape: {isaac.shape}, pos: "
          f"{(isaac[label_col]==True).sum():,} ({(isaac[label_col]==True).mean()*100:.1f}%)")

    print(f"\nLoading finetune: {finetune_path}")
    finetune = pd.read_csv(finetune_path)
    print(f"  shape: {finetune.shape}, pos: "
          f"{(finetune[label_col]==True).sum():,} ({(finetune[label_col]==True).mean()*100:.1f}%)")

    if list(isaac.columns) != list(finetune.columns):
        print("ERROR: column mismatch between isaac and finetune datasets")
        return 1

    print(f"\nStratified sampling {args.isaac_samples:,} rows from isaac "
          f"with pos_ratio={args.pos_ratio:.2f}")
    isaac_sub = stratified_sample(isaac, args.isaac_samples, args.pos_ratio,
                                  label_col, rng)
    print(f"  isaac subset: {len(isaac_sub):,} rows, pos: "
          f"{(isaac_sub[label_col]==True).sum():,} "
          f"({(isaac_sub[label_col]==True).mean()*100:.1f}%)")

    print(f"\nConcatenating + shuffling")
    mixed = pd.concat([isaac_sub, finetune], ignore_index=True)
    mixed = mixed.sample(frac=1, random_state=args.seed).reset_index(drop=True)
    print(f"  total: {len(mixed):,} rows, pos: "
          f"{(mixed[label_col]==True).sum():,} "
          f"({(mixed[label_col]==True).mean()*100:.1f}%)")

    print(f"\nWriting -> {output_path}")
    mixed.to_csv(output_path, index=False)
    print(f"  done ({output_path.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
