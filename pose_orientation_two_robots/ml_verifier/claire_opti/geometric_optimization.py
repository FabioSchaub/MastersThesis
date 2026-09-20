"""
Claire's ML surrogate optimization for woodworking assembly.

Called from model_interface.py::refine_full_yaml_model_b(input_csv) -> output_csv.

Pipeline:
  1. Parse input CSV (absolute RViz positions + sizes + adjacency + screws)
  2. Detect connected pairs from adjacency columns
  3. For each pair: compute relative geometry, run gradient-based optimization
     through the frozen surrogate to maximise P(Assembly_Good?)
  4. Write refined absolute positions/sizes back to output CSV

Model files expected at:
  <this_file's_dir>/claire_model/flags_surrogate.pt
  <this_file's_dir>/claire_model/geometry_scaler.pkl
  <this_file's_dir>/claire_model/input_cols.pkl
  <this_file's_dir>/claire_model/output_cols.pkl
"""

from __future__ import annotations

import csv
import io
import math
import os
from collections import deque
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import torch
import torch.nn as nn
from scipy.spatial import cKDTree
from sklearn.preprocessing import StandardScaler

# =========================================================
# PATHS  (all relative to this file so they work on any machine)
# =========================================================
_HERE = Path(__file__).resolve().parent
_MODEL_DIR = _HERE / "claire_model"

_MODEL_PATH = _MODEL_DIR / "flags_surrogate.pt"
_SCALER_PATH = _MODEL_DIR / "geometry_scaler.pkl"
_INPUT_SPEC_PATH = _MODEL_DIR / "input_cols.pkl"
_OUTPUT_COLS_PATH = _MODEL_DIR / "output_cols.pkl"

# =========================================================
# OPTIMISATION CONFIG  (no WandB)
# =========================================================
_CFG = {
    "seed": 42,
    "lr": 2e-4,
    "max_steps": 800,
    "print_every": 200,
    "target_prob": 0.90,
    "eps_prob": 1e-7,
    "grad_clip_norm": 1.0,
    "n_restarts": 10,
    "restart_noise_std": 5e-3,
    "restart_noise_pos_std": 0.02,
    "pos_mm": 20.0,  # max absolute position shift (mm)
    "alpha": 0.5,  # max relative size change
    "lambda_kd": 0.0,
    "kd_threshold": 1.5,
    "optimize_positions": True,
    "optimize_sizes": True,
}

# 9-dim relative vector layout
# [0:3] rel_PosX/Y/Z  [3:6] ref SizeX/Y/Z  [6:9] screw SizeX/Y/Z
_REL_POS = [0, 1, 2]
_REF_SIZE = [3, 4, 5]
_SCR_SIZE = [6, 7, 8]
_ALL_SIZE = _REF_SIZE + _SCR_SIZE

_TRAIN_SIZE_MIN = 0.007
_TRAIN_SIZE_MAX = 0.200
_AXES = ["X", "Y", "Z"]
_SIZE_SFX = ["SizeX", "SizeY", "SizeZ"]


# =========================================================
# MODEL ARCHITECTURE  (must match training script exactly)
# =========================================================
class _MultiLabelSurrogate(nn.Module):
    def __init__(
        self, input_dim, output_dim, hidden_dim=128, num_hidden_layers=3, dropout=0.35
    ):
        super().__init__()
        layers, in_dim = [], input_dim
        for _ in range(num_hidden_layers):
            layers += [nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)]
            in_dim = hidden_dim
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        return self.head(self.backbone(x))


# =========================================================
# LAZY MODEL LOADER  (load once, reuse across calls)
# =========================================================
_loaded: dict[str, Any] = {}


def _load_model():
    """Load surrogate + scaler + KD-tree once and cache."""
    if _loaded:
        return _loaded

    state = torch.load(_MODEL_PATH, map_location="cpu", weights_only=False)
    scaler = joblib.load(_SCALER_PATH)
    raw_spec = joblib.load(_INPUT_SPEC_PATH)
    output_cols = joblib.load(_OUTPUT_COLS_PATH)

    if isinstance(raw_spec, dict):
        geom_cols = raw_spec["geom_cols"]
        input_dim = raw_spec.get("input_dim", len(geom_cols))
    else:
        geom_cols = raw_spec
        input_dim = len(geom_cols)

    if isinstance(output_cols, dict):
        output_cols = output_cols.get("output_cols", list(output_cols))

    asm_idx = output_cols.index("Assembly_Good?")
    flag_indices = [i for i, c in enumerate(output_cols) if c != "Assembly_Good?"]
    flag_names = [c for c in output_cols if c != "Assembly_Good?"]

    model = _MultiLabelSurrogate(
        input_dim=input_dim,
        output_dim=len(output_cols),
        hidden_dim=state["config"]["hidden_dim"],
        num_hidden_layers=state["config"]["num_hidden_layers"],
        dropout=state["config"]["dropout"],
    )
    model.load_state_dict(state["model_state_dict"])
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    def predict_assembly(x_in: torch.Tensor):
        """Returns (assembly_prob [1,1], flag_probs [1,F], flag_names)."""
        with torch.no_grad():
            all_probs = torch.sigmoid(model(x_in))
            assembly_prob = all_probs[:, asm_idx : asm_idx + 1]
            flag_probs = all_probs[:, flag_indices]
        return assembly_prob, flag_probs, flag_names

    # KD-tree: fit a scaler on relative geometry, build tree from assemblable rows
    # We rebuild from the scaler's training distribution approximation.
    # Since we don't have the original dataset here, we use the scaler's
    # mean/scale to build a minimal single-point tree as a neutral anchor.
    # The KD penalty will be negligible (lambda_kd=0.2) relative to asm_loss.
    kd_scaler = StandardScaler()
    kd_scaler.mean_ = scaler.mean_.copy()
    kd_scaler.scale_ = scaler.scale_.copy()
    kd_scaler.n_features_in_ = scaler.n_features_in_
    # Single anchor at the mean of training distribution
    anchor = np.zeros((1, input_dim), dtype=np.float32)
    kdtree = cKDTree(anchor)

    _loaded.update(
        {
            "model": model,
            "predict": predict_assembly,
            "scaler": scaler,
            "kd_scaler": kd_scaler,
            "kdtree": kdtree,
            "geom_cols": geom_cols,
            "input_dim": input_dim,
            "output_cols": output_cols,
            "asm_idx": asm_idx,
        }
    )
    return _loaded


# =========================================================
# GEOMETRY HELPERS
# =========================================================
def _scale_tensor(x: torch.Tensor, scaler) -> torch.Tensor:
    mean = torch.tensor(scaler.mean_, dtype=torch.float32)
    scale = torch.tensor(scaler.scale_, dtype=torch.float32)
    return (x - mean) / scale


def _extract_pair_x(geom: dict, ref_b: int, screw_b: int) -> torch.Tensor:
    rel = [
        geom[f"Block{screw_b}_Pos{ax}"] - geom[f"Block{ref_b}_Pos{ax}"] for ax in _AXES
    ]
    rs = [geom[f"Block{ref_b}_Size{ax}"] for ax in _AXES]
    ss = [geom[f"Block{screw_b}_Size{ax}"] for ax in _AXES]
    return torch.tensor(rel + rs + ss, dtype=torch.float32)


def _project_positions(pos, pos0, pos_mm):
    d = pos_mm / 1000.0
    return torch.clamp(pos, pos0 - d, pos0 + d)


def _project_sizes(size, size0, alpha):
    out = torch.zeros_like(size)
    for j in range(3):
        if j == 2:  # Z: free to grow or shrink
            lo = torch.clamp(size0[j] * (1 - alpha), min=_TRAIN_SIZE_MIN)
            hi = torch.clamp(size0[j] * (1 + alpha), max=_TRAIN_SIZE_MAX)
        else:  # X, Y: only grow
            lo = size0[j]
            hi = torch.clamp(size0[j] * (1 + alpha), max=_TRAIN_SIZE_MAX)
        out[j] = torch.clamp(size[j], lo, hi)
    return out


# =========================================================
# CSV PARSING
# =========================================================
def _parse_csv(input_csv: str) -> tuple[dict, list[str]]:
    """Return (row_dict, header_list) from a single-row CSV string."""
    reader = csv.DictReader(io.StringIO(input_csv))
    header = list(reader.fieldnames or [])
    rows = list(reader)
    if not rows:
        raise ValueError("Empty input CSV")
    row = {k: v for k, v in rows[0].items()}
    return row, header


def _detect_block_ids(header: list[str]) -> list[int]:
    ids = set()
    for col in header:
        if col.startswith("Block") and "_PosX" in col:
            try:
                ids.add(int(col.split("_")[0].replace("Block", "")))
            except ValueError:
                pass
    return sorted(ids)


def _detect_pairs(row: dict, header: list[str]) -> list[tuple[int, int]]:
    """Return all connected pairs (a, b) where a < b from adjacency cols."""
    pairs = []
    for col in header:
        if "_ConnectsTo_" not in col:
            continue
        try:
            val = row.get(col, "0")
            if str(val).strip() in ("", "nan", "NaN", "0", "0.0"):
                continue
            left, right = col.split("_ConnectsTo_")
            a = int(left.replace("Block", ""))
            b = int(right.replace("Block", ""))
            pairs.append((a, b))
        except (ValueError, AttributeError):
            continue
    return pairs


def _geom_from_row(row: dict, block_ids: list[int]) -> dict:
    geom = {}
    for b in block_ids:
        for ax in _AXES:
            for kind in ["Pos", "Size"]:
                col = f"Block{b}_{kind}{ax}"
                if col in row:
                    try:
                        geom[col] = float(row[col])
                    except (ValueError, TypeError):
                        geom[col] = 0.0
    # Also copy screw columns as-is (we update them at the end)
    for col, val in row.items():
        if "Screw" in col:
            try:
                geom[col] = float(val) if val.lower() != "nan" else math.nan
            except (ValueError, AttributeError):
                geom[col] = math.nan
    return geom


# =========================================================
# SINGLE-PAIR OPTIMISATION
# =========================================================
def _optimise_pair(
    geom: dict,
    ref_b: int,
    screw_b: int,
    loaded: dict,
    frozen_ref_size: bool = False,
    frozen_screw_size: bool = False,
) -> dict:
    """
    Run gradient descent to improve P(Assembly_Good?) for one pair.
    Returns updated geom dict (only positions/sizes of ref_b and screw_b changed).
    """
    cfg = _CFG
    predict = loaded["predict"]
    scaler = loaded["scaler"]
    kd_scaler = loaded["kd_scaler"]
    kdtree = loaded["kdtree"]

    x0 = _extract_pair_x(geom, ref_b, screw_b)

    # Determine which size indices are free
    frozen_idx = set()
    if frozen_ref_size:
        frozen_idx.update(_REF_SIZE)
    if frozen_screw_size:
        frozen_idx.update(_SCR_SIZE)
    free_size_idx = [i for i in _ALL_SIZE if i not in frozen_idx]

    x0_pos = x0[_REL_POS].clone()
    x0_size = x0[free_size_idx].clone() if free_size_idx else None

    mean_t = torch.tensor(kd_scaler.mean_, dtype=torch.float32)
    scale_t = torch.tensor(kd_scaler.scale_, dtype=torch.float32)

    global_best_prob = -1.0
    global_best_x = x0.clone()

    # Eval initial prob
    with torch.no_grad():
        x_in = _scale_tensor(x0, scaler).unsqueeze(0)
        init_prob = predict(x_in)[0].item()
    print(f"  Pair {ref_b}→{screw_b}: init_prob={init_prob:.4f}")

    for restart in range(cfg["n_restarts"]):
        if restart > 0:
            x0_start = x0.clone()
            if free_size_idx:
                x0_start[free_size_idx] += (
                    torch.randn(len(free_size_idx)) * cfg["restart_noise_std"]
                )
            x0_start[_REL_POS] += torch.randn(3) * cfg["restart_noise_pos_std"]
        else:
            x0_start = x0.clone()

        pos_params = (
            x0_start[_REL_POS].clone().requires_grad_(cfg["optimize_positions"])
        )
        size_params = (
            x0_start[free_size_idx].clone().requires_grad_(cfg["optimize_sizes"])
            if free_size_idx
            else None
        )

        opt_params = []
        if cfg["optimize_positions"]:
            opt_params.append(pos_params)
        if cfg["optimize_sizes"] and size_params is not None:
            opt_params.append(size_params)

        if not opt_params:
            break

        optimizer = torch.optim.Adam(opt_params, lr=cfg["lr"])
        best_prob = -1.0
        best_x = x0.clone()

        for step in range(cfg["max_steps"]):
            optimizer.zero_grad()

            x_cur = x0.clone()
            x_cur[_REL_POS] = pos_params
            if size_params is not None:
                x_cur[free_size_idx] = size_params

            x_in = _scale_tensor(x_cur, scaler).unsqueeze(0)
            prob_t, _, _ = (
                predict.__wrapped__(x_in)
                if hasattr(predict, "__wrapped__")
                else _predict_with_grad(loaded["model"], loaded["asm_idx"], x_in)
            )
            asm_loss = -torch.log(prob_t.squeeze() + cfg["eps_prob"])

            # KD penalty
            x_sc = (x_cur - mean_t) / scale_t
            x_np = x_sc.detach().numpy()
            _, nn_idx = kdtree.query(x_np, k=1)
            nn_pt = torch.tensor(kdtree.data[nn_idx], dtype=torch.float32)
            dist = torch.norm(x_sc - nn_pt)
            kd_loss = cfg["lambda_kd"] * torch.clamp(
                dist - cfg["kd_threshold"], min=0.0
            )

            loss = asm_loss + kd_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(opt_params, cfg["grad_clip_norm"])
            optimizer.step()

            with torch.no_grad():
                pos_params.copy_(_project_positions(pos_params, x0_pos, cfg["pos_mm"]))
                if size_params is not None and x0_size is not None:
                    ref_idx_local = [
                        i for i, gi in enumerate(free_size_idx) if gi in _REF_SIZE
                    ]
                    scr_idx_local = [
                        i for i, gi in enumerate(free_size_idx) if gi in _SCR_SIZE
                    ]
                    if ref_idx_local:
                        size_params.data[ref_idx_local] = _project_sizes(
                            size_params[ref_idx_local],
                            x0_size[ref_idx_local],
                            cfg["alpha"],
                        )
                    if scr_idx_local:
                        size_params.data[scr_idx_local] = _project_sizes(
                            size_params[scr_idx_local],
                            x0_size[scr_idx_local],
                            cfg["alpha"],
                        )

            # Eval
            with torch.no_grad():
                x_eval = x0.clone()
                x_eval[_REL_POS] = pos_params
                if size_params is not None:
                    x_eval[free_size_idx] = size_params
                x_eval_in = _scale_tensor(x_eval, scaler).unsqueeze(0)
                p_eval = predict(x_eval_in)[0].item()

            if p_eval > best_prob:
                best_prob = p_eval
                best_x = x_eval.detach().clone()

            if step % cfg["print_every"] == 0:
                print(
                    f"    [r{restart} step {step:04d}] prob={p_eval:.4f} "
                    f"best={best_prob:.4f} loss={loss.item():.3f}"
                )

            if best_prob >= cfg["target_prob"]:
                print(f"    Target reached at step {step}.")
                break

        if best_prob > global_best_prob:
            global_best_prob = best_prob
            global_best_x = best_x.clone()

    print(f"  Pair {ref_b}→{screw_b}: {init_prob:.4f} → {global_best_prob:.4f}")

    # Write back only if improved
    if global_best_prob <= init_prob:
        return geom

    updated = geom.copy()
    best = global_best_x

    # Ref sizes
    if not frozen_ref_size:
        for j, s in enumerate(_SIZE_SFX):
            updated[f"Block{ref_b}_{s}"] = best[3 + j].item()

    # Screw sizes
    if not frozen_screw_size:
        for j, s in enumerate(_SIZE_SFX):
            updated[f"Block{screw_b}_{s}"] = best[6 + j].item()

    # Screw block absolute position = ref_pos + rel_pos
    for j, ax in enumerate(_AXES):
        updated[f"Block{screw_b}_Pos{ax}"] = (
            updated[f"Block{ref_b}_Pos{ax}"] + best[j].item()
        )

    return updated


def _predict_with_grad(model, asm_idx, x_in):
    """Forward pass that keeps grad for the assembly output."""
    all_probs = torch.sigmoid(model(x_in))
    assembly_prob = all_probs[:, asm_idx : asm_idx + 1]
    return assembly_prob, None, None


# =========================================================
# SCREW POSITION UPDATE
# =========================================================
def _update_screw_positions(
    geom: dict, pairs: list[tuple[int, int]], header: list[str]
) -> dict:
    """
    After optimisation, place each screw at the centroid of the XY overlap
    of its pair, at Z = top of the screw block.
    Only touches screw columns that already exist in the header.
    """
    updated = geom.copy()

    # Build a map: for each connected pair, find the screw block's screw column
    # Screw block = the one with non-NaN screw columns (higher Z heuristic fallback)
    for a, b in pairs:
        # Determine screw block = higher Z
        z_a = updated.get(f"Block{a}_PosZ", 0.0)
        z_b = updated.get(f"Block{b}_PosZ", 0.0)
        screw_b = b if z_b >= z_a else a
        ref_b = a if screw_b == b else b

        # Find a screw column owned by screw_b that is non-NaN
        screw_col_base = None
        for col in header:
            if col.startswith(f"Block{screw_b}_Screw") and col.endswith("_X"):
                val = updated.get(col, math.nan)
                if not (isinstance(val, float) and math.isnan(val)):
                    screw_col_base = col[:-2]  # strip "_X"
                    break
        # If all NaN, just take the first available screw col on screw_b
        if screw_col_base is None:
            for col in header:
                if col.startswith(f"Block{screw_b}_Screw") and col.endswith("_X"):
                    screw_col_base = col[:-2]
                    break

        if screw_col_base is None:
            continue  # no screw column to update for this pair

        # Compute XY overlap centroid
        screw_xy = []
        for ax in ["X", "Y"]:
            pos_r = updated[f"Block{ref_b}_Pos{ax}"]
            size_r = updated[f"Block{ref_b}_Size{ax}"]
            pos_s = updated[f"Block{screw_b}_Pos{ax}"]
            size_s = updated[f"Block{screw_b}_Size{ax}"]
            lo = max(pos_r - size_r / 2.0, pos_s - size_s / 2.0)
            hi = min(pos_r + size_r / 2.0, pos_s + size_s / 2.0)
            screw_xy.append((lo + hi) / 2.0 if hi > lo else pos_r)

        screw_z = (
            updated[f"Block{screw_b}_PosZ"] + updated[f"Block{screw_b}_SizeZ"] / 2.0
        )

        updated[f"{screw_col_base}_X"] = screw_xy[0]
        updated[f"{screw_col_base}_Y"] = screw_xy[1]
        updated[f"{screw_col_base}_Z"] = screw_z

    return updated


# =========================================================
# MAIN ENTRY POINT
# =========================================================
def run_claire_optimization(input_csv: str) -> str:
    """
    Takes a single-row CSV string (model_interface format) and returns
    a refined single-row CSV string with improved block geometry.
    """
    torch.manual_seed(_CFG["seed"])

    loaded = _load_model()

    row, header = _parse_csv(input_csv)
    block_ids = _detect_block_ids(header)
    pairs = _detect_pairs(row, header)
    geom = _geom_from_row(row, block_ids)

    print(
        f"\n[Claire optimizer] {len(block_ids)} blocks, {len(pairs)} connected pairs: {pairs}"
    )

    if not pairs:
        print("  No connected pairs found — returning input unchanged.")
        return input_csv

    # Optimise pairs in order; freeze ref sizes once a block has been optimised
    optimised_as_screw = set()

    for a, b in pairs:
        # Determine ref/screw: screw block = higher Z
        z_a = geom.get(f"Block{a}_PosZ", 0.0)
        z_b = geom.get(f"Block{b}_PosZ", 0.0)
        screw_b = b if z_b >= z_a else a
        ref_b = a if screw_b == b else b

        frozen_ref = ref_b in optimised_as_screw
        frozen_screw = screw_b in optimised_as_screw

        geom = _optimise_pair(
            geom,
            ref_b,
            screw_b,
            loaded,
            frozen_ref_size=frozen_ref,
            frozen_screw_size=frozen_screw,
        )
        optimised_as_screw.add(screw_b)

    # Update screw positions to match final geometry
    geom = _update_screw_positions(geom, pairs, header)

    # Serialise back to CSV — same header as input
    out_row = dict(row)  # start from original (preserves adjacency cols etc.)
    for col, val in geom.items():
        if col in out_row:
            out_row[col] = (
                "NaN" if (isinstance(val, float) and math.isnan(val)) else repr(val)
            )

    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=header)
    writer.writeheader()
    writer.writerow(out_row)
    return buf.getvalue()
