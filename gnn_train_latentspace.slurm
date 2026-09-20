#!/bin/bash
#SBATCH --job-name=gnn_latent
#SBATCH --output=logs/gnn_latentspace_%j.log
#SBATCH --error=logs/gnn_latentspace_%j.log
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --gres=gpu:1
#SBATCH --gpus-per-node=1

# ============================================================================
# GNN Training on Euler — latent-space branch
#
# Same architecture/hyperparams as the parameter-optimization branch but the
# node features come from the frozen BoxEncoder (8d latent + 2d node_type
# = node_dim 10). Trains on the same mixed_dataset_small_10mm.txt for a
# fair Param-vs-Latent comparison.
#
# Submit with:
#   sbatch gnn_train_latentspace.slurm
#
# Monitor with:
#   squeue -u $USER
#   tail -f logs/gnn_latentspace_<jobid>.log
# ============================================================================

# --------------------------------------------------------------------------
# Environment
# --------------------------------------------------------------------------
module load stack/2024-04 gcc/8.5.0 python_gpu/3.11.2 eth_proxy

# Activate your conda/venv environment if applicable.
# source "$HOME/miniconda3/etc/profile.d/conda.sh"
# conda activate masterthesis

# Set project root (adjust if your repo is elsewhere on Euler)
PROJECT_DIR="/cluster/scratch/$USER/MasterThesis"
cd "$PROJECT_DIR" || { echo "ERROR: $PROJECT_DIR not found"; exit 1; }

# --------------------------------------------------------------------------
# wandb: run online via eth_proxy (loaded above).
# --------------------------------------------------------------------------
# export WANDB_MODE=offline

# --------------------------------------------------------------------------
# Reproducibility + Live Logs
# --------------------------------------------------------------------------
export PYTHONHASHSEED=42
export PYTHONUNBUFFERED=1

# Tag the checkpoint filename so it never clashes with the
# parameter-optimization checkpoints in gnn_models/.
export GNN_EXP_NAME="latentspace"

# --------------------------------------------------------------------------
# Create log directory if it doesn't exist
# --------------------------------------------------------------------------
mkdir -p logs

# --------------------------------------------------------------------------
# Run
# --------------------------------------------------------------------------
echo "=========================================="
echo "Job ID:     $SLURM_JOB_ID"
echo "Node:       $SLURMD_NODENAME"
echo "Start time: $(date)"
echo "Project:    $PROJECT_DIR"
echo "Branch:     $(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo 'unknown')"
echo "Commit:     $(git rev-parse --short HEAD 2>/dev/null || echo 'unknown')"
echo "Exp tag:    $GNN_EXP_NAME"
echo "=========================================="

python src/gnn_training.py

EXIT_CODE=$?

echo "=========================================="
echo "End time:  $(date)"
echo "Exit code: $EXIT_CODE"
echo "=========================================="

exit $EXIT_CODE
