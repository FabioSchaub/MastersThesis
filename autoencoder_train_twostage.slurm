#!/bin/bash
#SBATCH --job-name=ae_twostage
#SBATCH --output=logs/ae_twostage_%j.log
#SBATCH --error=logs/ae_twostage_%j.log
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --gres=gpu:1
#SBATCH --gpus-per-node=1

# ============================================================================
# Two-Stage Autoencoder Training on Euler
#
# Submit with:
#   sbatch autoencoder_train_twostage.slurm
#
# Monitor with:
#   squeue -u $USER
#   tail -f logs/ae_twostage_<jobid>.log
#
# Memory breakdown:
#   20000 shapes × 7000 pts × 3 floats × 4 bytes ≈ 1.7 GB surface+query
#   Decoder SIREN (128-dim, 5 layers): ~200k params
#   Encoder PointNet: ~180k params
#   Latent embedding: 20000 × 64 × 4 bytes ≈ 5 MB
#   GPU VRAM: batch_size=64, n_query=5000 → ~100 MB per batch → 1 GB total
#   Safe RAM budget: 32 GB (4 CPUs × 8 GB)
# ============================================================================

# --------------------------------------------------------------------------
# Environment: activate the conda env that has the project deps. The compute
# node does not inherit the login-node shell, so conda has to be initialised
# manually here.
# --------------------------------------------------------------------------
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate masterthesis

# Project root = directory sbatch was called from. Works whether the repo is
# under $HOME, /cluster/scratch, or anywhere else; no hardcoded path.
PROJECT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$PROJECT_DIR" || { echo "ERROR: $PROJECT_DIR not found"; exit 1; }

# --------------------------------------------------------------------------
# wandb: run offline on Euler (no outbound HTTP on compute nodes).
# Sync manually afterwards with: wandb sync wandb/offline-run-*
# --------------------------------------------------------------------------
export WANDB_MODE=offline

# --------------------------------------------------------------------------
# Reproducibility
# --------------------------------------------------------------------------
export PYTHONHASHSEED=42

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
echo "=========================================="

python src/enc_dec_training.py

EXIT_CODE=$?

echo "=========================================="
echo "End time:  $(date)"
echo "Exit code: $EXIT_CODE"
echo "=========================================="

exit $EXIT_CODE
