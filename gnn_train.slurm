#!/bin/bash
#SBATCH --job-name=gnn_train
#SBATCH --output=logs/gnn_train_%j.log
#SBATCH --error=logs/gnn_train_%j.log
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --gres=gpu:1
#SBATCH --gpus-per-node=1

# ============================================================================
# GNN Training on Euler
#
# Submit with:
#   sbatch gnn_train.slurm
#
# Monitor with:
#   squeue -u $USER
#   tail -f logs/gnn_train_<jobid>.log
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
# wandb: run online via eth_proxy (loaded above). Compute nodes can reach
# api.wandb.ai through the proxy, so runs stream live to the cloud and the
# Files tab fills with config + console log automatically.
#
# Pre-requisite: `wandb login` must have been run once on a login node so
# the API key is in ~/.netrc, OR export WANDB_API_KEY here.
#
# To fall back to offline mode (e.g. proxy down), uncomment the next line
# and sync afterwards from a login node with `wandb sync wandb/offline-run-*`.
# --------------------------------------------------------------------------
# export WANDB_MODE=offline

# --------------------------------------------------------------------------
# Reproducibility + Live Logs
# --------------------------------------------------------------------------
export PYTHONHASHSEED=42
# Unbuffered stdout so tqdm + print() arrive in the log in real time.
export PYTHONUNBUFFERED=1

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
echo "=========================================="

python src/gnn_training.py

EXIT_CODE=$?

echo "=========================================="
echo "End time:  $(date)"
echo "Exit code: $EXIT_CODE"
echo "=========================================="

exit $EXIT_CODE
