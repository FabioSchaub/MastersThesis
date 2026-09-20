#!/bin/bash
#SBATCH --job-name=sim_gnn
#SBATCH --output=logs/sim_gnn_%j.log
#SBATCH --error=logs/sim_gnn_%j.log
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --gres=gpu:1,gpumem:24g
#SBATCH --gpus-per-node=1

# ============================================================================
# SimAssemblyGNN training on Euler (branch feature/ae-shape-vocab-14).
#
# Trains the late-fork GATv2 GNN (SimAssemblyGNN) on the Isaac-Sim 2-block
# feasibility dataset. Node = z(LV latent) + bbox(3) + node_type(2); node_dim
# read dynamically from the latents .pt. Targets are the sim's binary flags:
#   failure head (4): OBJECT_GAP, OVERLAP_INSUFFICIENT, THICKNESS_EXCEEDED, TIPPING
#   good head   (1): Assembly_Good
#
# Files that MUST be present on Euler (in $PROJECT_DIR):
#   1. data/sim_shape_dataset_1613.txt                       (the dataset, ~74MB)
#   2. encoder_decoder_model/shape_latents_lv_v14_lam005.pt  (per-type LV latents)
#   3. the code: src/sim_gnn_training.py, src/gnn.py, src/sim_gnn_dataset.py,
#      config/config.yaml (+ config/)
# Output checkpoint: gnn_models/sim_gnn_shape_node<node_dim>_<latent_dim>.pth
#
# Monitor:  squeue -u $USER  ;  tail -f logs/sim_gnn_<jobid>.log
# ============================================================================

source /cluster/home/$USER/MasterThesis/env_setup.sh
# conda activate masterthesis   # if not handled by env_setup.sh

PROJECT_DIR="/cluster/home/$USER/MasterThesis"
cd "$PROJECT_DIR" || { echo "ERROR: $PROJECT_DIR not found"; exit 1; }

export PYTHONHASHSEED=42
export PYTHONUNBUFFERED=1
export WANDB_MODE=offline

mkdir -p logs

echo "=========================================="
echo "Job ID:     $SLURM_JOB_ID"
echo "Node:       $SLURMD_NODENAME"
echo "Start time: $(date)"
echo "Project:    $PROJECT_DIR"
echo "Branch:     $(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo 'unknown')"
echo "Commit:     $(git rev-parse --short HEAD 2>/dev/null || echo 'unknown')"
echo "=========================================="

python -m src.sim_gnn_training data/sim_shape_dataset_1613.txt

EXIT_CODE=$?

echo "=========================================="
echo "End time:  $(date)"
echo "Exit code: $EXIT_CODE"
echo "=========================================="

exit $EXIT_CODE
