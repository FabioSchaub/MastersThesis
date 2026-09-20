#!/bin/bash
#SBATCH --job-name=ae_general
#SBATCH --output=logs/ae_general_%j.log
#SBATCH --error=logs/ae_general_%j.log
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --gres=gpu:1,gpumem:24g    # 24GB cards (Quadro RTX 6000 / Titan RTX) = the biggest on this cluster; NO A100 available, 24G is the max gpumem. Avoids the 11GB 2080 Ti. (Do NOT use --gpus=1 + gpumem:40g: routes to gpupr.24h -> QOSGrpCpuLimit, and 40G doesn't exist here.)
#SBATCH --exclude=eu-lo-g2-028
#SBATCH --gpus-per-node=1

# ============================================================================
# General-shape Autoencoder Training on Euler (PointNet encoder)
#
# Trains the SDF auto-decoder + PointNet encoder on the MIXED shape vocabulary
# (box / cylinder / rounded plate / L-bracket / T-piece) via generate_random_shape.
# GENERAL_SHAPES=1 selects the PointNet (surface-points -> z) path; the final
# checkpoint is best_encoder_decoder_GENERAL_latentdim{N}.pth, which does NOT
# overwrite the box codec.
#
# Submit (from the repo root):  sbatch autoencoder_general.slurm
# Monitor:  squeue -u $USER  ;  tail -f logs/ae_general_<jobid>.log
# ============================================================================

source /cluster/scratch/$USER/MasterThesis/env_setup.sh

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$PROJECT_DIR" || { echo "ERROR: $PROJECT_DIR not found"; exit 1; }

export WANDB_MODE=offline
export PYTHONHASHSEED=42
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True  # reduce CUDA fragmentation (larger decoder at hidden_dim 512)

# --- General-shape run ---
export GENERAL_SHAPES=1     # PointNet encoder on surface points
export AE_CANON_NORM=1      # centre+scale each shape to max-extent 0.9 (match inference frame) -> ckpt *_general_canon_*
# AE_N_SHAPES is intentionally NOT forced here: config.autoencoder.n_shapes
# (100000) is the single source of truth. Pass --export=...,AE_N_SHAPES=<n> to
# override per-run. (Was hard-set to 50000/150000 before, which silently
# overrode the config.)
# Least Volume — the paper-faithful setup (Chen & Fuge ICLR 2024): the volume
# penalty belongs in a TRUE autoencoder phase (our Stage 3), where z=encoder(x)
# anchors the codes so they cannot collapse. It is a TINY penalty over MANY
# epochs (their lam~1e-4..1e-3, up to 10000 ep), NOT a strong one over few.
#   - LV_S3_ONLY=1: skip Stage 1/2, load the trained baseline codec and run only
#     the Stage-3 LV fine-tune (fast; the baseline already reconstructs sub-mm).
#   - lam_vol ~1e-3 (vs recon ~0.003 -> a small nudge), eta 0.1 (our z scale),
#     AE_STAGE3_EPOCHS large. -> ckpt *_general_canon_lv_* (baseline untouched).
# (Stage-1 LV / warm-start / spectral norm are OFF: in the DeepSDF auto-decoder
# spectral norm starved the free codes to z=0; strong lam over few epochs did not
# compress at 50k. The Stage-3 true-AE route fixes both.)
export LV_S3_ONLY="${LV_S3_ONLY:-1}"
export LV_LAMBDA_VOL="${LV_LAMBDA_VOL:-0.001}"
export LV_VOL_ETA="${LV_VOL_ETA:-0.1}"
export AE_STAGE3_EPOCHS="${AE_STAGE3_EPOCHS:-300}"
export LV_WARMSTART="${LV_WARMSTART:-0}"
export LV_SPECTRAL="${LV_SPECTRAL:-0}"
export LV_ENC_SPECTRAL="${LV_ENC_SPECTRAL:-0}"
export LV_LIPSCHITZ_K="${LV_LIPSCHITZ_K:-6}"

mkdir -p logs

echo "=========================================="
echo "Job ID:     $SLURM_JOB_ID"
echo "Node:       $SLURMD_NODENAME"
echo "Start time: $(date)"
echo "Project:    $PROJECT_DIR"
echo "Branch:     $(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo 'unknown')"
echo "GENERAL_SHAPES=$GENERAL_SHAPES  AE_N_SHAPES=${AE_N_SHAPES:-<config:100000>}  LV_S3_ONLY=$LV_S3_ONLY  LV_LAMBDA_VOL=$LV_LAMBDA_VOL eta=$LV_VOL_ETA  AE_STAGE3_EPOCHS=$AE_STAGE3_EPOCHS"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi -L 2>/dev/null || echo "no GPU visible (nvidia-smi -L failed)"
echo "=========================================="

python src/enc_dec_training.py

EXIT_CODE=$?

echo "=========================================="
echo "End time:  $(date)"
echo "Exit code: $EXIT_CODE"
echo "Checkpoint: encoder_decoder_model/best_encoder_decoder_general_canon${LV_LAMBDA_VOL:+_lv}_latentdim*.pth"
echo "=========================================="

exit $EXIT_CODE
