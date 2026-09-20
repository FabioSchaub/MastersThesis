#!/bin/bash
# ============================================================================
# Zentrales Environment-Setup fuer alle SLURM-Jobs (Euler).
# In jedem .slurm-File EINZEILIG einbinden mit:
#     source "$(dirname "$0")/env_setup.sh"
# oder, robuster (da $0 bei sbatch nicht immer der Skriptpfad ist):
#     source /cluster/scratch/$USER/MasterThesis/env_setup.sh
#
# Bei einem Euler-Stack-Update NUR diese Datei anpassen, nicht 7 SLURM-Files.
# ============================================================================
 
# --- Modules ---------------------------------------------------------------
# stack/2024-05 ist "frozen" (deprecated). Beim naechsten Update hier umstellen.
if ! command -v module >/dev/null 2>&1; then
  [ -n "$LMOD_PKG" ] && [ -f "$LMOD_PKG/init/bash" ] && source "$LMOD_PKG/init/bash"
  command -v module >/dev/null 2>&1 || source /cluster/software/stacks/2024-06/spack/opt/spack/linux-ubuntu22.04-x86_64_v3/gcc-12.2.0/lmod-8.7.24-ou4i7x2rgiaysly4vgawaga6muhkdye4/lmod/lmod/init/bash
fi
module load stack/2024-05 gcc/13.2.0 cuda/12.1.1 python/3.11.6_cuda eth_proxy
 
# --- venv -----------------------------------------------------------------
PROJECT_DIR="$HOME/MasterThesis"
VENV_DIR="$HOME/venv/masterthesis"
source "${VENV_DIR}/bin/activate"
 
# --- Pfad-Reihenfolge ------------------------------------------------------
# WICHTIG: venv-site-packages MUSS vor src und vor den Modul-packages stehen,
# sonst werden venv-Pakete (trimesh, pyg 2.7.0, ...) vom Modul-Python ueberschattet.
# torch kommt bewusst weiter vom Modul (venv hat kein torch), daher kein Konflikt.
export PYTHONPATH="${VENV_DIR}/lib/python3.11/site-packages:/cluster/software/stacks/2024-05/python-cuda/3.11.6/lib/python3.11/site-packages:${PROJECT_DIR}/src"
 
# --- Reproduzierbarkeit / Live-Logs ---------------------------------------
export PYTHONHASHSEED=42
export PYTHONUNBUFFERED=1
 
# --- CUDA-Fragmentierung (grosse Decoder) ---------------------------------
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
