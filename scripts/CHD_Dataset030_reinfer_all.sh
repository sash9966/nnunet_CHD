#!/bin/bash
# =============================================================================
#  CHD_Dataset030_reinfer_all.sh
#  Cross-inference: re-run fold-0 test-set predictions for every trained
#  Dataset030 model (baseline + AuxDiag + CrossAttn).
#
#  - Baseline (DA5 fullres): standard nnUNetv2_predict, no disease vec.
#  - AuxDiag trainers: predict_disease_conditioned (DA5AuxDiag/Topo fall
#    back automatically; FiLMAuxDiag uses per-case disease vectors).
#  - CrossAttn trainers: predict_disease_conditioned, per-case disease vec.
#
#  Output: ${nnUNet_results}/Dataset030_imageCHD_HU/predictions/<trainer>/
#
#  Overwrites existing predictions. Safe to resubmit — .done markers skip
#  completed trainers.
#
#  Before first submission:
#    mkdir -p /scratch/users/sastocke/nnunet_CHD/logs
# =============================================================================
#SBATCH --job-name=D030-reinfer-all
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sastocke@stanford.edu
#SBATCH --output=/scratch/users/sastocke/nnunet_CHD/logs/D030-reinfer-all_%j.out
#SBATCH --error=/scratch/users/sastocke/nnunet_CHD/logs/D030-reinfer-all_%j.err

set -euo pipefail

# ─────────────────────────────────────────────
# Environment
# ─────────────────────────────────────────────
module purge
module load gcc/12.4.0 cuda/11.7.1 cmake/3.24.2
source /oak/stanford/groups/amarsden/sastocke/miniconda/etc/profile.d/conda.sh
conda activate /scratch/users/sastocke/conda_envs/nnunet310
hash -r

export nnUNet_raw="/scratch/users/sastocke/nnunet_CHD/nnUNet_raw"
export nnUNet_preprocessed="/scratch/users/sastocke/nnunet_CHD/nnUNet_preprocessed"
export nnUNet_results="/scratch/users/sastocke/nnunet_CHD/nnUNet_results"
export PYTHONPATH="/scratch/users/sastocke/nnunet_CHD:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────
DATASET_ID=30
DATASET_NAME="Dataset030_imageCHD_HU"
PLANS="nnUNetResEncUNetMPlans"
FULLRES="3d_fullres"
IN_DIR="${nnUNet_raw}/${DATASET_NAME}/imagesTs"
PRED_BASE="${nnUNet_results}/${DATASET_NAME}/predictions"
CKPT_DIR="${nnUNet_results}/${DATASET_NAME}/.checkpoints/reinfer_all"

mkdir -p "${CKPT_DIR}" "${PRED_BASE}"

sk() { echo "$1" | sed 's/nnUNetTrainer//' | sed 's/_200epochs/200e/'; }
mark_done() { touch "${CKPT_DIR}/${1}.done"; }
is_done()   { [[ -f "${CKPT_DIR}/${1}.done" ]]; }

echo ""
echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║  CHD_Dataset030_reinfer_all.sh  — START                        ║"
echo "╠══════════════════════════════════════════════════════════════════╣"
printf "║  %-66s ║\n" "Date/Time  : $(date '+%Y-%m-%d %H:%M:%S')"
printf "║  %-66s ║\n" "SLURM Job  : ${SLURM_JOB_ID:-manual}"
printf "║  %-66s ║\n" "Output dir : ${PRED_BASE}"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""
echo "  Completed steps (from previous runs, if any):"
ls "${CKPT_DIR}/"*.done 2>/dev/null \
    | xargs -I{} basename {} .done \
    | sort | sed 's/^/    [DONE] /' \
    || echo "    (none)"
echo ""

# ─────────────────────────────────────────────
# 1. Baseline — standard nnUNetv2_predict
# ─────────────────────────────────────────────
echo "================================================================"
echo "1. Baseline: DA5 fullres (no disease conditioning)"
echo "================================================================"
KEY="baseline_DA5fullres"
OUT_DIR="${PRED_BASE}/DA5_fullres"
if is_done "${KEY}"; then
    echo "[SKIP] ${KEY}"
else
    mkdir -p "${OUT_DIR}"
    nnUNetv2_predict \
        -i "${IN_DIR}" \
        -o "${OUT_DIR}" \
        -d ${DATASET_ID} -c ${FULLRES} \
        -f 0 \
        -tr nnUNetTrainerDA5_200epochs -p ${PLANS}
    mark_done "${KEY}"
    echo "[DONE] ${KEY}"
fi

# Helper: nnUNetv2_predict for non-conditioned trainers (no disease vec needed).
run_standard_infer() {
    local key="$1" trainer="$2"
    local out_dir="${PRED_BASE}/$(sk ${trainer})"
    local ckpt="${nnUNet_results}/${DATASET_NAME}/${trainer}__${PLANS}__${FULLRES}/fold_0/checkpoint_final.pth"

    if is_done "${key}"; then echo "[SKIP] ${key}"; return; fi
    if [[ ! -f "${ckpt}" ]]; then
        echo "[WARN] ${key}: no checkpoint — skipping"; return
    fi
    echo "--- ${key} ---"
    mkdir -p "${out_dir}"
    nnUNetv2_predict \
        -i "${IN_DIR}" -o "${out_dir}" \
        -d ${DATASET_ID} -c ${FULLRES} \
        -f 0 -tr ${trainer} -p ${PLANS}
    mark_done "${key}"
    echo "[DONE] ${key}"
}

# Helper: predict_disease_conditioned for trainers that need disease vecs.
run_conditioned_infer() {
    local key="$1" trainer="$2"
    local model_dir="${nnUNet_results}/${DATASET_NAME}/${trainer}__${PLANS}__${FULLRES}"
    local out_dir="${PRED_BASE}/$(sk ${trainer})"
    local ckpt="${model_dir}/fold_0/checkpoint_final.pth"

    if is_done "${key}"; then echo "[SKIP] ${key}"; return; fi
    if [[ ! -f "${ckpt}" ]]; then
        echo "[WARN] ${key}: no checkpoint — skipping"; return
    fi
    echo "--- ${key} ---"
    mkdir -p "${out_dir}"
    python -m nnunetv2.inference.predict_disease_conditioned \
        -i "${IN_DIR}" -o "${out_dir}" -m "${model_dir}" -f 0
    mark_done "${key}"
    echo "[DONE] ${key}"
}

# ─────────────────────────────────────────────
# 2. AuxDiag trainers
#    DA5AuxDiag / DA5AuxDiagTopo: plain segmentation network, no disease vec
#      at inference → use standard nnUNetv2_predict (avoids silent multiprocessing
#      failure in predict_disease_conditioned's non-conditioned fallback path).
#    DA5FiLMAuxDiag: FiLM conditioning → needs per-case disease vec.
# ─────────────────────────────────────────────
echo "================================================================"
echo "2. AuxDiag trainers"
echo "================================================================"
run_standard_infer   "auxdiag_DA5AuxDiag200e"        "nnUNetTrainerDA5AuxDiag_200epochs"
run_standard_infer   "auxdiag_DA5AuxDiagTopo200e"    "nnUNetTrainerDA5AuxDiagTopo_200epochs"
run_conditioned_infer "auxdiag_DA5FiLMAuxDiag200e"   "nnUNetTrainerDA5FiLMAuxDiag_200epochs"

# ─────────────────────────────────────────────
# 3. CrossAttn — predict_disease_conditioned (all need disease vecs)
# ─────────────────────────────────────────────
echo "================================================================"
echo "3. CrossAttn trainers (per-case disease vectors)"
echo "================================================================"
run_conditioned_infer "crossattn_DA5CrossAttn200e"         "nnUNetTrainerDA5CrossAttn_200epochs"
run_conditioned_infer "crossattn_DA5CrossAttnTopo200e"     "nnUNetTrainerDA5CrossAttnTopo_200epochs"
run_conditioned_infer "crossattn_DA5AuxDiagCrossAttn200e"  "nnUNetTrainerDA5AuxDiagCrossAttn_200epochs"

echo ""
echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║  CHD_Dataset030_reinfer_all.sh  — COMPLETE                     ║"
printf "║  %-66s ║\n" "Date/Time : $(date '+%Y-%m-%d %H:%M:%S')"
printf "║  %-66s ║\n" "Results   : ${PRED_BASE}/"
echo "╚══════════════════════════════════════════════════════════════════╝"
