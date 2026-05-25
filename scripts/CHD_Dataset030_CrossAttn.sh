#!/bin/bash
# =============================================================================
#  CHD_Dataset030_CrossAttn.sh
#  Dataset030_imageCHD_HU — Cross-attention conditioning ablation, 200 epochs
#
#  Tests spatially-localised disease conditioning via per-stage cross-attention
#  against FiLM (global bottleneck modulation) and the baseline.
#
#  Experiments (3d_fullres, 5-fold ensemble):
#    1. CrossAttn         — DA5 + cross-attn at every decoder stage
#    2. CrossAttnTopo     — DA5 + cross-attn + soft-clDice on AO/PA
#    3. AuxDiagCrossAttn  — DA5 + bottleneck aux head + cross-attn (embedding reuse)
#
#  Attention entropy is logged every 50 steps per decoder stage in the training
#  log — low entropy → selective disease-token attention (good conditioning).
#
#  Requires:
#    - disease_map.json in ${nnUNet_preprocessed}/Dataset030_imageCHD_HU/
#    - Preprocessing already done (CHD_Dataset030_imageCHD.sh Phase 0, or run fresh)
#
#  RESUME SUPPORT
#    Checkpoint dir: ${nnUNet_results}/Dataset030_imageCHD_HU/.checkpoints/CHD_Dataset030_CrossAttn/
#
#  Before first submission:
#    mkdir -p /scratch/users/sastocke/nnunet_CHD/logs
# =============================================================================
#SBATCH --job-name=D030-xattn
#SBATCH --partition=bioe
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sastocke@stanford.edu
#SBATCH --output=/scratch/users/sastocke/nnunet_CHD/logs/D030-xattn_%j.out
#SBATCH --error=/scratch/users/sastocke/nnunet_CHD/logs/D030-xattn_%j.err

set -euo pipefail

# ─────────────────────────────────────────────
# 1.  Environment
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
# 2.  Configuration
# ─────────────────────────────────────────────
DATASET_ID=30
DATASET_NAME="Dataset030_imageCHD_HU"
PLANNER="nnUNetPlannerResEncM"
PLANS="nnUNetResEncUNetMPlans"
FULLRES="3d_fullres"
REPO="/scratch/users/sastocke/nnunet_CHD"
IN_DIR="${nnUNet_raw}/${DATASET_NAME}/imagesTs"
PRED_BASE="${nnUNet_results}/${DATASET_NAME}/predictions"
CKPT_DIR="${nnUNet_results}/${DATASET_NAME}/.checkpoints/CHD_Dataset030_CrossAttn"
SHARED_CKPT_DIR="${nnUNet_results}/${DATASET_NAME}/.checkpoints/shared"
START_TS=$(date +%s)

TRAINERS=(
    "nnUNetTrainerDA5CrossAttn_200epochs"
    "nnUNetTrainerDA5CrossAttnTopo_200epochs"
    "nnUNetTrainerDA5AuxDiagCrossAttn_200epochs"
)

# ─────────────────────────────────────────────
# 3.  Checkpoint helpers
# ─────────────────────────────────────────────
mkdir -p "${CKPT_DIR}" "${SHARED_CKPT_DIR}"

sk() { echo "$1" | sed 's/nnUNetTrainer//' | sed 's/_200epochs/200e/'; }
mark_done() { touch "${CKPT_DIR}/${1}.done"; }
is_done()   { [[ -f "${CKPT_DIR}/${1}.done" ]]; }
shared_mark_done() { touch "${SHARED_CKPT_DIR}/${1}.done"; }
shared_is_done()   { [[ -f "${SHARED_CKPT_DIR}/${1}.done" ]]; }

verify_preprocessing() {
    local cfg=$1
    local n_raw n_prep prep_dir
    n_raw=$(ls "${nnUNet_raw}/${DATASET_NAME}/imagesTr/" | grep -c "_0000")
    prep_dir=$(find "${nnUNet_preprocessed}/${DATASET_NAME}" -maxdepth 1 -type d -name "*_${cfg}" 2>/dev/null | head -1)
    if [[ -z "${prep_dir}" ]]; then
        echo "ERROR: No preprocessed directory found for ${cfg}."
        echo "  Looked in: ${nnUNet_preprocessed}/${DATASET_NAME}/*_${cfg}"
        echo "  Fix: nnUNetv2_preprocess -d ${DATASET_ID} -pl ${PLANNER} -c ${cfg}"
        exit 1
    fi
    n_prep=$(find "${prep_dir}" -maxdepth 1 -name "*_image.b2nd" 2>/dev/null | wc -l)
    echo "[VERIFY] ${cfg}: ${n_prep}/${n_raw} cases in ${prep_dir}"
    if [[ ${n_prep} -lt ${n_raw} ]]; then
        echo "ERROR: Missing preprocessed files for ${cfg} (${n_prep}/${n_raw})."
        echo "  Fix: nnUNetv2_preprocess -d ${DATASET_ID} -pl ${PLANNER} -c ${cfg}"
        exit 1
    fi
}

# ─────────────────────────────────────────────
# 4.  Banner helpers
# ─────────────────────────────────────────────
print_banner() {
    echo ""
    echo "╔══════════════════════════════════════════════════════════════════╗"
    echo "║  CHD_Dataset030_CrossAttn.sh  — START                          ║"
    echo "╠══════════════════════════════════════════════════════════════════╣"
    printf "║  %-66s ║\n" "Date/Time  : $(date '+%Y-%m-%d %H:%M:%S')"
    printf "║  %-66s ║\n" "SLURM Job  : ${SLURM_JOB_ID:-manual}  node=${SLURMD_NODENAME:-local}"
    printf "║  %-66s ║\n" "Dataset    : ${DATASET_NAME}  (ID=${DATASET_ID})"
    printf "║  %-66s ║\n" "Plans      : ${PLANS}"
    printf "║  %-66s ║\n" "Config     : ${FULLRES}  |  Folds: 0 1 2 3 4"
    printf "║  %-66s ║\n" "Epochs     : 200"
    printf "║  %-66s ║\n" ""
    printf "║  %-66s ║\n" "Exp 1  DA5CrossAttn_200e         (cross-attn only)"
    printf "║  %-66s ║\n" "Exp 2  DA5CrossAttnTopo_200e     (cross-attn + clDice)"
    printf "║  %-66s ║\n" "Exp 3  DA5AuxDiagCrossAttn_200e  (aux head + cross-attn)"
    printf "║  %-66s ║\n" ""
    printf "║  %-66s ║\n" "Attention entropy logged every 50 steps in training log."
    printf "║  %-66s ║\n" "  Low entropy -> selective disease-token attention (good)"
    printf "║  %-66s ║\n" "  High entropy -> uniform (ln(8)=2.08, FiLM-like)"
    printf "║  %-66s ║\n" ""
    printf "║  %-66s ║\n" "Raw data   : ${IN_DIR}"
    printf "║  %-66s ║\n" "Results    : ${nnUNet_results}/${DATASET_NAME}"
    printf "║  %-66s ║\n" "Checkpoint : ${CKPT_DIR}"
    echo "╚══════════════════════════════════════════════════════════════════╝"
    echo ""
    echo "  Completed steps (from previous runs, if any):"
    ls "${CKPT_DIR}/"*.done 2>/dev/null \
        | xargs -I{} basename {} .done \
        | sort | sed 's/^/    [DONE] /' \
        || echo "    (none — fresh run)"
    echo ""
}

print_footer() {
    local elapsed=$(( $(date +%s) - START_TS ))
    local hh=$(( elapsed / 3600 ))
    local mm=$(( (elapsed % 3600) / 60 ))
    local ss=$(( elapsed % 60 ))
    echo ""
    echo "╔══════════════════════════════════════════════════════════════════╗"
    echo "║  CHD_Dataset030_CrossAttn.sh  — COMPLETE                       ║"
    echo "╠══════════════════════════════════════════════════════════════════╣"
    printf "║  %-66s ║\n" "Date/Time  : $(date '+%Y-%m-%d %H:%M:%S')"
    printf "║  %-66s ║\n" "SLURM Job  : ${SLURM_JOB_ID:-manual}"
    printf "║  %-66s ║\n" "Dataset    : ${DATASET_NAME}  (ID=${DATASET_ID})"
    printf "║  %-66s ║\n" "Elapsed    : ${hh}h ${mm}m ${ss}s"
    printf "║  %-66s ║\n" ""
    printf "║  %-66s ║\n" "Inference results in: ${PRED_BASE}/"
    for TR in "${TRAINERS[@]}"; do
        printf "║    %-64s ║\n" "$(sk ${TR})"
    done
    printf "║  %-66s ║\n" ""
    printf "║  %-66s ║\n" "Compare DSC vs DA5 baseline, FiLMV3, and AuxDiag."
    printf "║  %-66s ║\n" "Check attention entropy in training logs to confirm"
    printf "║  %-66s ║\n" "the model is learning selective disease attention."
    echo "╚══════════════════════════════════════════════════════════════════╝"
}

# ─────────────────────────────────────────────
# START
# ─────────────────────────────────────────────
print_banner

# ─────────────────────────────────────────────
# Phase 0 — Preprocess (skip if done by CHD_Dataset030_imageCHD.sh)
# ─────────────────────────────────────────────
if shared_is_done "p0_preprocess"; then
    echo "[SKIP] Phase 0: preprocess already done (shared marker)"
else
    echo "================================================================"
    echo "Phase 0: plan_and_preprocess — ${FULLRES}"
    echo "================================================================"
    nnUNetv2_plan_and_preprocess \
        -d ${DATASET_ID} \
        -pl ${PLANNER} \
        -c ${FULLRES} \
        --verify_dataset_integrity
    shared_mark_done "p0_preprocess"
fi
verify_preprocessing "${FULLRES}"

# ─────────────────────────────────────────────
# Phase 0b — Build disease_map.json from diagnosis CSV
# ─────────────────────────────────────────────
if shared_is_done "p0b_disease_map"; then
    echo "[SKIP] Phase 0b: disease_map.json already built (shared marker)"
else
    echo "================================================================"
    echo "Phase 0b: Build disease_map.json"
    echo "================================================================"
    python "${REPO}/scripts/make_disease_map.py" \
        --dataset-id ${DATASET_ID} \
        --csv-name imageCHD_dataset_info.xlsx
    shared_mark_done "p0b_disease_map"
fi

# ─────────────────────────────────────────────
# Phase 1 — Training: 3 trainers x 5 folds
# ─────────────────────────────────────────────
echo "================================================================"
echo "Phase 1: Fullres training — ${#TRAINERS[@]} trainers x 5 folds"
echo "================================================================"
for TRAINER in "${TRAINERS[@]}"; do
    for FOLD in 0; do
        KEY="p1_$(sk ${TRAINER})_fold${FOLD}"
        if is_done "${KEY}"; then
            echo "[SKIP] ${KEY}"
        else
            echo "--- ${KEY} ---"
            nnUNetv2_train ${DATASET_ID} ${FULLRES} ${FOLD} \
                -tr ${TRAINER} -p ${PLANS} --npz
            mark_done "${KEY}"
        fi
    done
done

# ─────────────────────────────────────────────
# Phase 2 — Inference on test set (5-fold ensemble)
# ─────────────────────────────────────────────
echo "================================================================"
echo "Phase 2: Inference on test set — 5-fold ensemble"
echo "================================================================"
mkdir -p "${PRED_BASE}"

for TRAINER in "${TRAINERS[@]}"; do
    KEY="p2_infer_$(sk ${TRAINER})"
    OUT_DIR="${PRED_BASE}/$(sk ${TRAINER})"
    mkdir -p "${OUT_DIR}"
    if is_done "${KEY}"; then
        echo "[SKIP] ${KEY}"
    else
        echo "--- ${KEY} ---"
        nnUNetv2_predict \
            -i "${IN_DIR}" -o "${OUT_DIR}" \
            -d ${DATASET_ID} -c ${FULLRES} \
            -f 0 \
            -tr ${TRAINER} -p ${PLANS}
        mark_done "${KEY}"
    fi
done

# ─────────────────────────────────────────────
# END
# ─────────────────────────────────────────────
print_footer
