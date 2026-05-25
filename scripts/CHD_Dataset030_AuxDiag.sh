#!/bin/bash
# =============================================================================
#  CHD_Dataset030_AuxDiag.sh
#  Dataset030_imageCHD_HU — Auxiliary diagnosis head ablation, 200 epochs
#
#  Tests whether a bottleneck classification head (auxiliary supervision on
#  the 8 CHD disease labels) improves segmentation via encoder regularisation.
#
#  Experiments (3d_fullres, 5-fold ensemble):
#    1. AuxDiag baseline  — DA5 + aux diagnosis head (no topology, no FiLM)
#    2. AuxDiag + Topo    — DA5 + aux head + fixed-weight soft-clDice on AO/PA
#    3. FiLM + AuxDiag    — DA5 + FiLM conditioning + aux head (embedding reuse)
#
#  Requires:
#    - disease_map.json in ${nnUNet_preprocessed}/Dataset030_imageCHD_HU/
#    - Preprocessing already done (CHD_Dataset030_imageCHD.sh Phase 0, or run fresh)
#
#  RESUME SUPPORT
#    Each training run creates a .done marker. On resubmission the script
#    skips any step whose marker already exists.
#    Checkpoint dir: ${nnUNet_results}/Dataset030_imageCHD_HU/.checkpoints/CHD_Dataset030_AuxDiag/
#
#  Before first submission:
#    mkdir -p /scratch/users/sastocke/nnunet_CHD/logs
# =============================================================================
#SBATCH --job-name=D030-auxdiag
#SBATCH --partition=bioe
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sastocke@stanford.edu
#SBATCH --output=/scratch/users/sastocke/nnunet_CHD/logs/D030-auxdiag_%j.out
#SBATCH --error=/scratch/users/sastocke/nnunet_CHD/logs/D030-auxdiag_%j.err

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
CKPT_DIR="${nnUNet_results}/${DATASET_NAME}/.checkpoints/CHD_Dataset030_AuxDiag"
SHARED_CKPT_DIR="${nnUNet_results}/${DATASET_NAME}/.checkpoints/shared"
START_TS=$(date +%s)

TRAINERS=(
    "nnUNetTrainerDA5AuxDiag_200epochs"
    "nnUNetTrainerDA5AuxDiagTopo_200epochs"
    "nnUNetTrainerDA5FiLMAuxDiag_200epochs"
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
    echo "║  CHD_Dataset030_AuxDiag.sh  — START                            ║"
    echo "╠══════════════════════════════════════════════════════════════════╣"
    printf "║  %-66s ║\n" "Date/Time  : $(date '+%Y-%m-%d %H:%M:%S')"
    printf "║  %-66s ║\n" "SLURM Job  : ${SLURM_JOB_ID:-manual}  node=${SLURMD_NODENAME:-local}"
    printf "║  %-66s ║\n" "Dataset    : ${DATASET_NAME}  (ID=${DATASET_ID})"
    printf "║  %-66s ║\n" "Plans      : ${PLANS}"
    printf "║  %-66s ║\n" "Config     : ${FULLRES}  |  Folds: 0 1 2 3 4"
    printf "║  %-66s ║\n" "Epochs     : 200"
    printf "║  %-66s ║\n" ""
    printf "║  %-66s ║\n" "Exp 1  DA5AuxDiag_200e      (aux head only)"
    printf "║  %-66s ║\n" "Exp 2  DA5AuxDiagTopo_200e  (aux head + clDice)"
    printf "║  %-66s ║\n" "Exp 3  DA5FiLMAuxDiag_200e  (FiLM + aux head)"
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
    echo "║  CHD_Dataset030_AuxDiag.sh  — COMPLETE                         ║"
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
    printf "║  %-66s ║\n" "Next step: compare DSC against DA5 baseline and FiLMV3"
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
# Phase 2 — Inference on test set (fold 0)
# ─────────────────────────────────────────────
# Uses predict_disease_conditioned.py for all trainers.
# - DA5AuxDiag / DA5AuxDiagTopo: no inference_config.json → auto-falls
#   back to standard nnUNetv2_predict (aux head is training-only).
# - DA5FiLMAuxDiag: has inference_config.json + disease_map.json in
#   model folder → per-case disease vector set before each prediction.
echo "================================================================"
echo "Phase 2: Inference on test set — fold 0"
echo "================================================================"
mkdir -p "${PRED_BASE}"

for TRAINER in "${TRAINERS[@]}"; do
    KEY="p2_infer_$(sk ${TRAINER})"
    OUT_DIR="${PRED_BASE}/$(sk ${TRAINER})"
    MODEL_DIR="${nnUNet_results}/${DATASET_NAME}/${TRAINER}__${PLANS}__${FULLRES}"
    mkdir -p "${OUT_DIR}"
    if is_done "${KEY}"; then
        echo "[SKIP] ${KEY}"
    else
        echo "--- ${KEY} ---"
        python -m nnunetv2.inference.predict_disease_conditioned \
            -i "${IN_DIR}" \
            -o "${OUT_DIR}" \
            -m "${MODEL_DIR}" \
            -f 0
        mark_done "${KEY}"
    fi
done

# ─────────────────────────────────────────────
# END
# ─────────────────────────────────────────────
print_footer
