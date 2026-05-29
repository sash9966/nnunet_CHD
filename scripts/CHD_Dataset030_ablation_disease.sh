#!/bin/bash
# =============================================================================
#  CHD_Dataset030_ablation_disease.sh   (Job 2 of 3)
#  Dataset030_imageCHD_HU — Disease conditioning alone (no topology)
#  Fold 0, 200 epochs
#
#  Companion scripts:
#    Job 1: CHD_Dataset030_ablation_topo.sh   (B1, B2, T1, T2, T3)
#    Job 3: CHD_Dataset030_ablation_combos.sh (C1, C2, C3, C4, C5)
#
#  ─── Rows trained here ───────────────────────────────────────────────────
#  ID    Trainer                                    Config         Inference path
#  ────  ─────────────────────────────────────────  ─────────────  ───────────────────────
#  D1    DA5FiLMV3_200e                             3d_fullres     predict_disease_conditioned
#  D2    DA5AuxDiag_200e                            3d_fullres     nnUNetv2_predict (training-only aux)
#  D3    DA5CrossAttn_200e                          3d_fullres     predict_disease_conditioned
#
#  Total: 3 fullres trainings, 3 inferences
#  Approx 45h walltime — likely fits in one 48h SLURM slot but resume-safe if not.
#
#  RESUME SUPPORT
#    Same as sibling scripts: .done markers per step, nnU-Net auto-resumes
#    interrupted training, preprocess + disease_map markers live in SHARED.
#    Local checkpoint dir: ${nnUNet_results}/Dataset030_imageCHD_HU/.checkpoints/CHD_Dataset030_ablation_disease/
#
#  Before first submission:
#    mkdir -p /scratch/users/sastocke/nnunet_CHD/logs
# =============================================================================
#SBATCH --job-name=D030-abl-disease
#SBATCH --partition=bioe
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sastocke@stanford.edu
#SBATCH --output=/scratch/users/sastocke/nnunet_CHD/logs/D030-abl-disease_%j.out
#SBATCH --error=/scratch/users/sastocke/nnunet_CHD/logs/D030-abl-disease_%j.err

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
PRED_BASE="${nnUNet_results}/${DATASET_NAME}/predictions_ablation"
CKPT_DIR="${nnUNet_results}/${DATASET_NAME}/.checkpoints/CHD_Dataset030_ablation_disease"
SHARED_CKPT_DIR="${nnUNet_results}/${DATASET_NAME}/.checkpoints/shared"
START_TS=$(date +%s)

# Parallel arrays: trainer + inference mode ("plain" = nnUNetv2_predict, "cond" = disease-conditioned)
FULLRES_TRAINERS=(
    "nnUNetTrainerDA5FiLMV3_200epochs"            # D1
    "nnUNetTrainerDA5AuxDiag_200epochs"           # D2
    "nnUNetTrainerDA5CrossAttn_200epochs"         # D3
)
FULLRES_INFER_MODES=(
    "cond"    # D1: FiLM needs disease vec
    "plain"   # D2: AuxDiag is training-only
    "cond"    # D3: CrossAttn needs disease vec
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
    local n_raw n_prep
    local prep_dir="${nnUNet_preprocessed}/${DATASET_NAME}/${PLANS}_${cfg}"
    n_raw=$(ls "${nnUNet_raw}/${DATASET_NAME}/imagesTr/" | grep -c "_0000" || true)
    if [[ ! -d "${prep_dir}" ]]; then
        echo "ERROR: preprocessed dir not found: ${prep_dir}"
        echo "  Fix: nnUNetv2_preprocess -d ${DATASET_ID} -pl ${PLANNER} -c ${cfg}"
        exit 1
    fi
    n_prep=$(find "${prep_dir}" -maxdepth 1 -name "*_image.b2nd" 2>/dev/null | wc -l)
    echo "[VERIFY] ${cfg}: ${n_prep}/${n_raw} cases in ${PLANS}_${cfg}"
    if [[ ${n_prep} -lt ${n_raw} ]]; then
        echo "ERROR: only ${n_prep}/${n_raw} preprocessed files found for ${cfg}."
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
    echo "║  CHD_Dataset030_ablation_disease.sh  — START  (Job 2 of 3)      ║"
    echo "╠══════════════════════════════════════════════════════════════════╣"
    printf "║  %-66s ║\n" "Date/Time  : $(date '+%Y-%m-%d %H:%M:%S')"
    printf "║  %-66s ║\n" "SLURM Job  : ${SLURM_JOB_ID:-manual}  node=${SLURMD_NODENAME:-local}"
    printf "║  %-66s ║\n" "Dataset    : ${DATASET_NAME}  (ID=${DATASET_ID})"
    printf "║  %-66s ║\n" "Fold       : 0  |  Epochs: 200"
    printf "║  %-66s ║\n" ""
    printf "║  %-66s ║\n" "Trainings  : 3 fullres (D1, D2, D3)"
    printf "║  %-66s ║\n" ""
    printf "║  %-66s ║\n" "D1  DA5FiLMV3_200e         (FiLM bottleneck conditioning)"
    printf "║  %-66s ║\n" "D2  DA5AuxDiag_200e        (training-only encoder regulariser)"
    printf "║  %-66s ║\n" "D3  DA5CrossAttn_200e      (per-stage cross-attention)"
    printf "║  %-66s ║\n" ""
    printf "║  %-66s ║\n" "Inference  : ${PRED_BASE}"
    printf "║  %-66s ║\n" "Checkpoint : ${CKPT_DIR}"
    echo "╚══════════════════════════════════════════════════════════════════╝"
    echo ""
    echo "  Completed steps from previous runs (.done markers):"
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
    echo "║  CHD_Dataset030_ablation_disease.sh  — END  (Job 2 of 3)        ║"
    echo "╠══════════════════════════════════════════════════════════════════╣"
    printf "║  %-66s ║\n" "Date/Time  : $(date '+%Y-%m-%d %H:%M:%S')"
    printf "║  %-66s ║\n" "Elapsed    : ${hh}h ${mm}m ${ss}s"
    printf "║  %-66s ║\n" ""
    printf "║  %-66s ║\n" "If incomplete: resubmit this script — it resumes automatically"
    echo "╚══════════════════════════════════════════════════════════════════╝"
}

# ─────────────────────────────────────────────
# 5.  Inference dispatcher
# ─────────────────────────────────────────────
infer_fullres() {
    # $1 = trainer class, $2 = infer mode ("plain"|"cond"), $3 = output subdir name
    local TRAINER=$1
    local MODE=$2
    local OUT_NAME=$3
    local OUT_DIR="${PRED_BASE}/${OUT_NAME}"
    local MODEL_DIR="${nnUNet_results}/${DATASET_NAME}/${TRAINER}__${PLANS}__${FULLRES}"
    mkdir -p "${OUT_DIR}"
    if [[ "${MODE}" == "cond" ]]; then
        python -m nnunetv2.inference.predict_disease_conditioned \
            -i "${IN_DIR}" \
            -o "${OUT_DIR}" \
            -m "${MODEL_DIR}" \
            -f 0
    else
        nnUNetv2_predict \
            -i "${IN_DIR}" -o "${OUT_DIR}" \
            -d ${DATASET_ID} -c ${FULLRES} \
            -f 0 \
            -tr ${TRAINER} -p ${PLANS}
    fi
}

# ─────────────────────────────────────────────
# START
# ─────────────────────────────────────────────
print_banner
mkdir -p "${PRED_BASE}"

# ─────────────────────────────────────────────
# Phase 0 — Plan and preprocess (shared with sibling jobs; only fullres needed here)
# ─────────────────────────────────────────────
_prep_check="${nnUNet_preprocessed}/${DATASET_NAME}/${PLANS}_${FULLRES}"
_n_prep_check=$(find "${_prep_check}" -maxdepth 1 -name "*_image.b2nd" 2>/dev/null | wc -l)
if [[ ${_n_prep_check} -gt 0 ]]; then
    echo "[SKIP] Phase 0: ${_n_prep_check} cases already in ${PLANS}_${FULLRES}"
    shared_mark_done "p0_preprocess_all3"
else
    echo "================================================================"
    echo "Phase 0: plan_and_preprocess — ${FULLRES} (and lowres/cascade for sibling jobs)"
    echo "================================================================"
    nnUNetv2_plan_and_preprocess \
        -d ${DATASET_ID} \
        -pl ${PLANNER} \
        -c ${FULLRES} 3d_lowres 3d_cascade_fullres \
        --verify_dataset_integrity
    shared_mark_done "p0_preprocess_all3"
fi
verify_preprocessing "${FULLRES}"

# ─────────────────────────────────────────────
# Phase 0b — Build disease_map.json (shared with sibling jobs)
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
# Phase 1 — Fullres trainings (D1, D2, D3)
# ─────────────────────────────────────────────
echo "================================================================"
echo "Phase 1: Fullres training — ${#FULLRES_TRAINERS[@]} trainers, fold 0"
echo "================================================================"
for TRAINER in "${FULLRES_TRAINERS[@]}"; do
    KEY="p1_fullres_$(sk ${TRAINER})_fold0"
    if is_done "${KEY}"; then
        echo "[SKIP] ${KEY}"
    else
        echo "--- ${KEY} ---"
        nnUNetv2_train ${DATASET_ID} ${FULLRES} 0 \
            -tr ${TRAINER} -p ${PLANS} --npz
        mark_done "${KEY}"
    fi
done

# ─────────────────────────────────────────────
# Phase 1b — Inference on imagesTs
# ─────────────────────────────────────────────
echo "================================================================"
echo "Phase 1b: Inference — fullres trainers on imagesTs"
echo "================================================================"
for i in "${!FULLRES_TRAINERS[@]}"; do
    TRAINER="${FULLRES_TRAINERS[$i]}"
    MODE="${FULLRES_INFER_MODES[$i]}"
    OUT_NAME="$(sk ${TRAINER})"
    KEY="p1b_infer_${OUT_NAME}"
    if is_done "${KEY}"; then
        echo "[SKIP] ${KEY}"
    else
        echo "--- ${KEY}  (mode=${MODE}) ---"
        infer_fullres "${TRAINER}" "${MODE}" "${OUT_NAME}"
        mark_done "${KEY}"
    fi
done

# ─────────────────────────────────────────────
# END
# ─────────────────────────────────────────────
print_footer
