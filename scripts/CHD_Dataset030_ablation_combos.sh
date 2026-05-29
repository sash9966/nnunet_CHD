#!/bin/bash
# =============================================================================
#  CHD_Dataset030_ablation_combos.sh   (Job 3 of 3)
#  Dataset030_imageCHD_HU — Disease-conditioning × topology / aux combinations
#  Fold 0, 200 epochs
#
#  Companion scripts:
#    Job 1: CHD_Dataset030_ablation_topo.sh    (B1, B2, T1, T2, T3)
#    Job 2: CHD_Dataset030_ablation_disease.sh (D1, D2, D3)
#
#  ─── Rows trained here ───────────────────────────────────────────────────
#  ID    Trainer                                    Config         Inference
#  ────  ─────────────────────────────────────────  ─────────────  ─────────────────────────
#  C1    DA5FiLMTopo_200e                           3d_fullres     predict_disease_conditioned
#  C2    DA5AuxDiagTopo_200e                        3d_fullres     nnUNetv2_predict
#  C3    DA5CrossAttnTopo_200e                      3d_fullres     predict_disease_conditioned
#  C4    DA5AuxDiagCrossAttn_200e                   3d_fullres     predict_disease_conditioned
#  C5    DA5FiLMAuxDiag_200e                        3d_fullres     predict_disease_conditioned
#
#  Total: 5 fullres trainings, 5 inferences (~75h, expect ~2 walltime cycles)
#
#  RESUME SUPPORT
#    Same as sibling scripts: .done markers per step, nnU-Net auto-resumes
#    interrupted training, preprocess + disease_map markers live in SHARED.
#    Local checkpoint dir: ${nnUNet_results}/Dataset030_imageCHD_HU/.checkpoints/CHD_Dataset030_ablation_combos/
#
#  Before first submission:
#    mkdir -p /scratch/users/sastocke/nnunet_CHD/logs
# =============================================================================
#SBATCH --job-name=D030-abl-combos
#SBATCH --partition=bioe
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sastocke@stanford.edu
# stdout + stderr merged into ONE file (no separate .err) so every error is visible in one place
#SBATCH --output=/scratch/users/sastocke/nnunet_CHD/logs/D030-abl-combos_%j.log

set -euo pipefail

# ─────────────────────────────────────────────
# 1.  Environment
# ─────────────────────────────────────────────
module purge
module load gcc/12.4.0 cuda/11.7.1 cmake/3.24.2
source /oak/stanford/groups/amarsden/sastocke/miniconda/etc/profile.d/conda.sh
conda activate /scratch/users/sastocke/conda_envs/nnunet310
hash -r

# PROJECT ISOLATION CONTRACT: these roots are EXCLUSIVE to the CHD nnU-Net
# project. Any other project (e.g. MedDINO) MUST use a DIFFERENT
# nnUNet_preprocessed / nnUNet_results, or the two will corrupt each other's
# preprocessed data and plans. This script is a READ-ONLY consumer — it never
# runs plan_and_preprocess; preprocess once up front before submitting jobs.
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
CKPT_DIR="${nnUNet_results}/${DATASET_NAME}/.checkpoints/CHD_Dataset030_ablation_combos"
SHARED_CKPT_DIR="${nnUNet_results}/${DATASET_NAME}/.checkpoints/shared"
START_TS=$(date +%s)

# Parallel arrays: trainer + inference mode ("plain" = nnUNetv2_predict, "cond" = disease-conditioned)
FULLRES_TRAINERS=(
    "nnUNetTrainerDA5FiLMTopo_200epochs"          # C1
    "nnUNetTrainerDA5AuxDiagTopo_200epochs"       # C2
    "nnUNetTrainerDA5CrossAttnTopo_200epochs"     # C3
    "nnUNetTrainerDA5AuxDiagCrossAttn_200epochs"  # C4
    "nnUNetTrainerDA5FiLMAuxDiag_200epochs"       # C5
)
FULLRES_INFER_MODES=(
    "cond"    # C1: FiLM + Topo
    "plain"   # C2: AuxDiag + Topo (no inference conditioning)
    "cond"    # C3: CrossAttn + Topo
    "cond"    # C4: AuxDiag + CrossAttn (CrossAttn drives inference)
    "cond"    # C5: FiLM + AuxDiag (FiLM drives inference)
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
    echo "║  CHD_Dataset030_ablation_combos.sh  — START  (Job 3 of 3)       ║"
    echo "╠══════════════════════════════════════════════════════════════════╣"
    printf "║  %-66s ║\n" "Date/Time  : $(date '+%Y-%m-%d %H:%M:%S')"
    printf "║  %-66s ║\n" "SLURM Job  : ${SLURM_JOB_ID:-manual}  node=${SLURMD_NODENAME:-local}"
    printf "║  %-66s ║\n" "Dataset    : ${DATASET_NAME}  (ID=${DATASET_ID})"
    printf "║  %-66s ║\n" "Fold       : 0  |  Epochs: 200"
    printf "║  %-66s ║\n" ""
    printf "║  %-66s ║\n" "Trainings  : 5 fullres (C1, C2, C3, C4, C5)"
    printf "║  %-66s ║\n" ""
    printf "║  %-66s ║\n" "C1  DA5FiLMTopo_200e        (FiLM + Topo)"
    printf "║  %-66s ║\n" "C2  DA5AuxDiagTopo_200e     (Aux + Topo)"
    printf "║  %-66s ║\n" "C3  DA5CrossAttnTopo_200e   (CrossAttn + Topo)"
    printf "║  %-66s ║\n" "C4  DA5AuxDiagCrossAttn_200e (embedding reuse)"
    printf "║  %-66s ║\n" "C5  DA5FiLMAuxDiag_200e     (FiLM + Aux)"
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
    echo "║  CHD_Dataset030_ablation_combos.sh  — END  (Job 3 of 3)         ║"
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
# Phase 0 — none. This script does NOT preprocess (read-only consumer).
# ─────────────────────────────────────────────
# Preprocess ONCE up front before submitting (it is destructive shared state):
#   nnUNetv2_plan_and_preprocess -d 30 -pl nnUNetPlannerResEncM \
#       -c 3d_fullres 3d_lowres 3d_cascade_fullres --verify_dataset_integrity
# No preprocessed-folder check here on purpose: nnU-Net stores each config under
# its own data_identifier (fullres -> nnUNetPlans_3d_fullres, lowres ->
# nnUNetResEncUNetMPlans_3d_lowres), so guessing the folder name is unreliable
# and was falsely failing valid runs. nnUNetv2_train validates the preprocessed
# data itself and errors clearly if it is genuinely missing.

# ─────────────────────────────────────────────
# Phase 1 — Fullres trainings (C1, C2, C3, C4, C5)
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
