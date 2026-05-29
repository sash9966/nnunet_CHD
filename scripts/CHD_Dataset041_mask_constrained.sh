#!/bin/bash
# =============================================================================
#  CHD_Dataset041_mask_constrained.sh
#  Dataset041_ImageCHD_HU_MaskCond — 2-channel 7-class (CT + binary heart prior)
#                                    Stage-2 Approach A, fold 0, 200 epochs
#
#  Three trainers, ranked by ablation intent:
#    Exp 1  DA5 baseline                          (3d_fullres)
#    Exp 2  DA5 + scheduled TopologyLoss (AO+PA)  (3d_fullres)
#    Exp 3  DA5 + FiLM disease conditioning + Topo (3d_fullres)
#
#  Reads:
#    - Dataset041 must already exist (build it with
#      experiments/wholeheart_decomposition/mask_constrained_nnunet/convert_to_mask_conditioned.py)
#    - disease_map.json gets auto-built for FiLM Exp 3 (Phase 0b)
#
#  Phases
#    0    plan_and_preprocess (3d_fullres only — no cascade for now)
#    0b   build disease_map.json
#    1    train Exp 1 (DA5)                  → infer on imagesTs
#    2    train Exp 2 (DA5 + TopoScheduled)  → infer on imagesTs
#    3    train Exp 3 (DA5 + FiLM + Topo)    → infer on imagesTs  (disease-conditioned)
#
#  RESUME
#    Each phase writes a .done marker; resubmission skips completed phases.
#    Preprocess + disease_map markers go in SHARED_CKPT_DIR.
#    nnU-Net auto-resumes interrupted training from checkpoint_latest.pth.
#
#  Before first submission:
#    1.  python experiments/wholeheart_decomposition/mask_constrained_nnunet/convert_to_mask_conditioned.py
#        (default --mask-source gt — no Stage-1 dependency)
#    2.  mkdir -p /scratch/users/sastocke/nnunet_CHD/logs
# =============================================================================
#SBATCH --job-name=D041-maskcond
#SBATCH --partition=bioe
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sastocke@stanford.edu
#SBATCH --output=/scratch/users/sastocke/nnunet_CHD/logs/D041-maskcond_%j.out
#SBATCH --error=/scratch/users/sastocke/nnunet_CHD/logs/D041-maskcond_%j.err

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
DATASET_ID=41
DATASET_NAME="Dataset041_ImageCHD_HU_MaskCond"
PLANNER="nnUNetPlannerResEncM"
PLANS="nnUNetResEncUNetMPlans"
FULLRES="3d_fullres"

TRAINER_BASE="nnUNetTrainerDA5_200epochs"
TRAINER_TOPO="nnUNetTrainerDA5TopoScheduled_200epochs"
TRAINER_FILM="nnUNetTrainerDA5FiLMTopo_200epochs"

REPO="/scratch/users/sastocke/nnunet_CHD"
IN_DIR="${nnUNet_raw}/${DATASET_NAME}/imagesTs"
PRED_BASE="${nnUNet_results}/${DATASET_NAME}/predictions_stage2"
CKPT_DIR="${nnUNet_results}/${DATASET_NAME}/.checkpoints/CHD_Dataset041_mask_constrained"
SHARED_CKPT_DIR="${nnUNet_results}/${DATASET_NAME}/.checkpoints/shared"
START_TS=$(date +%s)

# ─────────────────────────────────────────────
# 3.  Checkpoint helpers
# ─────────────────────────────────────────────
mkdir -p "${CKPT_DIR}" "${SHARED_CKPT_DIR}" "${PRED_BASE}"

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
        echo "ERROR: No preprocessed directory for ${cfg}."
        exit 1
    fi
    n_prep=$(find "${prep_dir}" -maxdepth 1 -name "*_image.b2nd" 2>/dev/null | wc -l)
    echo "[VERIFY] ${cfg}: ${n_prep}/${n_raw} cases in ${prep_dir}"
    if [[ ${n_prep} -lt ${n_raw} ]]; then
        echo "ERROR: Missing preprocessed files for ${cfg} (${n_prep}/${n_raw})."
        exit 1
    fi
}

# ─────────────────────────────────────────────
# 4.  Banner helpers
# ─────────────────────────────────────────────
print_banner() {
    echo ""
    echo "╔══════════════════════════════════════════════════════════════════╗"
    echo "║  CHD_Dataset041_mask_constrained.sh  — START                    ║"
    echo "╠══════════════════════════════════════════════════════════════════╣"
    printf "║  %-66s ║\n" "Date/Time  : $(date '+%Y-%m-%d %H:%M:%S')"
    printf "║  %-66s ║\n" "SLURM Job  : ${SLURM_JOB_ID:-manual}  node=${SLURMD_NODENAME:-local}"
    printf "║  %-66s ║\n" "Dataset    : ${DATASET_NAME}  (ID=${DATASET_ID})"
    printf "║  %-66s ║\n" "Plans      : ${PLANS}   Config: ${FULLRES}  Fold: 0  Epochs: 200"
    printf "║  %-66s ║\n" ""
    printf "║  %-66s ║\n" "Exp 1  DA5 baseline                  : ${TRAINER_BASE}"
    printf "║  %-66s ║\n" "Exp 2  DA5 + scheduled TopoLoss      : ${TRAINER_TOPO}"
    printf "║  %-66s ║\n" "Exp 3  DA5 + FiLM + TopoLoss         : ${TRAINER_FILM}"
    printf "║  %-66s ║\n" ""
    printf "║  %-66s ║\n" "Raw data   : ${IN_DIR}"
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
    echo "║  CHD_Dataset041_mask_constrained.sh  — END                      ║"
    echo "╠══════════════════════════════════════════════════════════════════╣"
    printf "║  %-66s ║\n" "Date/Time  : $(date '+%Y-%m-%d %H:%M:%S')"
    printf "║  %-66s ║\n" "Elapsed    : ${hh}h ${mm}m ${ss}s"
    printf "║  %-66s ║\n" ""
    printf "║  %-66s ║\n" "Inference outputs in ${PRED_BASE}/"
    printf "║  %-66s ║\n" "  exp1_da5/             DA5 baseline preds"
    printf "║  %-66s ║\n" "  exp2_da5_topo/        DA5 + scheduled topo preds"
    printf "║  %-66s ║\n" "  exp3_da5_film_topo/   DA5 + FiLM + topo preds (disease-cond)"
    printf "║  %-66s ║\n" ""
    printf "║  %-66s ║\n" "If incomplete: resubmit this script — it resumes automatically"
    echo "╚══════════════════════════════════════════════════════════════════╝"
}

# ─────────────────────────────────────────────
# START
# ─────────────────────────────────────────────
print_banner

# ─────────────────────────────────────────────
# Phase 0 — Plan and preprocess (fullres only)
# ─────────────────────────────────────────────
if shared_is_done "p0_preprocess_${FULLRES}"; then
    echo "[SKIP] Phase 0: preprocess (${FULLRES}) already done"
else
    echo "================================================================"
    echo "Phase 0: plan_and_preprocess — ${FULLRES}"
    echo "================================================================"
    nnUNetv2_plan_and_preprocess \
        -d ${DATASET_ID} \
        -pl ${PLANNER} \
        -c ${FULLRES} \
        --verify_dataset_integrity
    shared_mark_done "p0_preprocess_${FULLRES}"
fi
verify_preprocessing "${FULLRES}"

# ─────────────────────────────────────────────
# Phase 0b — Build disease_map.json (needed by Exp 3 / FiLM)
# ─────────────────────────────────────────────
if shared_is_done "p0b_disease_map_d41"; then
    echo "[SKIP] Phase 0b: disease_map.json already built"
else
    echo "================================================================"
    echo "Phase 0b: Build disease_map.json (case IDs share with Dataset030)"
    echo "================================================================"
    python "${REPO}/scripts/make_disease_map.py" \
        --dataset-id ${DATASET_ID} \
        --csv-name imageCHD_dataset_info.xlsx || \
    echo "[WARN] disease_map.json build failed — Exp 3 will fail at startup. Continuing."
    shared_mark_done "p0b_disease_map_d41"
fi

# ─────────────────────────────────────────────
# Helper — train + infer one experiment
# ─────────────────────────────────────────────
train_and_infer() {
    local exp_label=$1     # short tag, used for .done keys + pred subdir
    local trainer=$2       # full trainer class name
    local pred_subdir=$3   # e.g. exp1_da5

    local train_key="train_${exp_label}_$(sk ${trainer})_fold0"
    local infer_key="infer_${exp_label}_$(sk ${trainer})_fold0"
    local pred_dir="${PRED_BASE}/${pred_subdir}"

    echo "================================================================"
    echo "${exp_label}: train — ${trainer}  (fold 0)"
    echo "================================================================"
    if is_done "${train_key}"; then
        echo "[SKIP] ${train_key}"
    else
        echo "--- ${train_key} ---"
        nnUNetv2_train ${DATASET_ID} ${FULLRES} 0 \
            -tr ${trainer} -p ${PLANS} --npz
        mark_done "${train_key}"
    fi

    echo "================================================================"
    echo "${exp_label}: infer on imagesTs — ${trainer}"
    echo "================================================================"
    if is_done "${infer_key}"; then
        echo "[SKIP] ${infer_key}"
    else
        echo "--- ${infer_key} ---"
        mkdir -p "${pred_dir}"
        nnUNetv2_predict \
            -i "${IN_DIR}" -o "${pred_dir}" \
            -d ${DATASET_ID} -c ${FULLRES} \
            -f 0 \
            -tr ${trainer} -p ${PLANS}
        mark_done "${infer_key}"
    fi
}

# ─────────────────────────────────────────────
# Phase 1 — Exp 1: DA5 baseline
# ─────────────────────────────────────────────
train_and_infer "p1_exp1" "${TRAINER_BASE}" "exp1_da5"

# ─────────────────────────────────────────────
# Phase 2 — Exp 2: DA5 + scheduled topology
# ─────────────────────────────────────────────
train_and_infer "p2_exp2" "${TRAINER_TOPO}" "exp2_da5_topo"

# ─────────────────────────────────────────────
# Phase 3 — Exp 3: DA5 + FiLM + topology  (disease-conditioned)
#   NOTE: inference here uses the unconditioned `nnUNetv2_predict` entry point,
#   which means FiLM gets a zero-vector disease condition by default.  To run
#   inference with per-case disease conditioning use
#     python nnunetv2/inference/predict_disease_conditioned.py
#   after this script finishes; the trained checkpoint will be reused.
# ─────────────────────────────────────────────
train_and_infer "p3_exp3" "${TRAINER_FILM}" "exp3_da5_film_topo"

# ─────────────────────────────────────────────
# END
# ─────────────────────────────────────────────
print_footer
