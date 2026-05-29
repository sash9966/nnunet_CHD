#!/bin/bash
# =============================================================================
#  CHD_Dataset040_wholeheart.sh
#  Dataset040_WH_ImageCHD_HU_Detail — Binary whole-heart segmentation, fold 0, 200 epochs
#
#  Stage 1 of the whole-heart-first pipeline. Trains three binary heart models
#  (fullres / lowres / cascade) and runs imagesTs inference for each.  These
#  predictions become the input to any Stage-2 decomposition method later.
#
#  Experiments
#    1. DA5 fullres baseline       (3d_fullres)
#    2. DA5 lowres                 (3d_lowres)  → cascade prev_stage
#    3. DA5 cascade fullres        (3d_cascade_fullres)
#
#  Execution order (inference immediately follows its training so partial runs
#  still yield usable test-set predictions):
#
#    Phase 0    plan_and_preprocess (all 3 configs)
#    Phase 0b   build disease_map.json (for stratified eval; not used for conditioning)
#    Phase 1    train fullres DA5 (fold 0)
#    Phase 1b   infer fullres DA5 on imagesTs
#    Phase 2    train lowres DA5 (fold 0)
#    Phase 2b   infer lowres DA5 on imagesTs
#    Phase 2.5  generate predicted_next_stage for ALL training cases
#    Phase 3    symlink predicted_next_stage into cascade trainer dir
#    Phase 4    train cascade fullres DA5 (fold 0)
#    Phase 4b   infer cascade fullres DA5 on imagesTs
#
#  RESUME SUPPORT
#    - Each step writes a .done marker; resubmission skips completed steps.
#    - nnU-Net auto-resumes interrupted training from checkpoint_latest.pth.
#    - Preprocess + disease_map markers in SHARED_CKPT_DIR so re-conversion of
#      Dataset040 doesn't repeat them.
#    Local checkpoint dir: ${nnUNet_results}/Dataset040_WH_ImageCHD_HU_Detail/.checkpoints/CHD_Dataset040_wholeheart/
#
#  Before first submission:
#    1.  Run the conversion script:
#          python scripts/convert_imagechd_to_wholeheart.py
#    2.  mkdir -p /scratch/users/sastocke/nnunet_CHD/logs
# =============================================================================
#SBATCH --job-name=D040-wholeheart
#SBATCH --partition=bioe
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sastocke@stanford.edu
#SBATCH --output=/scratch/users/sastocke/nnunet_CHD/logs/D040-wholeheart_%j.out
#SBATCH --error=/scratch/users/sastocke/nnunet_CHD/logs/D040-wholeheart_%j.err

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
DATASET_ID=40
DATASET_NAME="Dataset040_WH_ImageCHD_HU_Detail"
PLANNER="nnUNetPlannerResEncM"
PLANS="nnUNetResEncUNetMPlans"
FULLRES="3d_fullres"
LOWRES="3d_lowres"
CASCADE="3d_cascade_fullres"
TRAINER="nnUNetTrainerDA5_200epochs"
CASCADE_TRAINER="nnUNetTrainerDA5_200epochs"   # same class; output folder distinguished by config
REPO="/scratch/users/sastocke/nnunet_CHD"
IN_DIR="${nnUNet_raw}/${DATASET_NAME}/imagesTs"
PRED_BASE="${nnUNet_results}/${DATASET_NAME}/predictions_wholeheart"
CKPT_DIR="${nnUNet_results}/${DATASET_NAME}/.checkpoints/CHD_Dataset040_wholeheart"
SHARED_CKPT_DIR="${nnUNet_results}/${DATASET_NAME}/.checkpoints/shared"
START_TS=$(date +%s)

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
    echo "║  CHD_Dataset040_wholeheart.sh  — START                          ║"
    echo "╠══════════════════════════════════════════════════════════════════╣"
    printf "║  %-66s ║\n" "Date/Time  : $(date '+%Y-%m-%d %H:%M:%S')"
    printf "║  %-66s ║\n" "SLURM Job  : ${SLURM_JOB_ID:-manual}  node=${SLURMD_NODENAME:-local}"
    printf "║  %-66s ║\n" "Dataset    : ${DATASET_NAME}  (ID=${DATASET_ID})"
    printf "║  %-66s ║\n" "Plans      : ${PLANS}"
    printf "║  %-66s ║\n" "Configs    : ${FULLRES} | ${LOWRES} | ${CASCADE}"
    printf "║  %-66s ║\n" "Fold       : 0  |  Epochs: 200  |  Trainer: ${TRAINER}"
    printf "║  %-66s ║\n" ""
    printf "║  %-66s ║\n" "Exp 1  DA5 fullres binary heart   (3d_fullres)"
    printf "║  %-66s ║\n" "Exp 2  DA5 lowres binary heart    (3d_lowres)"
    printf "║  %-66s ║\n" "Exp 3  DA5 cascade binary heart   (3d_cascade_fullres)"
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
    echo "║  CHD_Dataset040_wholeheart.sh  — END                            ║"
    echo "╠══════════════════════════════════════════════════════════════════╣"
    printf "║  %-66s ║\n" "Date/Time  : $(date '+%Y-%m-%d %H:%M:%S')"
    printf "║  %-66s ║\n" "Elapsed    : ${hh}h ${mm}m ${ss}s"
    printf "║  %-66s ║\n" ""
    printf "║  %-66s ║\n" "Inference outputs in ${PRED_BASE}/"
    printf "║  %-66s ║\n" "  DA5_fullres/         Stage-1 fullres heart masks"
    printf "║  %-66s ║\n" "  DA5_lowres/          Stage-1 lowres heart masks"
    printf "║  %-66s ║\n" "  DA5_cascade/         Stage-1 cascade-refined heart masks"
    printf "║  %-66s ║\n" ""
    printf "║  %-66s ║\n" "If incomplete: resubmit this script — it resumes automatically"
    echo "╚══════════════════════════════════════════════════════════════════╝"
}

# ─────────────────────────────────────────────
# START
# ─────────────────────────────────────────────
print_banner
mkdir -p "${PRED_BASE}"

# ─────────────────────────────────────────────
# Phase 0 — Plan and preprocess
# ─────────────────────────────────────────────
_prep_check="${nnUNet_preprocessed}/${DATASET_NAME}/${PLANS}_${FULLRES}"
# Guard the find: running it on a non-existent dir returns exit 1, which under
# set -euo pipefail propagates through the command substitution and kills the
# script SILENTLY (no error message). Only run find when the dir exists.
if [[ -d "${_prep_check}" ]]; then
    _n_prep_check=$(find "${_prep_check}" -maxdepth 1 -name "*_image.b2nd" 2>/dev/null | wc -l)
else
    _n_prep_check=0
fi
if [[ ${_n_prep_check} -gt 0 ]]; then
    echo "[SKIP] Phase 0: ${_n_prep_check} cases already in ${PLANS}_${FULLRES}"
    shared_mark_done "p0_preprocess_all3"
else
    echo "================================================================"
    echo "Phase 0: plan_and_preprocess — ${FULLRES} | ${LOWRES} | ${CASCADE}"
    echo "================================================================"
    nnUNetv2_plan_and_preprocess \
        -d ${DATASET_ID} \
        -pl ${PLANNER} \
        -c ${FULLRES} ${LOWRES} ${CASCADE} \
        --verify_dataset_integrity
    shared_mark_done "p0_preprocess_all3"
fi
verify_preprocessing "${FULLRES}"
verify_preprocessing "${LOWRES}"

# ─────────────────────────────────────────────
# Phase 0b — Build disease_map.json for Dataset040 (stratified eval only)
# ─────────────────────────────────────────────
if shared_is_done "p0b_disease_map_d40"; then
    echo "[SKIP] Phase 0b: disease_map.json already built (shared marker)"
else
    echo "================================================================"
    echo "Phase 0b: Build disease_map.json (for stratified eval)"
    echo "================================================================"
    # The xlsx lives next to Dataset030's raw folder; case IDs are the same so
    # the same map is valid for Dataset040.  make_disease_map.py auto-locates
    # the xlsx via the dataset ID's raw folder.
    python "${REPO}/scripts/make_disease_map.py" \
        --dataset-id ${DATASET_ID} \
        --csv-name imageCHD_dataset_info.xlsx || \
    echo "[WARN] disease_map.json build failed — stratified eval will be unavailable. Continuing."
    shared_mark_done "p0b_disease_map_d40"
fi

# ─────────────────────────────────────────────
# Phase 1 — Fullres DA5 training
# ─────────────────────────────────────────────
echo "================================================================"
echo "Phase 1: Fullres DA5 training (binary heart) — fold 0"
echo "================================================================"
KEY="p1_fullres_$(sk ${TRAINER})_fold0"
if is_done "${KEY}"; then
    echo "[SKIP] ${KEY}"
else
    echo "--- ${KEY} ---"
    nnUNetv2_train ${DATASET_ID} ${FULLRES} 0 \
        -tr ${TRAINER} -p ${PLANS} --npz
    mark_done "${KEY}"
fi

# ─────────────────────────────────────────────
# Phase 1b — Inference: fullres DA5 on test set
# ─────────────────────────────────────────────
echo "================================================================"
echo "Phase 1b: Inference — fullres DA5 on imagesTs"
echo "================================================================"
if is_done "p1b_infer_fullres"; then
    echo "[SKIP] p1b_infer_fullres"
else
    echo "--- p1b_infer_fullres ---"
    mkdir -p "${PRED_BASE}/DA5_fullres"
    nnUNetv2_predict \
        -i "${IN_DIR}" -o "${PRED_BASE}/DA5_fullres" \
        -d ${DATASET_ID} -c ${FULLRES} \
        -f 0 \
        -tr ${TRAINER} -p ${PLANS}
    mark_done "p1b_infer_fullres"
fi

# ─────────────────────────────────────────────
# Phase 2 — Lowres DA5 training
# ─────────────────────────────────────────────
echo "================================================================"
echo "Phase 2: Lowres DA5 training (binary heart) — fold 0"
echo "================================================================"
KEY="p2_lowres_$(sk ${TRAINER})_fold0"
if is_done "${KEY}"; then
    echo "[SKIP] ${KEY}"
else
    echo "--- ${KEY} ---"
    nnUNetv2_train ${DATASET_ID} ${LOWRES} 0 \
        -tr ${TRAINER} -p ${PLANS} --npz
    mark_done "${KEY}"
fi

# ─────────────────────────────────────────────
# Phase 2b — Inference: lowres on test set
# ─────────────────────────────────────────────
echo "================================================================"
echo "Phase 2b: Inference — lowres DA5 on imagesTs"
echo "================================================================"
LR_PRED="${PRED_BASE}/DA5_lowres"
if is_done "p2b_infer_lowres"; then
    echo "[SKIP] p2b_infer_lowres"
else
    echo "--- p2b_infer_lowres ---"
    mkdir -p "${LR_PRED}"
    nnUNetv2_predict \
        -i "${IN_DIR}" -o "${LR_PRED}" \
        -d ${DATASET_ID} -c ${LOWRES} \
        -f 0 \
        -tr ${TRAINER} -p ${PLANS}
    mark_done "p2b_infer_lowres"
fi

# ─────────────────────────────────────────────
# Phase 2.5 — Generate predicted_next_stage for ALL training cases
# ─────────────────────────────────────────────
echo "================================================================"
echo "Phase 2.5: Generate cascade next-stage preds for all training cases"
echo "================================================================"
KEY="p2c_cascadepreds_$(sk ${TRAINER})_fold0"
if is_done "${KEY}"; then
    echo "[SKIP] ${KEY}"
else
    echo "--- ${KEY} ---"
    python "${REPO}/scripts/generate_cascade_preds.py" \
        --dataset-id ${DATASET_ID} \
        --lowres-trainer ${TRAINER} \
        --plans ${PLANS} \
        --fold 0
    mark_done "${KEY}"
fi

# ─────────────────────────────────────────────
# Phase 3 — Symlink predicted_next_stage into cascade trainer dir
# (lowres trainer class == cascade trainer class; setup script handles the
# same-class case by gracefully no-op'ing since auto-resolution works.)
# ─────────────────────────────────────────────
echo "================================================================"
echo "Phase 3: Symlink lowres predictions into cascade directory"
echo "================================================================"
KEY="p3_symlink_$(sk ${TRAINER})_to_$(sk ${CASCADE_TRAINER})"
if is_done "${KEY}"; then
    echo "[SKIP] ${KEY}"
else
    echo "--- ${KEY} ---"
    python "${REPO}/scripts/setup_cascade_predictions.py" \
        --lowres_trainer  "${TRAINER}" \
        --cascade_trainer "${CASCADE_TRAINER}" \
        --dataset         "${DATASET_NAME}" \
        --plans           "${PLANS}" \
        --folds           0
    mark_done "${KEY}"
fi

# ─────────────────────────────────────────────
# Phase 3.5 — Verify cascade prev-stage predictions before training
# ─────────────────────────────────────────────
# The 3d_cascade_fullres trainer reads previous-stage segmentations from
#   <CASCADE_TRAINER>__<plans>__3d_lowres/predicted_next_stage/3d_cascade_fullres/
# (this is nnUNetTrainer.folder_with_segs_from_previous_stage). It needs one
# .b2nd per training case; a shortfall here is what makes Phase 4 fail. Verify
# loudly with the resolved path so the cause is never a mystery.
echo "================================================================"
echo "Phase 3.5: Verify cascade prev-stage predictions"
echo "================================================================"
NEXT_STAGE_DIR="${nnUNet_results}/${DATASET_NAME}/${CASCADE_TRAINER}__${PLANS}__${LOWRES}/predicted_next_stage/${CASCADE}"
LOWRES_PREP="${nnUNet_preprocessed}/${DATASET_NAME}/${PLANS}_${LOWRES}"
if [[ -d "${LOWRES_PREP}" ]]; then
    _n_cases=$(find "${LOWRES_PREP}" -maxdepth 1 -name "*_image.b2nd" 2>/dev/null | wc -l)
else
    _n_cases=0
fi
if [[ -d "${NEXT_STAGE_DIR}" ]]; then
    _n_preds=$(find "${NEXT_STAGE_DIR}" -maxdepth 1 -name "*.b2nd" 2>/dev/null | wc -l)
else
    _n_preds=0
fi
echo "[VERIFY] prev-stage preds: ${_n_preds}/${_n_cases} training cases"
echo "         read path: ${NEXT_STAGE_DIR}"
if [[ "${_n_cases}" -lt 1 ]]; then
    echo "ERROR: no lowres preprocessed cases found in ${LOWRES_PREP}"
    exit 1
fi
if [[ "${_n_preds}" -lt "${_n_cases}" ]]; then
    echo "ERROR: only ${_n_preds}/${_n_cases} cascade prev-stage predictions present."
    echo "  The 3d_cascade_fullres trainer needs one per training case. Phase 2.5"
    echo "  (generate_cascade_preds.py) likely errored or was interrupted — scroll up"
    echo "  for its traceback. To re-run it, clear the markers and resubmit:"
    echo "    rm -f ${CKPT_DIR}/p2c_cascadepreds_$(sk ${TRAINER})_fold0.done"
    echo "    rm -f ${CKPT_DIR}/p3_symlink_$(sk ${TRAINER})_to_$(sk ${CASCADE_TRAINER}).done"
    exit 1
fi
echo "[VERIFY] OK — all ${_n_cases} prev-stage predictions present"

# ─────────────────────────────────────────────
# Phase 4 — Cascade fullres DA5 training
# ─────────────────────────────────────────────
echo "================================================================"
echo "Phase 4: Cascade fullres DA5 training (binary heart) — fold 0"
echo "================================================================"
KEY="p4_cascade_$(sk ${CASCADE_TRAINER})_fold0"
if is_done "${KEY}"; then
    echo "[SKIP] ${KEY}"
else
    echo "--- ${KEY} ---"
    nnUNetv2_train ${DATASET_ID} ${CASCADE} 0 \
        -tr ${CASCADE_TRAINER} -p ${PLANS} --npz
    mark_done "${KEY}"
fi

# ─────────────────────────────────────────────
# Phase 4b — Inference: cascade fullres on test set
# (uses lowres test predictions from Phase 2b as prev_stage)
# ─────────────────────────────────────────────
echo "================================================================"
echo "Phase 4b: Inference — cascade fullres DA5 on imagesTs"
echo "================================================================"
CASCADE_PRED="${PRED_BASE}/DA5_cascade"
if is_done "p4b_infer_cascade"; then
    echo "[SKIP] p4b_infer_cascade"
else
    echo "--- p4b_infer_cascade ---"
    mkdir -p "${CASCADE_PRED}"
    nnUNetv2_predict \
        -i "${IN_DIR}" -o "${CASCADE_PRED}" \
        -d ${DATASET_ID} -c ${CASCADE} \
        -f 0 \
        -tr ${CASCADE_TRAINER} -p ${PLANS} \
        -prev_stage_predictions "${LR_PRED}"
    mark_done "p4b_infer_cascade"
fi

# ─────────────────────────────────────────────
# END
# ─────────────────────────────────────────────
print_footer
