#!/bin/bash
# =============================================================================
#  CHD_Dataset013_Fanwei.sh
#  Dataset013_Fanweidatacleaned — DA5 baseline + cascade, 200 epochs
#
#  Fanwei cleaned clinical data. No disease diagnosis labels → no FiLM,
#  no disease conditioning. Same experiment structure as Dataset020
#  (which combines this data with imageCHD).
#
#  Experiments (5-fold ensemble):
#    1. DA5 fullres baseline          (3d_fullres, folds 0-4)
#    2. Cascade baseline              (DA5_200e lowres → CascadeFullresBaseline_200e)
#
#  RESUME SUPPORT
#    Each training run creates a .done marker in CKPT_DIR.
#    Resubmit the same script to continue from where it stopped.
#    nnUNet resumes mid-epoch training from its own checkpoint automatically.
#    Checkpoint dir: ${nnUNet_results}/Dataset013_Fanweidatacleaned/.checkpoints/
#
#  Before first submission:
#    mkdir -p /scratch/users/sastocke/nnunet_CHD/logs
# =============================================================================
#SBATCH --job-name=D013-fanwei
#SBATCH --partition=bioe
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sastocke@stanford.edu
#SBATCH --output=/scratch/users/sastocke/nnunet_CHD/logs/D013-fanwei_%j.out
#SBATCH --error=/scratch/users/sastocke/nnunet_CHD/logs/D013-fanwei_%j.err

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
DATASET_ID=13
DATASET_NAME="Dataset013_Fanweidatacleaned"
PLANNER="nnUNetPlannerResEncM"
PLANS="nnUNetResEncUNetMPlans"
FULLRES="3d_fullres"
LOWRES="3d_lowres"
CASCADE="3d_cascade_fullres"
REPO="/scratch/users/sastocke/nnunet_CHD"
IN_DIR="${nnUNet_raw}/${DATASET_NAME}/imagesTs"
PRED_BASE="${nnUNet_results}/${DATASET_NAME}/predictions"
CKPT_DIR="${nnUNet_results}/${DATASET_NAME}/.checkpoints/CHD_Dataset013_Fanwei"
START_TS=$(date +%s)

LR_TRAINER="nnUNetTrainerDA5_200epochs"
CASCADE_TRAINER="nnUNetTrainerDA5CascadeFullresBaseline_200epochs"

# ─────────────────────────────────────────────
# 3.  Checkpoint helpers
# ─────────────────────────────────────────────
mkdir -p "${CKPT_DIR}"
mark_done() { touch "${CKPT_DIR}/${1}.done"; }
is_done()   { [[ -f "${CKPT_DIR}/${1}.done" ]]; }

# verify_preprocessing CONFIG
#   Counts .b2nd files for CONFIG and errors if any training case is missing.
#   Call this after the p0 block (even on skip) to catch silent worker failures.
verify_preprocessing() {
    local cfg=$1
    local n_raw n_prep prep_dir
    n_raw=$(ls "${nnUNet_raw}/${DATASET_NAME}/imagesTr/" | grep -c "_0000")
    prep_dir=$(find "${nnUNet_preprocessed}/${DATASET_NAME}" -maxdepth 1 -type d -name "*_${cfg}" 2>/dev/null | head -1)
    if [[ -z "${prep_dir}" ]]; then
        echo "ERROR: No preprocessed directory found for ${cfg}."
        echo "  Looked in: ${nnUNet_preprocessed}/${DATASET_NAME}/*_${cfg}"
        echo "  Fix: nnUNetv2_preprocess -d ${DATASET_ID} -pl ${PLANNER} -c ${cfg}"
        echo "  Then delete: ${CKPT_DIR}/p0_preprocess.done and resubmit."
        exit 1
    fi
    n_prep=$(find "${prep_dir}" -maxdepth 1 \( -name "*.b2nd" -o -name "*.npy" \) 2>/dev/null | wc -l)
    echo "[VERIFY] ${cfg}: ${n_prep}/${n_raw} cases in ${prep_dir}"
    if [[ ${n_prep} -lt ${n_raw} ]]; then
        echo "ERROR: Missing preprocessed files for ${cfg} (${n_prep}/${n_raw})."
        echo "  Fix: nnUNetv2_preprocess -d ${DATASET_ID} -pl ${PLANNER} -c ${cfg}"
        echo "  Then delete: ${CKPT_DIR}/p0_preprocess.done and resubmit."
        exit 1
    fi
}

# ─────────────────────────────────────────────
# 4.  Banner helpers
# ─────────────────────────────────────────────
print_banner() {
    echo ""
    echo "╔══════════════════════════════════════════════════════════════════╗"
    echo "║  CHD_Dataset013_Fanwei.sh  — START                             ║"
    echo "╠══════════════════════════════════════════════════════════════════╣"
    printf "║  %-66s ║\n" "Date/Time    : $(date '+%Y-%m-%d %H:%M:%S')"
    printf "║  %-66s ║\n" "SLURM Job    : ${SLURM_JOB_ID:-manual}  node=${SLURMD_NODENAME:-local}"
    printf "║  %-66s ║\n" "Dataset      : ${DATASET_NAME}  (ID=${DATASET_ID})"
    printf "║  %-66s ║\n" "Plans        : ${PLANS}"
    printf "║  %-66s ║\n" "Configs      : ${FULLRES} | ${LOWRES} | ${CASCADE}"
    printf "║  %-66s ║\n" "Folds        : 0 1 2 3 4  (5-fold ensemble — no disease labels)"
    printf "║  %-66s ║\n" "Epochs       : 200"
    printf "║  %-66s ║\n" ""
    printf "║  %-66s ║\n" "Exp 1 — fullres : ${LR_TRAINER}"
    printf "║  %-66s ║\n" "Exp 2 — cascade : ${LR_TRAINER} (lowres)"
    printf "║  %-66s ║\n" "               -> ${CASCADE_TRAINER}"
    printf "║  %-66s ║\n" ""
    printf "║  %-66s ║\n" "Raw data     : ${IN_DIR}"
    printf "║  %-66s ║\n" "Results      : ${nnUNet_results}/${DATASET_NAME}"
    printf "║  %-66s ║\n" "Inference    : ${PRED_BASE}"
    printf "║  %-66s ║\n" "Checkpoint   : ${CKPT_DIR}"
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
    echo "║  CHD_Dataset013_Fanwei.sh  — COMPLETE                          ║"
    echo "╠══════════════════════════════════════════════════════════════════╣"
    printf "║  %-66s ║\n" "Date/Time    : $(date '+%Y-%m-%d %H:%M:%S')"
    printf "║  %-66s ║\n" "SLURM Job    : ${SLURM_JOB_ID:-manual}"
    printf "║  %-66s ║\n" "Dataset      : ${DATASET_NAME}  (ID=${DATASET_ID})"
    printf "║  %-66s ║\n" "Elapsed      : ${hh}h ${mm}m ${ss}s"
    printf "║  %-66s ║\n" ""
    printf "║  %-66s ║\n" "Fullres preds: ${PRED_BASE}/DA5_fullres_ensemble"
    printf "║  %-66s ║\n" "Cascade preds: ${PRED_BASE}/CascadeFullresBaseline_200e_ensemble"
    printf "║  %-66s ║\n" ""
    printf "║  %-66s ║\n" "Next step:"
    printf "║  %-66s ║\n" "  nnUNetv2_find_best_configuration ${DATASET_ID} \\"
    printf "║  %-66s ║\n" "    -c ${FULLRES} ${CASCADE} -tr ${LR_TRAINER} -p ${PLANS}"
    echo "╚══════════════════════════════════════════════════════════════════╝"
}

# ─────────────────────────────────────────────
# START
# ─────────────────────────────────────────────
print_banner

# ─────────────────────────────────────────────
# Phase 0 — Plan and preprocess
# ─────────────────────────────────────────────
if is_done "p0_preprocess"; then
    echo "[SKIP] Phase 0: preprocess already done"
else
    echo "================================================================"
    echo "Phase 0: plan_and_preprocess — ${FULLRES} | ${LOWRES} | ${CASCADE}"
    echo "================================================================"
    nnUNetv2_plan_and_preprocess \
        -d ${DATASET_ID} \
        -pl ${PLANNER} \
        -c ${FULLRES} ${LOWRES} ${CASCADE} \
        --verify_dataset_integrity
    mark_done "p0_preprocess"
fi
verify_preprocessing "${FULLRES}"
verify_preprocessing "${LOWRES}"

# ─────────────────────────────────────────────
# Phase 1 — DA5 fullres baseline (Experiment 1)
# ─────────────────────────────────────────────
echo "================================================================"
echo "Phase 1: Fullres DA5 baseline — 5 folds"
echo "================================================================"
for FOLD in 0 1 2 3 4; do
    KEY="p1_fullres_DA5_200e_fold${FOLD}"
    if is_done "${KEY}"; then
        echo "[SKIP] ${KEY}"
    else
        echo "--- ${KEY} ---"
        nnUNetv2_train ${DATASET_ID} ${FULLRES} ${FOLD} \
            -tr ${LR_TRAINER} -p ${PLANS} --npz
        mark_done "${KEY}"
    fi
done

# ─────────────────────────────────────────────
# Phase 2 — Lowres training (for cascade)
# ─────────────────────────────────────────────
echo "================================================================"
echo "Phase 2: Lowres training — 5 folds"
echo "================================================================"
for FOLD in 0 1 2 3 4; do
    KEY="p2_lowres_DA5_200e_fold${FOLD}"
    if is_done "${KEY}"; then
        echo "[SKIP] ${KEY}"
    else
        echo "--- ${KEY} ---"
        nnUNetv2_train ${DATASET_ID} ${LOWRES} ${FOLD} \
            -tr ${LR_TRAINER} -p ${PLANS} --npz
        mark_done "${KEY}"
    fi
done

# ─────────────────────────────────────────────
# Phase 3 — Symlink predicted_next_stage
# ─────────────────────────────────────────────
if is_done "p3_symlink"; then
    echo "[SKIP] Phase 3: symlink already done"
else
    echo "================================================================"
    echo "Phase 3: Symlink lowres predictions into cascade directory"
    echo "================================================================"
    python "${REPO}/scripts/setup_cascade_predictions.py" \
        --lowres_trainer  "${LR_TRAINER}" \
        --cascade_trainer "${CASCADE_TRAINER}" \
        --dataset         "${DATASET_NAME}" \
        --plans           "${PLANS}" \
        --folds           0 1 2 3 4
    mark_done "p3_symlink"
fi

# ─────────────────────────────────────────────
# Phase 4 — Cascade fullres training
# ─────────────────────────────────────────────
echo "================================================================"
echo "Phase 4: Cascade fullres training — 5 folds"
echo "================================================================"
for FOLD in 0 1 2 3 4; do
    KEY="p4_cascade_CascadeBaseline_200e_fold${FOLD}"
    if is_done "${KEY}"; then
        echo "[SKIP] ${KEY}"
    else
        echo "--- ${KEY} ---"
        nnUNetv2_train ${DATASET_ID} ${CASCADE} ${FOLD} \
            -tr ${CASCADE_TRAINER} -p ${PLANS} --npz
        mark_done "${KEY}"
    fi
done

# ─────────────────────────────────────────────
# Phase 5 — Inference on test set (5-fold ensemble)
# ─────────────────────────────────────────────
echo "================================================================"
echo "Phase 5: Inference on test set — 5-fold ensemble"
echo "================================================================"
mkdir -p "${PRED_BASE}"

FULLRES_PRED="${PRED_BASE}/DA5_fullres_ensemble"
LR_PRED="${PRED_BASE}/DA5_lowres_ensemble"
CASCADE_PRED="${PRED_BASE}/CascadeFullresBaseline_200e_ensemble"
mkdir -p "${FULLRES_PRED}" "${LR_PRED}" "${CASCADE_PRED}"

if is_done "p5_infer_fullres"; then
    echo "[SKIP] p5_infer_fullres"
else
    echo "--- p5_infer_fullres ---"
    nnUNetv2_predict \
        -i "${IN_DIR}" -o "${FULLRES_PRED}" \
        -d ${DATASET_ID} -c ${FULLRES} \
        -f 0 1 2 3 4 \
        -tr ${LR_TRAINER} -p ${PLANS}
    mark_done "p5_infer_fullres"
fi

if is_done "p5_infer_cascade_lowres"; then
    echo "[SKIP] p5_infer_cascade_lowres"
else
    echo "--- p5_infer_cascade_lowres ---"
    nnUNetv2_predict \
        -i "${IN_DIR}" -o "${LR_PRED}" \
        -d ${DATASET_ID} -c ${LOWRES} \
        -f 0 1 2 3 4 \
        -tr ${LR_TRAINER} -p ${PLANS}
    mark_done "p5_infer_cascade_lowres"
fi

if is_done "p5_infer_cascade"; then
    echo "[SKIP] p5_infer_cascade"
else
    echo "--- p5_infer_cascade ---"
    nnUNetv2_predict \
        -i "${IN_DIR}" -o "${CASCADE_PRED}" \
        -d ${DATASET_ID} -c ${CASCADE} \
        -f 0 1 2 3 4 \
        -tr ${CASCADE_TRAINER} -p ${PLANS} \
        -prev_stage_predictions "${LR_PRED}"
    mark_done "p5_infer_cascade"
fi

# ─────────────────────────────────────────────
# END
# ─────────────────────────────────────────────
print_footer
