#!/bin/bash
# =============================================================================
#  CHD_Dataset030_imageCHD.sh
#  Dataset030_imageCHD_HU — 3 experiments, fold 0, 200 epochs
#
#  Experiments
#    1. DA5 fullres baseline          (3d_fullres only)
#    2. Cascade baseline              (DA5_200e lowres  → CascadeFullresBaseline_200e)
#    3. Cascade topology              (DA5CascadeTopo_200e → CascadeFullresTopo_200e)
#
#  Execution order (inference immediately follows its training phase so
#  partial runs still yield usable test-set predictions):
#
#    Phase 0   — plan_and_preprocess (all 3 configs)
#    Phase 0b  — build disease_map.json
#    Phase 1   — train fullres DA5 (fold 0)
#    Phase 1b  — infer fullres DA5 on imagesTs          ← first payoff
#    Phase 2   — train lowres (2 trainers, fold 0)
#    Phase 2b  — infer lowres on imagesTs               ← used as prev_stage below
#    Phase 2.5 — generate predicted_next_stage for ALL training cases
#    Phase 3   — symlink predicted_next_stage into cascade trainer dirs
#    Phase 4   — train cascade fullres (2 trainers, fold 0)
#    Phase 4b  — infer cascade fullres on imagesTs      ← second payoff
#
#  RESUME SUPPORT
#    Each step writes a .done marker. On resubmission the script skips
#    completed steps. nnUNet resumes from its own checkpoint if training
#    was interrupted mid-epoch.
#    Checkpoint dir: ${nnUNet_results}/Dataset030_imageCHD_HU/.checkpoints/CHD_Dataset030_imageCHD/
#
#  Before first submission:
#    mkdir -p /scratch/users/sastocke/nnunet_CHD/logs
# =============================================================================
#SBATCH --job-name=D030-imageCHD
#SBATCH --partition=bioe
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sastocke@stanford.edu
#SBATCH --output=/scratch/users/sastocke/nnunet_CHD/logs/D030-imageCHD_%j.out
#SBATCH --error=/scratch/users/sastocke/nnunet_CHD/logs/D030-imageCHD_%j.err

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
LOWRES="3d_lowres"
CASCADE="3d_cascade_fullres"
REPO="/scratch/users/sastocke/nnunet_CHD"
IN_DIR="${nnUNet_raw}/${DATASET_NAME}/imagesTs"
PRED_BASE="${nnUNet_results}/${DATASET_NAME}/predictions"
CKPT_DIR="${nnUNet_results}/${DATASET_NAME}/.checkpoints/CHD_Dataset030_imageCHD"
SHARED_CKPT_DIR="${nnUNet_results}/${DATASET_NAME}/.checkpoints/shared"
START_TS=$(date +%s)

# Parallel arrays: lowres trainer[i] pairs with cascade trainer[i]
LOWRES_TRAINERS=(
    "nnUNetTrainerDA5_200epochs"
    "nnUNetTrainerDA5CascadeTopo_200epochs"
)
CASCADE_TRAINERS=(
    "nnUNetTrainerDA5CascadeFullresBaseline_200epochs"
    "nnUNetTrainerDA5CascadeFullresTopo_200epochs"
)

# ─────────────────────────────────────────────
# 3.  Checkpoint helpers
# ─────────────────────────────────────────────
mkdir -p "${CKPT_DIR}" "${SHARED_CKPT_DIR}"

# Strip nnUNetTrainer prefix and _200epochs suffix for compact checkpoint keys
sk() { echo "$1" | sed 's/nnUNetTrainer//' | sed 's/_200epochs/200e/'; }

mark_done() { touch "${CKPT_DIR}/${1}.done"; }
is_done()   { [[ -f "${CKPT_DIR}/${1}.done" ]]; }

# Shared markers: preprocessing and disease_map are dataset-level steps.
# All Dataset030 scripts share the same dir to avoid concurrent re-runs.
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
    echo "║  CHD_Dataset030_imageCHD.sh  — START                           ║"
    echo "╠══════════════════════════════════════════════════════════════════╣"
    printf "║  %-66s ║\n" "Date/Time  : $(date '+%Y-%m-%d %H:%M:%S')"
    printf "║  %-66s ║\n" "SLURM Job  : ${SLURM_JOB_ID:-manual}  node=${SLURMD_NODENAME:-local}"
    printf "║  %-66s ║\n" "Dataset    : ${DATASET_NAME}  (ID=${DATASET_ID})"
    printf "║  %-66s ║\n" "Plans      : ${PLANS}"
    printf "║  %-66s ║\n" "Configs    : ${FULLRES} | ${LOWRES} | ${CASCADE}"
    printf "║  %-66s ║\n" "Fold       : 0 only"
    printf "║  %-66s ║\n" "Epochs     : 200"
    printf "║  %-66s ║\n" ""
    printf "║  %-66s ║\n" "Exp 1  DA5 fullres         train → infer"
    printf "║  %-66s ║\n" "Exp 2  DA5 lowres          train → infer → cascade train → infer"
    printf "║  %-66s ║\n" "Exp 3  DA5CascadeTopo      train → infer → cascade train → infer"
    printf "║  %-66s ║\n" ""
    printf "║  %-66s ║\n" "Raw data   : ${IN_DIR}"
    printf "║  %-66s ║\n" "Results    : ${nnUNet_results}/${DATASET_NAME}"
    printf "║  %-66s ║\n" "Inference  : ${PRED_BASE}"
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
    echo "║  CHD_Dataset030_imageCHD.sh  — COMPLETE                        ║"
    echo "╠══════════════════════════════════════════════════════════════════╣"
    printf "║  %-66s ║\n" "Date/Time  : $(date '+%Y-%m-%d %H:%M:%S')"
    printf "║  %-66s ║\n" "SLURM Job  : ${SLURM_JOB_ID:-manual}"
    printf "║  %-66s ║\n" "Dataset    : ${DATASET_NAME}  (ID=${DATASET_ID})"
    printf "║  %-66s ║\n" "Elapsed    : ${hh}h ${mm}m ${ss}s"
    printf "║  %-66s ║\n" ""
    printf "║  %-66s ║\n" "Inference results:"
    printf "║  %-66s ║\n" "  ${PRED_BASE}/DA5_fullres"
    printf "║  %-66s ║\n" "  ${PRED_BASE}/DA5200e_lowres"
    printf "║  %-66s ║\n" "  ${PRED_BASE}/DA5CascadeFullresBaseline200e"
    printf "║  %-66s ║\n" "  ${PRED_BASE}/DA5CascadeTopo200e_lowres"
    printf "║  %-66s ║\n" "  ${PRED_BASE}/DA5CascadeFullresTopo200e"
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
if shared_is_done "p0_preprocess"; then
    echo "[SKIP] Phase 0: preprocess already done (shared marker)"
else
    echo "================================================================"
    echo "Phase 0: plan_and_preprocess — ${FULLRES} | ${LOWRES} | ${CASCADE}"
    echo "================================================================"
    nnUNetv2_plan_and_preprocess \
        -d ${DATASET_ID} \
        -pl ${PLANNER} \
        -c ${FULLRES} ${LOWRES} ${CASCADE} \
        --verify_dataset_integrity
    shared_mark_done "p0_preprocess"
fi
verify_preprocessing "${FULLRES}"
verify_preprocessing "${LOWRES}"

# ─────────────────────────────────────────────
# Phase 0b — Build disease_map.json
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
# Phase 1 — DA5 fullres training (Experiment 1)
# ─────────────────────────────────────────────
echo "================================================================"
echo "Phase 1: Fullres DA5 training — fold 0"
echo "================================================================"
KEY="p1_fullres_DA5200e_fold0"
if is_done "${KEY}"; then
    echo "[SKIP] ${KEY}"
else
    echo "--- ${KEY} ---"
    nnUNetv2_train ${DATASET_ID} ${FULLRES} 0 \
        -tr nnUNetTrainerDA5_200epochs -p ${PLANS} --npz
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
        -tr nnUNetTrainerDA5_200epochs -p ${PLANS}
    mark_done "p1b_infer_fullres"
fi

# ─────────────────────────────────────────────
# Phase 2 — Lowres training (Experiments 2 and 3)
# ─────────────────────────────────────────────
echo "================================================================"
echo "Phase 2: Lowres training — 2 trainers, fold 0"
echo "================================================================"
for LR_TRAINER in "${LOWRES_TRAINERS[@]}"; do
    KEY="p2_lowres_$(sk ${LR_TRAINER})_fold0"
    if is_done "${KEY}"; then
        echo "[SKIP] ${KEY}"
    else
        echo "--- ${KEY} ---"
        nnUNetv2_train ${DATASET_ID} ${LOWRES} 0 \
            -tr ${LR_TRAINER} -p ${PLANS} --npz
        mark_done "${KEY}"
    fi
done

# ─────────────────────────────────────────────
# Phase 2b — Inference: lowres on test set
# (stored now; reused as prev_stage in Phase 4b)
# ─────────────────────────────────────────────
echo "================================================================"
echo "Phase 2b: Inference — lowres trainers on imagesTs"
echo "================================================================"
for LR_TRAINER in "${LOWRES_TRAINERS[@]}"; do
    LR_PRED="${PRED_BASE}/$(sk ${LR_TRAINER})_lowres"
    KEY="p2b_infer_lowres_$(sk ${LR_TRAINER})"
    if is_done "${KEY}"; then
        echo "[SKIP] ${KEY}"
    else
        echo "--- ${KEY} ---"
        mkdir -p "${LR_PRED}"
        nnUNetv2_predict \
            -i "${IN_DIR}" -o "${LR_PRED}" \
            -d ${DATASET_ID} -c ${LOWRES} \
            -f 0 \
            -tr ${LR_TRAINER} -p ${PLANS}
        mark_done "${KEY}"
    fi
done

# ─────────────────────────────────────────────
# Phase 2.5 — Generate predicted_next_stage for ALL training cases
# (perform_actual_validation only saves the ~7 fold-0 val cases;
#  cascade-fullres training needs all 73 training+val cases)
# ─────────────────────────────────────────────
echo "================================================================"
echo "Phase 2.5: Generate cascade next-stage preds for all training cases"
echo "================================================================"
for LR_TRAINER in "${LOWRES_TRAINERS[@]}"; do
    KEY="p2c_cascadepreds_$(sk ${LR_TRAINER})_fold0"
    if is_done "${KEY}"; then
        echo "[SKIP] ${KEY}"
    else
        echo "--- ${KEY} ---"
        python "${REPO}/scripts/generate_cascade_preds.py" \
            --dataset-id ${DATASET_ID} \
            --lowres-trainer ${LR_TRAINER} \
            --plans ${PLANS} \
            --fold 0
        mark_done "${KEY}"
    fi
done

# ─────────────────────────────────────────────
# Phase 3 — Symlink predicted_next_stage into cascade trainer dirs
# ─────────────────────────────────────────────
echo "================================================================"
echo "Phase 3: Symlink lowres predictions into cascade directories"
echo "================================================================"
for i in "${!LOWRES_TRAINERS[@]}"; do
    KEY="p3_symlink_$(sk ${LOWRES_TRAINERS[$i]})_to_$(sk ${CASCADE_TRAINERS[$i]})"
    if is_done "${KEY}"; then
        echo "[SKIP] ${KEY}"
    else
        echo "--- ${KEY} ---"
        python "${REPO}/scripts/setup_cascade_predictions.py" \
            --lowres_trainer  "${LOWRES_TRAINERS[$i]}" \
            --cascade_trainer "${CASCADE_TRAINERS[$i]}" \
            --dataset         "${DATASET_NAME}" \
            --plans           "${PLANS}" \
            --folds           0
        mark_done "${KEY}"
    fi
done

# ─────────────────────────────────────────────
# Phase 4 — Cascade fullres training (Experiments 2 and 3)
# ─────────────────────────────────────────────
echo "================================================================"
echo "Phase 4: Cascade fullres training — 2 trainers, fold 0"
echo "================================================================"
for CASCADE_TRAINER in "${CASCADE_TRAINERS[@]}"; do
    KEY="p4_cascade_$(sk ${CASCADE_TRAINER})_fold0"
    if is_done "${KEY}"; then
        echo "[SKIP] ${KEY}"
    else
        echo "--- ${KEY} ---"
        nnUNetv2_train ${DATASET_ID} ${CASCADE} 0 \
            -tr ${CASCADE_TRAINER} -p ${PLANS} --npz
        mark_done "${KEY}"
    fi
done

# ─────────────────────────────────────────────
# Phase 4b — Inference: cascade fullres on test set
# (uses lowres test predictions from Phase 2b as prev_stage)
# ─────────────────────────────────────────────
echo "================================================================"
echo "Phase 4b: Inference — cascade fullres on imagesTs"
echo "================================================================"
for i in "${!CASCADE_TRAINERS[@]}"; do
    CASCADE_TRAINER="${CASCADE_TRAINERS[$i]}"
    LR_TRAINER="${LOWRES_TRAINERS[$i]}"
    LR_PRED="${PRED_BASE}/$(sk ${LR_TRAINER})_lowres"
    CASCADE_PRED="${PRED_BASE}/$(sk ${CASCADE_TRAINER})"
    KEY="p4b_infer_cascade_$(sk ${CASCADE_TRAINER})"
    if is_done "${KEY}"; then
        echo "[SKIP] ${KEY}"
    else
        echo "--- ${KEY} ---"
        mkdir -p "${CASCADE_PRED}"
        nnUNetv2_predict \
            -i "${IN_DIR}" -o "${CASCADE_PRED}" \
            -d ${DATASET_ID} -c ${CASCADE} \
            -f 0 \
            -tr ${CASCADE_TRAINER} -p ${PLANS} \
            -prev_stage_predictions "${LR_PRED}"
        mark_done "${KEY}"
    fi
done

# ─────────────────────────────────────────────
# END
# ─────────────────────────────────────────────
print_footer
