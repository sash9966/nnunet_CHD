#!/bin/bash
# =============================================================================
#  CHD_Dataset030_imageCHD.sh
#  Dataset030_imageCHD_HU — 3 experiments, 5-fold, 200 epochs
#
#  Experiments
#    1. DA5 fullres baseline          (3d_fullres only)
#    2. Cascade baseline              (DA5_200e lowres  → CascadeFullresBaseline_200e)
#    3. Cascade topology              (DA5CascadeTopo_200e → CascadeFullresTopo_200e)
#
#  Topology loss (soft-clDice) applied to AO and PA labels.
#  Prerequisite: Dataset030 dataset.json must name those labels "AO" and "PA".
#
#  RESUME SUPPORT
#    Each training run creates a .done marker. On resubmission the script
#    skips any step whose marker already exists. nnUNet itself resumes
#    from its own checkpoint if training was mid-epoch when interrupted.
#    Checkpoint dir: ${nnUNet_results}/Dataset030_imageCHD_HU/.checkpoints/
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
CKPT_DIR="${nnUNet_results}/${DATASET_NAME}/.checkpoints/$(basename "${BASH_SOURCE[0]}" .sh)"
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
mkdir -p "${CKPT_DIR}"

# Strip nnUNetTrainer prefix and _200epochs suffix for compact checkpoint keys
sk() { echo "$1" | sed 's/nnUNetTrainer//' | sed 's/_200epochs/200e/'; }

mark_done() { touch "${CKPT_DIR}/${1}.done"; }
is_done()   { [[ -f "${CKPT_DIR}/${1}.done" ]]; }

verify_preprocessing() {
    local cfg=$1
    local n_raw n_prep prep_dir
    n_raw=$(ls "${nnUNet_raw}/${DATASET_NAME}/imagesTr/" | grep -c "_0000")
    prep_dir=$(find "${nnUNet_preprocessed}/${DATASET_NAME}" -maxdepth 1 -type d -name "*${cfg}" | head -1)
    n_prep=$(find "${prep_dir}" -maxdepth 1 -name "*.b2nd" 2>/dev/null | wc -l)
    echo "[VERIFY] ${cfg}: ${n_prep}/${n_raw} cases preprocessed"
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
    echo "║  CHD_Dataset030_imageCHD.sh  — START                           ║"
    echo "╠══════════════════════════════════════════════════════════════════╣"
    printf "║  %-66s ║\n" "Date/Time  : $(date '+%Y-%m-%d %H:%M:%S')"
    printf "║  %-66s ║\n" "SLURM Job  : ${SLURM_JOB_ID:-manual}  node=${SLURMD_NODENAME:-local}"
    printf "║  %-66s ║\n" "Dataset    : ${DATASET_NAME}  (ID=${DATASET_ID})"
    printf "║  %-66s ║\n" "Plans      : ${PLANS}"
    printf "║  %-66s ║\n" "Configs    : ${FULLRES} | ${LOWRES} | ${CASCADE}"
    printf "║  %-66s ║\n" "Folds      : 0 1 2 3 4  (5-fold ensemble inference)"
    printf "║  %-66s ║\n" "Epochs     : 200  (via _200epochs trainer variants)"
    printf "║  %-66s ║\n" ""
    printf "║  %-66s ║\n" "Experiment 1 — fullres:  nnUNetTrainerDA5_200epochs"
    printf "║  %-66s ║\n" "Experiment 2 — cascade:  DA5_200e -> CascadeFullresBaseline_200e"
    printf "║  %-66s ║\n" "Experiment 3 — cascade:  DA5CascadeTopo_200e -> CascadeFullresTopo_200e"
    printf "║  %-66s ║\n" ""
    printf "║  %-66s ║\n" "Raw data   : ${IN_DIR}"
    printf "║  %-66s ║\n" "Results    : ${nnUNet_results}/${DATASET_NAME}"
    printf "║  %-66s ║\n" "Inference  : ${PRED_BASE}"
    printf "║  %-66s ║\n" "Checkpoint : ${CKPT_DIR}"
    echo "╚══════════════════════════════════════════════════════════════════╝"
    echo ""
    # Print existing checkpoint status so you can see what already ran
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
    printf "║  %-66s ║\n" "  ${PRED_BASE}/nnUNetTrainerDA5CascadeFullresBaseline_200epochs"
    printf "║  %-66s ║\n" "  ${PRED_BASE}/nnUNetTrainerDA5CascadeFullresTopo_200epochs"
    printf "║  %-66s ║\n" "Checkpoint : ${CKPT_DIR}"
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
    KEY="p1_fullres_$(sk nnUNetTrainerDA5_200epochs)_fold${FOLD}"
    if is_done "${KEY}"; then
        echo "[SKIP] ${KEY}"
    else
        echo "--- ${KEY} ---"
        nnUNetv2_train ${DATASET_ID} ${FULLRES} ${FOLD} \
            -tr nnUNetTrainerDA5_200epochs -p ${PLANS} --npz
        mark_done "${KEY}"
    fi
done

# ─────────────────────────────────────────────
# Phase 2 — Lowres training (Experiments 2 and 3)
# ─────────────────────────────────────────────
echo "================================================================"
echo "Phase 2: Lowres training — 2 trainers x 5 folds"
echo "================================================================"
for LR_TRAINER in "${LOWRES_TRAINERS[@]}"; do
    for FOLD in 0 1 2 3 4; do
        KEY="p2_lowres_$(sk ${LR_TRAINER})_fold${FOLD}"
        if is_done "${KEY}"; then
            echo "[SKIP] ${KEY}"
        else
            echo "--- ${KEY} ---"
            nnUNetv2_train ${DATASET_ID} ${LOWRES} ${FOLD} \
                -tr ${LR_TRAINER} -p ${PLANS} --npz
            mark_done "${KEY}"
        fi
    done
done

# ─────────────────────────────────────────────
# Phase 3 — Symlink predicted_next_stage
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
            --folds           0 1 2 3 4
        mark_done "${KEY}"
    fi
done

# ─────────────────────────────────────────────
# Phase 4 — Cascade fullres training (Experiments 2 and 3)
# ─────────────────────────────────────────────
echo "================================================================"
echo "Phase 4: Cascade fullres training — 2 trainers x 5 folds"
echo "================================================================"
for CASCADE_TRAINER in "${CASCADE_TRAINERS[@]}"; do
    for FOLD in 0 1 2 3 4; do
        KEY="p4_cascade_$(sk ${CASCADE_TRAINER})_fold${FOLD}"
        if is_done "${KEY}"; then
            echo "[SKIP] ${KEY}"
        else
            echo "--- ${KEY} ---"
            nnUNetv2_train ${DATASET_ID} ${CASCADE} ${FOLD} \
                -tr ${CASCADE_TRAINER} -p ${PLANS} --npz
            mark_done "${KEY}"
        fi
    done
done

# ─────────────────────────────────────────────
# Phase 5 — Inference on test set (5-fold ensemble)
# ─────────────────────────────────────────────
echo "================================================================"
echo "Phase 5: Inference on test set — 5-fold ensemble"
echo "================================================================"
mkdir -p "${PRED_BASE}"

# Experiment 1: fullres DA5
if is_done "p5_infer_fullres"; then
    echo "[SKIP] p5_infer_fullres"
else
    echo "--- p5_infer_fullres ---"
    nnUNetv2_predict \
        -i "${IN_DIR}" -o "${PRED_BASE}/DA5_fullres" \
        -d ${DATASET_ID} -c ${FULLRES} \
        -f 0 1 2 3 4 \
        -tr nnUNetTrainerDA5_200epochs -p ${PLANS}
    mark_done "p5_infer_fullres"
fi

# Experiments 2 and 3: cascade (lowres predict → cascade predict)
for i in "${!LOWRES_TRAINERS[@]}"; do
    LR_TRAINER="${LOWRES_TRAINERS[$i]}"
    CASCADE_TRAINER="${CASCADE_TRAINERS[$i]}"
    LR_PRED="${PRED_BASE}/$(sk ${LR_TRAINER})_lowres"
    CASCADE_PRED="${PRED_BASE}/$(sk ${CASCADE_TRAINER})"
    mkdir -p "${LR_PRED}" "${CASCADE_PRED}"

    KEY_LR="p5_infer_lowres_$(sk ${LR_TRAINER})"
    if is_done "${KEY_LR}"; then
        echo "[SKIP] ${KEY_LR}"
    else
        echo "--- ${KEY_LR} ---"
        nnUNetv2_predict \
            -i "${IN_DIR}" -o "${LR_PRED}" \
            -d ${DATASET_ID} -c ${LOWRES} \
            -f 0 1 2 3 4 \
            -tr ${LR_TRAINER} -p ${PLANS}
        mark_done "${KEY_LR}"
    fi

    KEY_CASCADE="p5_infer_cascade_$(sk ${CASCADE_TRAINER})"
    if is_done "${KEY_CASCADE}"; then
        echo "[SKIP] ${KEY_CASCADE}"
    else
        echo "--- ${KEY_CASCADE} ---"
        nnUNetv2_predict \
            -i "${IN_DIR}" -o "${CASCADE_PRED}" \
            -d ${DATASET_ID} -c ${CASCADE} \
            -f 0 1 2 3 4 \
            -tr ${CASCADE_TRAINER} -p ${PLANS} \
            -prev_stage_predictions "${LR_PRED}"
        mark_done "${KEY_CASCADE}"
    fi
done

# ─────────────────────────────────────────────
# END
# ─────────────────────────────────────────────
print_footer
