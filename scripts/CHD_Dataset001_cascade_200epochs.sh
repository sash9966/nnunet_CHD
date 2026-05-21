#!/bin/bash
# =============================================================================
#  CHD_Dataset001_cascade_200epochs.sh
#  Dataset001_all_imageCHD — Cascade ablation, updated to 200 epochs
#
#  4 cascade experiments (lowres trainer → cascade fullres trainer):
#    DA5_200e              → CascadeFullresBaseline_200e    (baseline)
#    DA5CascadeFiLM_200e   → CascadeFullresFiLM_200e        (disease conditioning)
#    DA5CascadeTopo_200e   → CascadeFullresTopo_200e         (topology loss)
#    DA5CascadeFiLMTopo_200e → CascadeFullresFiLMTopo_200e  (conditioning + topology)
#
#  FiLM trainers require disease_map.json in:
#    ${nnUNet_preprocessed}/Dataset001_all_imageCHD/
#  Topology loss (soft-clDice) targets labels named "AO" and "PA".
#
#  RESUME SUPPORT
#    Each training run creates a .done marker. Resubmitting picks up from
#    the last completed step. nnUNet resumes mid-epoch training automatically
#    from its own checkpoint_latest.pth.
#    Checkpoint dir: ${nnUNet_results}/Dataset001_all_imageCHD/.checkpoints/
#    To inspect progress: ls ${CKPT_DIR}/*.done
#
#  Before first submission:
#    mkdir -p /scratch/users/sastocke/nnunet_CHD/logs
# =============================================================================
#SBATCH --job-name=D001-cascade
#SBATCH --partition=bioe
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sastocke@stanford.edu
#SBATCH --output=/scratch/users/sastocke/nnunet_CHD/logs/D001-cascade_%j.out
#SBATCH --error=/scratch/users/sastocke/nnunet_CHD/logs/D001-cascade_%j.err

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
DATASET_ID=1
DATASET_NAME="Dataset001_all_imageCHD"
PLANS="nnUNetResEncUNetMPlans"
LOWRES="3d_lowres"
CASCADE="3d_cascade_fullres"
REPO="/scratch/users/sastocke/nnunet_CHD"
IN_DIR="${nnUNet_raw}/${DATASET_NAME}/imagesTs"
PRED_BASE="${nnUNet_results}/${DATASET_NAME}/predictions"
CKPT_DIR="${nnUNet_results}/${DATASET_NAME}/.checkpoints/CHD_Dataset001_cascade_200epochs"
START_TS=$(date +%s)

# Parallel arrays: LOWRES_TRAINERS[i] pairs with CASCADE_TRAINERS[i]
LOWRES_TRAINERS=(
    "nnUNetTrainerDA5_200epochs"
    "nnUNetTrainerDA5CascadeFiLM_200epochs"
    "nnUNetTrainerDA5CascadeTopo_200epochs"
    "nnUNetTrainerDA5CascadeFiLMTopo_200epochs"
)
CASCADE_TRAINERS=(
    "nnUNetTrainerDA5CascadeFullresBaseline_200epochs"
    "nnUNetTrainerDA5CascadeFullresFiLM_200epochs"
    "nnUNetTrainerDA5CascadeFullresTopo_200epochs"
    "nnUNetTrainerDA5CascadeFullresFiLMTopo_200epochs"
)

# ─────────────────────────────────────────────
# 3.  Checkpoint helpers
# ─────────────────────────────────────────────
mkdir -p "${CKPT_DIR}"

sk() { echo "$1" | sed 's/nnUNetTrainer//' | sed 's/_200epochs/200e/'; }
mark_done() { touch "${CKPT_DIR}/${1}.done"; }
is_done()   { [[ -f "${CKPT_DIR}/${1}.done" ]]; }

verify_preprocessing() {
    local cfg=$1
    local n_raw n_prep prep_dir
    n_raw=$(ls "${nnUNet_raw}/${DATASET_NAME}/imagesTr/" | grep -c "_0000")
    prep_dir=$(find "${nnUNet_preprocessed}/${DATASET_NAME}" -maxdepth 1 -type d -name "*_${cfg}" 2>/dev/null | head -1)
    if [[ -z "${prep_dir}" ]]; then
        echo "ERROR: No preprocessed directory found for ${cfg}."
        echo "  Looked in: ${nnUNet_preprocessed}/${DATASET_NAME}/*_${cfg}"
        echo "  Fix: nnUNetv2_preprocess -d ${DATASET_ID} -plans_name ${PLANS} -c ${cfg}"
        echo "  Then delete: ${CKPT_DIR}/p0_preprocess.done and resubmit."
        exit 1
    fi
    n_prep=$(find "${prep_dir}" -maxdepth 1 \( -name "*.b2nd" -o -name "*.npy" \) 2>/dev/null | wc -l)
    echo "[VERIFY] ${cfg}: ${n_prep}/${n_raw} cases in ${prep_dir}"
    if [[ ${n_prep} -lt ${n_raw} ]]; then
        echo "ERROR: Missing preprocessed files for ${cfg} (${n_prep}/${n_raw})."
        echo "  Fix: nnUNetv2_preprocess -d ${DATASET_ID} -plans_name ${PLANS} -c ${cfg}"
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
    echo "║  CHD_Dataset001_cascade_200epochs.sh  — START                  ║"
    echo "╠══════════════════════════════════════════════════════════════════╣"
    printf "║  %-66s ║\n" "Date/Time  : $(date '+%Y-%m-%d %H:%M:%S')"
    printf "║  %-66s ║\n" "SLURM Job  : ${SLURM_JOB_ID:-manual}  node=${SLURMD_NODENAME:-local}"
    printf "║  %-66s ║\n" "Dataset    : ${DATASET_NAME}  (ID=${DATASET_ID})"
    printf "║  %-66s ║\n" "Plans      : ${PLANS}"
    printf "║  %-66s ║\n" "Configs    : ${LOWRES} | ${CASCADE}"
    printf "║  %-66s ║\n" "Folds      : 0 1 2 3 4"
    printf "║  %-66s ║\n" "Epochs     : 200"
    printf "║  %-66s ║\n" ""
    printf "║  %-66s ║\n" "Cascade pairs (lowres → fullres):"
    for i in "${!LOWRES_TRAINERS[@]}"; do
        printf "║  %-66s ║\n" "  $(sk ${LOWRES_TRAINERS[$i]}) → $(sk ${CASCADE_TRAINERS[$i]})"
    done
    printf "║  %-66s ║\n" ""
    printf "║  %-66s ║\n" "Raw data   : ${IN_DIR}"
    printf "║  %-66s ║\n" "Results    : ${nnUNet_results}/${DATASET_NAME}"
    printf "║  %-66s ║\n" "Inference  : ${PRED_BASE}  (fold 0)"
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
    echo "║  CHD_Dataset001_cascade_200epochs.sh  — COMPLETE               ║"
    echo "╠══════════════════════════════════════════════════════════════════╣"
    printf "║  %-66s ║\n" "Date/Time  : $(date '+%Y-%m-%d %H:%M:%S')"
    printf "║  %-66s ║\n" "SLURM Job  : ${SLURM_JOB_ID:-manual}"
    printf "║  %-66s ║\n" "Dataset    : ${DATASET_NAME}  (ID=${DATASET_ID})"
    printf "║  %-66s ║\n" "Elapsed    : ${hh}h ${mm}m ${ss}s"
    printf "║  %-66s ║\n" ""
    printf "║  %-66s ║\n" "Inference results (fold 0):"
    for i in "${!CASCADE_TRAINERS[@]}"; do
        printf "║  %-66s ║\n" "  ${PRED_BASE}/$(sk ${CASCADE_TRAINERS[$i]})/fold_0"
    done
    printf "║  %-66s ║\n" "Checkpoint : ${CKPT_DIR}"
    echo "╚══════════════════════════════════════════════════════════════════╝"
}

# ─────────────────────────────────────────────
# START
# ─────────────────────────────────────────────
print_banner

# ─────────────────────────────────────────────
# Phase 0 — Preprocess lowres + cascade configs
# (Plans assumed to exist from previous fullres training on Dataset001)
# ─────────────────────────────────────────────
if is_done "p0_preprocess"; then
    echo "[SKIP] Phase 0: preprocess already done"
else
    echo "================================================================"
    echo "Phase 0: Preprocess ${LOWRES} + ${CASCADE}"
    echo "================================================================"
    nnUNetv2_preprocess \
        -d ${DATASET_ID} \
        -plans_name ${PLANS} \
        -c ${LOWRES} ${CASCADE} \
        -n 4 2
    mark_done "p0_preprocess"
fi
verify_preprocessing "${LOWRES}"

# ─────────────────────────────────────────────
# Phase 1 — Lowres training (all 4 trainers x 5 folds)
# ─────────────────────────────────────────────
echo "================================================================"
echo "Phase 1: Lowres training — 4 trainers x 5 folds"
echo "================================================================"
for LR_TRAINER in "${LOWRES_TRAINERS[@]}"; do
    for FOLD in 0 1 2 3 4; do
        KEY="p1_lowres_$(sk ${LR_TRAINER})_fold${FOLD}"
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
# Phase 2 — Symlink predicted_next_stage for all trainer pairs
# ─────────────────────────────────────────────
echo "================================================================"
echo "Phase 2: Symlink lowres predictions into cascade directories"
echo "================================================================"
for i in "${!LOWRES_TRAINERS[@]}"; do
    KEY="p2_symlink_$(sk ${LOWRES_TRAINERS[$i]})_to_$(sk ${CASCADE_TRAINERS[$i]})"
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
# Phase 3 — Cascade fullres training (all 4 trainers x 5 folds)
# ─────────────────────────────────────────────
echo "================================================================"
echo "Phase 3: Cascade fullres training — 4 trainers x 5 folds"
echo "================================================================"
for CASCADE_TRAINER in "${CASCADE_TRAINERS[@]}"; do
    for FOLD in 0 1 2 3 4; do
        KEY="p3_cascade_$(sk ${CASCADE_TRAINER})_fold${FOLD}"
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
# Phase 4 — Inference on test set (fold 0)
# FiLM trainers use predict_disease_conditioned; others use nnUNetv2_predict.
# To use full 5-fold ensemble, change -f 0 to -f 0 1 2 3 4 in all predict calls.
# ─────────────────────────────────────────────
echo "================================================================"
echo "Phase 4: Inference on test set (fold 0)"
echo "================================================================"
mkdir -p "${PRED_BASE}"

LR_MODEL_BASE="${nnUNet_results}/${DATASET_NAME}"

for i in "${!LOWRES_TRAINERS[@]}"; do
    LR_TRAINER="${LOWRES_TRAINERS[$i]}"
    CASCADE_TRAINER="${CASCADE_TRAINERS[$i]}"
    LR_MODEL="${LR_MODEL_BASE}/${LR_TRAINER}__${PLANS}__${LOWRES}"
    CASCADE_MODEL="${LR_MODEL_BASE}/${CASCADE_TRAINER}__${PLANS}__${CASCADE}"
    LR_PRED="${PRED_BASE}/$(sk ${LR_TRAINER})/fold_0"
    CASCADE_PRED="${PRED_BASE}/$(sk ${CASCADE_TRAINER})/fold_0"
    mkdir -p "${LR_PRED}" "${CASCADE_PRED}"

    KEY_LR="p4_infer_lowres_$(sk ${LR_TRAINER})"
    if is_done "${KEY_LR}"; then
        echo "[SKIP] ${KEY_LR}"
    else
        echo "--- ${KEY_LR} ---"
        if [[ "${LR_TRAINER}" == *"FiLM"* ]]; then
            python -m nnunetv2.inference.predict_disease_conditioned \
                -i "${IN_DIR}" -o "${LR_PRED}" \
                -m "${LR_MODEL}" -f 0
        else
            nnUNetv2_predict \
                -i "${IN_DIR}" -o "${LR_PRED}" \
                -d ${DATASET_ID} -c ${LOWRES} \
                -f 0 -tr ${LR_TRAINER} -p ${PLANS}
        fi
        mark_done "${KEY_LR}"
    fi

    KEY_CASCADE="p4_infer_cascade_$(sk ${CASCADE_TRAINER})"
    if is_done "${KEY_CASCADE}"; then
        echo "[SKIP] ${KEY_CASCADE}"
    else
        echo "--- ${KEY_CASCADE} ---"
        if [[ "${CASCADE_TRAINER}" == *"FiLM"* ]]; then
            python -m nnunetv2.inference.predict_disease_conditioned \
                -i "${IN_DIR}" -o "${CASCADE_PRED}" \
                -m "${CASCADE_MODEL}" -f 0 \
                --prev_stage_predictions "${LR_PRED}"
        else
            nnUNetv2_predict \
                -i "${IN_DIR}" -o "${CASCADE_PRED}" \
                -d ${DATASET_ID} -c ${CASCADE} \
                -f 0 -tr ${CASCADE_TRAINER} -p ${PLANS} \
                -prev_stage_predictions "${LR_PRED}"
        fi
        mark_done "${KEY_CASCADE}"
    fi
done

# ─────────────────────────────────────────────
# END
# ─────────────────────────────────────────────
print_footer
