#!/bin/bash
# =============================================================================
#  CHD_Dataset030_ablation_topo.sh   (Job 1 of 3)
#  Dataset030_imageCHD_HU — Baselines + Topology cascade hypothesis
#  Fold 0, 200 epochs
#
#  Companion scripts (run independently, in parallel if multiple GPUs):
#    Job 2: CHD_Dataset030_ablation_disease.sh   (D1, D2, D3)
#    Job 3: CHD_Dataset030_ablation_combos.sh    (C1, C2, C3, C4, C5)
#
#  ─── Rows trained here ───────────────────────────────────────────────────
#  ID    Lowres trainer                                Fullres / Cascade trainer
#  ────  ──────────────────────────────────────────    ─────────────────────────────────────────────
#  B1    —                                             DA5_200e                                    (3d_fullres)        CONTROL
#  B2    DA5_200e                                      DA5_200e                                    (3d_lowres → 3d_cascade_fullres)
#  T1    —                                             DA5TopoScheduled_200e                       (3d_fullres)
#  T2    DA5CascadeTopoScheduled_200e                  DA5CascadeFullresBaseline_200e              (3d_lowres → 3d_cascade_fullres)   ← HYPOTHESIS
#  T3    DA5CascadeTopoScheduled_200e (shared w/ T2)   DA5CascadeFullresTopoScheduled_200e         (3d_lowres → 3d_cascade_fullres)
#
#  Total: 2 fullres + 2 lowres + 3 cascade-fullres = 7 trainings, 7 inferences
#
#  RESUME SUPPORT (critical — 200 epochs × 7 trainings > 48h walltime)
#    - Each step writes a .done marker; resubmission skips completed steps.
#    - nnU-Net auto-resumes interrupted training from checkpoint_latest.pth.
#    - Preprocess + disease_map markers live in SHARED_CKPT_DIR so the other
#      two ablation scripts don't re-run them.
#    Local checkpoint dir: ${nnUNet_results}/Dataset030_imageCHD_HU/.checkpoints/CHD_Dataset030_ablation_topo/
#
#  Resubmission:
#    Just run `sbatch scripts/CHD_Dataset030_ablation_topo.sh` again — it
#    picks up from the last completed step automatically.
#
#  Before first submission:
#    mkdir -p /scratch/users/sastocke/nnunet_CHD/logs
# =============================================================================
#SBATCH --job-name=D030-abl-topo
#SBATCH --partition=bioe
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sastocke@stanford.edu
#SBATCH --output=/scratch/users/sastocke/nnunet_CHD/logs/D030-abl-topo_%j.out
#SBATCH --error=/scratch/users/sastocke/nnunet_CHD/logs/D030-abl-topo_%j.err

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
PRED_BASE="${nnUNet_results}/${DATASET_NAME}/predictions_ablation"
CKPT_DIR="${nnUNet_results}/${DATASET_NAME}/.checkpoints/CHD_Dataset030_ablation_topo"
SHARED_CKPT_DIR="${nnUNet_results}/${DATASET_NAME}/.checkpoints/shared"
START_TS=$(date +%s)

# ── Fullres-only trainers (B1, T1) ──
FULLRES_TRAINERS=(
    "nnUNetTrainerDA5_200epochs"                  # B1
    "nnUNetTrainerDA5TopoScheduled_200epochs"     # T1
)

# ── Lowres trainers (B2-lowres, T2/T3-shared-lowres) ──
LOWRES_TRAINERS=(
    "nnUNetTrainerDA5_200epochs"                       # B2 lowres
    "nnUNetTrainerDA5CascadeTopoScheduled_200epochs"   # shared by T2 and T3
)

# ── Cascade-fullres trainings (B2-cascade, T2-cascade, T3-cascade) ──
# Indexed by experiment label so T2 and T3 can share the TopoScheduled lowres.
CASCADE_LOWRES_PAIR=(
    "nnUNetTrainerDA5_200epochs"                       # B2: lowres = DA5
    "nnUNetTrainerDA5CascadeTopoScheduled_200epochs"   # T2: lowres = TopoSched
    "nnUNetTrainerDA5CascadeTopoScheduled_200epochs"   # T3: lowres = TopoSched (shared with T2)
)
CASCADE_TRAINERS=(
    "nnUNetTrainerDA5_200epochs"                              # B2: plain DA5 on cascade config
    "nnUNetTrainerDA5CascadeFullresBaseline_200epochs"        # T2: plain cascade fullres baseline
    "nnUNetTrainerDA5CascadeFullresTopoScheduled_200epochs"   # T3: cascade fullres + scheduled topo
)
CASCADE_LABELS=( "B2" "T2" "T3" )

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
    # Use the exact plans-specific directory (not a wildcard match that can pick up
    # a different planner's folder, e.g. nnUNetPlans_3d_fullres instead of
    # nnUNetResEncUNetMPlans_3d_fullres).
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
    echo "║  CHD_Dataset030_ablation_topo.sh  — START  (Job 1 of 3)         ║"
    echo "╠══════════════════════════════════════════════════════════════════╣"
    printf "║  %-66s ║\n" "Date/Time  : $(date '+%Y-%m-%d %H:%M:%S')"
    printf "║  %-66s ║\n" "SLURM Job  : ${SLURM_JOB_ID:-manual}  node=${SLURMD_NODENAME:-local}"
    printf "║  %-66s ║\n" "Dataset    : ${DATASET_NAME}  (ID=${DATASET_ID})"
    printf "║  %-66s ║\n" "Fold       : 0  |  Epochs: 200"
    printf "║  %-66s ║\n" ""
    printf "║  %-66s ║\n" "Trainings  : 2 fullres + 2 lowres + 3 cascade-fullres = 7"
    printf "║  %-66s ║\n" ""
    printf "║  %-66s ║\n" "B1  DA5_200e                                  (fullres)"
    printf "║  %-66s ║\n" "B2  DA5_200e → DA5_200e                       (cascade)"
    printf "║  %-66s ║\n" "T1  DA5TopoScheduled_200e                     (fullres)"
    printf "║  %-66s ║\n" "T2  DA5CascadeTopoSched → DA5CascadeBaseline  (cascade) HYPO"
    printf "║  %-66s ║\n" "T3  DA5CascadeTopoSched → DA5CascadeTopoSched (cascade)"
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
    echo "║  CHD_Dataset030_ablation_topo.sh  — END  (Job 1 of 3)           ║"
    echo "╠══════════════════════════════════════════════════════════════════╣"
    printf "║  %-66s ║\n" "Date/Time  : $(date '+%Y-%m-%d %H:%M:%S')"
    printf "║  %-66s ║\n" "Elapsed    : ${hh}h ${mm}m ${ss}s"
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
# Phase 0 — Plan and preprocess (all three configs; shared with sibling jobs)
# ─────────────────────────────────────────────
# Check the plans-specific fullres directory for actual .b2nd files rather than
# relying solely on the shared marker (which may have been written by a run that
# used a different planner or an older preprocessing format without .b2nd files).
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
# Phase 1 — Fullres trainings (B1, T1)
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
# Phase 1b — Inference on imagesTs (B1, T1)
# ─────────────────────────────────────────────
echo "================================================================"
echo "Phase 1b: Inference — fullres trainers on imagesTs"
echo "================================================================"
for TRAINER in "${FULLRES_TRAINERS[@]}"; do
    OUT_NAME="$(sk ${TRAINER})"
    OUT_DIR="${PRED_BASE}/${OUT_NAME}"
    KEY="p1b_infer_${OUT_NAME}"
    if is_done "${KEY}"; then
        echo "[SKIP] ${KEY}"
    else
        echo "--- ${KEY} ---"
        mkdir -p "${OUT_DIR}"
        nnUNetv2_predict \
            -i "${IN_DIR}" -o "${OUT_DIR}" \
            -d ${DATASET_ID} -c ${FULLRES} \
            -f 0 \
            -tr ${TRAINER} -p ${PLANS}
        mark_done "${KEY}"
    fi
done

# ─────────────────────────────────────────────
# Phase 2 — Lowres trainings (B2-lowres, T2/T3-shared-lowres)
# ─────────────────────────────────────────────
echo "================================================================"
echo "Phase 2: Lowres training — ${#LOWRES_TRAINERS[@]} trainers, fold 0"
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
# Phase 2b — Inference: lowres on imagesTs (used later as prev_stage)
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
# Phase 3 — Symlink predicted_next_stage into each cascade trainer dir
# ─────────────────────────────────────────────
echo "================================================================"
echo "Phase 3: Symlink lowres predictions into cascade directories"
echo "================================================================"
for i in "${!CASCADE_TRAINERS[@]}"; do
    LABEL="${CASCADE_LABELS[$i]}"
    LR_TRAINER="${CASCADE_LOWRES_PAIR[$i]}"
    C_TRAINER="${CASCADE_TRAINERS[$i]}"
    KEY="p3_symlink_${LABEL}_$(sk ${LR_TRAINER})_to_$(sk ${C_TRAINER})"
    if is_done "${KEY}"; then
        echo "[SKIP] ${KEY}"
    else
        echo "--- ${KEY} ---"
        python "${REPO}/scripts/setup_cascade_predictions.py" \
            --lowres_trainer  "${LR_TRAINER}" \
            --cascade_trainer "${C_TRAINER}" \
            --dataset         "${DATASET_NAME}" \
            --plans           "${PLANS}" \
            --folds           0
        mark_done "${KEY}"
    fi
done

# ─────────────────────────────────────────────
# Phase 4 — Cascade-fullres trainings (B2, T2, T3)
# ─────────────────────────────────────────────
echo "================================================================"
echo "Phase 4: Cascade-fullres training — ${#CASCADE_TRAINERS[@]} trainings, fold 0"
echo "================================================================"
for i in "${!CASCADE_TRAINERS[@]}"; do
    LABEL="${CASCADE_LABELS[$i]}"
    C_TRAINER="${CASCADE_TRAINERS[$i]}"
    KEY="p4_cascade_${LABEL}_$(sk ${C_TRAINER})_fold0"
    if is_done "${KEY}"; then
        echo "[SKIP] ${KEY}"
    else
        echo "--- ${KEY} ---"
        nnUNetv2_train ${DATASET_ID} ${CASCADE} 0 \
            -tr ${C_TRAINER} -p ${PLANS} --npz
        mark_done "${KEY}"
    fi
done

# ─────────────────────────────────────────────
# Phase 4b — Inference: cascade-fullres on imagesTs
# ─────────────────────────────────────────────
echo "================================================================"
echo "Phase 4b: Inference — cascade-fullres trainers on imagesTs"
echo "================================================================"
for i in "${!CASCADE_TRAINERS[@]}"; do
    LABEL="${CASCADE_LABELS[$i]}"
    LR_TRAINER="${CASCADE_LOWRES_PAIR[$i]}"
    C_TRAINER="${CASCADE_TRAINERS[$i]}"
    LR_PRED="${PRED_BASE}/$(sk ${LR_TRAINER})_lowres"
    OUT_NAME="${LABEL}_$(sk ${C_TRAINER})_cascade"
    OUT_DIR="${PRED_BASE}/${OUT_NAME}"
    KEY="p4b_infer_${OUT_NAME}"
    if is_done "${KEY}"; then
        echo "[SKIP] ${KEY}"
    else
        echo "--- ${KEY} ---"
        mkdir -p "${OUT_DIR}"
        nnUNetv2_predict \
            -i "${IN_DIR}" -o "${OUT_DIR}" \
            -d ${DATASET_ID} -c ${CASCADE} \
            -f 0 \
            -tr ${C_TRAINER} -p ${PLANS} \
            -prev_stage_predictions "${LR_PRED}"
        mark_done "${KEY}"
    fi
done

# ─────────────────────────────────────────────
# END
# ─────────────────────────────────────────────
print_footer
