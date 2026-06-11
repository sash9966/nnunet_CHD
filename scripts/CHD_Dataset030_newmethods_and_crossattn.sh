#!/bin/bash
# =============================================================================
#  CHD_Dataset030_newmethods_and_crossattn.sh   (Job 5)
#  Dataset030_imageCHD_HU
#    PART A — train + infer the 3 NEW structural trainers (fold 0, 200 epochs)
#    PART B — re-run inference on the existing CrossAttn checkpoints (diagnostic)
#
#  ─── PART A: new "beat-the-baseline" trainers (3d_fullres, network unchanged) ──
#  ID    Fullres trainer                                       Idea
#  ────  ──────────────────────────────────────────────────    ──────────────────────────────
#  N1    nnUNetTrainerDA5RegionScaffold_200epochs              hierarchical region supervision
#  N2    nnUNetTrainerDA5VesselFocusedTopo_200epochs           binary AO∪PA vessel clDice
#  N3    nnUNetTrainerDA5CenterlineAux_200epochs               centerline-weighted CE
#
#  ─── PART B: CrossAttn re-inference DIAGNOSTIC (no retrain) ────────────────
#  The two broken runs (D3 CrossAttn = 0.001, C3 CrossAttnTopo = 0.000) trained
#  on the SAME block that the WORKING AuxDiagCrossAttn (0.822) used, so the 0.0
#  is most likely the inference / conditioned-forward path, not training.
#  This part re-infers the EXISTING fold-0 checkpoints two ways:
#     *_cond_reinfer  — predict_disease_conditioned (sets disease vec per case)
#     *_plain_reinfer — nnUNetv2_predict            (no conditioning)
#  so we can localise the failure.
#
#  IMPORTANT CAVEAT (read): cross_attention_mixin.py now has the zero-init
#  `gamma` residual gate fix. Old checkpoints have NO `gamma` key, so they load
#  with gamma=0 (strict=False) → each CrossAttnBlock is an identity → the
#  decoder runs on its BASELINE path. Interpretation of this re-inference:
#     • ~0.82 (baseline-ish) → the trained decoder is healthy; the original 0.0
#       came from the old conditioned forward at inference (it WAS the inference).
#     • still ~0.0          → the trained weights themselves are bad
#       (training collapse) → retrain with the fix (see Part C, commented).
#  To reproduce the ORIGINAL (pre-fix) inference behaviour instead, re-run Part B
#  from a checkout where cross_attention_mixin.py predates the gamma fix.
#
#  RESUME: each step writes a .done marker; resubmission skips completed steps.
#
#  ─── ISOLATION + UNPACK-SAFETY CONTRACT ──────────────────────────────────
#    * READ-ONLY consumer: never runs plan_and_preprocess.
#      Preprocess ONCE up front:
#        nnUNetv2_plan_and_preprocess -d 30 -pl nnUNetPlannerResEncM \
#            -c 3d_fullres 3d_lowres 3d_cascade_fullres --verify_dataset_integrity
#    * If .npy aren't already valid, rebuild ONCE, alone, before submitting:
#        python -c "from nnunetv2.training.dataloading.utils import unpack_dataset; \
#          unpack_dataset('${nnUNet_preprocessed}/Dataset030_imageCHD_HU/nnUNetPlans_3d_fullres', \
#          overwrite_existing=True, verify=True)"
#
#  Before first submission:
#    mkdir -p /scratch/users/sastocke/nnunet_CHD/logs
# =============================================================================
#SBATCH --job-name=D030-new-xattn
#SBATCH --partition=bioe
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sastocke@stanford.edu
# stdout + stderr merged into ONE file so every error is visible in one place
#SBATCH --output=/scratch/users/sastocke/nnunet_CHD/logs/D030-new-xattn_%j.log

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
# project. This script is a READ-ONLY consumer — it never runs plan_and_preprocess.
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
PLANS="nnUNetResEncUNetMPlans"
FULLRES="3d_fullres"
IN_DIR="${nnUNet_raw}/${DATASET_NAME}/imagesTs"
PRED_BASE="${nnUNet_results}/${DATASET_NAME}/predictions_ablation"
CKPT_DIR="${nnUNet_results}/${DATASET_NAME}/.checkpoints/CHD_Dataset030_newmethods_and_crossattn"
START_TS=$(date +%s)

# PART A — three new fullres trainers (N1, N2, N3)
NEW_TRAINERS=(
    "nnUNetTrainerDA5RegionScaffold_200epochs"      # N1
    "nnUNetTrainerDA5VesselFocusedTopo_200epochs"   # N2
    "nnUNetTrainerDA5CenterlineAux_200epochs"       # N3
)

# PART B — existing CrossAttn checkpoints to re-infer (the 0.0 runs)
XATTN_TRAINERS=(
    "nnUNetTrainerDA5CrossAttn_200epochs"       # D3 (was 0.001)
    "nnUNetTrainerDA5CrossAttnTopo_200epochs"   # C3 (was 0.000)
)

# ─────────────────────────────────────────────
# 3.  Checkpoint helpers
# ─────────────────────────────────────────────
mkdir -p "${CKPT_DIR}" "${PRED_BASE}"
sk() { echo "$1" | sed 's/nnUNetTrainer//' | sed 's/_200epochs/200e/'; }
mark_done() { touch "${CKPT_DIR}/${1}.done"; }
is_done()   { [[ -f "${CKPT_DIR}/${1}.done" ]]; }

# ─────────────────────────────────────────────
# START
# ─────────────────────────────────────────────
echo "================================================================"
echo "CHD_Dataset030_newmethods_and_crossattn.sh  — START"
echo "  Date       : $(date '+%Y-%m-%d %H:%M:%S')"
echo "  SLURM Job  : ${SLURM_JOB_ID:-manual}  node=${SLURMD_NODENAME:-local}"
echo "  Dataset    : ${DATASET_NAME} (ID=${DATASET_ID})  Fold 0"
echo "  Part A     : train+infer N1 RegionScaffold | N2 VesselFocusedTopo | N3 CenterlineAux (200e)"
echo "  Part B     : re-infer D3 CrossAttn + C3 CrossAttnTopo (cond + plain) — diagnostic"
echo "================================================================"
echo "  Completed steps from previous runs (.done markers):"
ls "${CKPT_DIR}/"*.done 2>/dev/null | xargs -I{} basename {} .done | sort \
    | sed 's/^/    [DONE] /' || echo "    (none — fresh run)"
echo ""

# No preprocessed-folder check here on purpose: nnU-Net validates the
# preprocessed data itself and errors clearly if it is genuinely missing.

# ════════════════════════════════════════════════════════════════
# PART B FIRST — CrossAttn re-inference (fast; gives immediate signal)
# ════════════════════════════════════════════════════════════════
echo "================================================================"
echo "Part B: CrossAttn re-inference diagnostic (existing fold-0 checkpoints)"
echo "================================================================"
for TRAINER in "${XATTN_TRAINERS[@]}"; do
    MODEL_DIR="${nnUNet_results}/${DATASET_NAME}/${TRAINER}__${PLANS}__${FULLRES}"
    if [[ ! -d "${MODEL_DIR}" ]]; then
        echo "[SKIP] ${TRAINER}: model dir not found (${MODEL_DIR}) — was it trained on this root?"
        continue
    fi

    # (b1) conditioned re-inference
    OUT_COND="${PRED_BASE}/$(sk ${TRAINER})_cond_reinfer"
    KEY="b_cond_$(sk ${TRAINER})"
    if is_done "${KEY}"; then
        echo "[SKIP] ${KEY}"
    else
        echo "--- ${KEY}  ->  ${OUT_COND} ---"
        mkdir -p "${OUT_COND}"
        python -m nnunetv2.inference.predict_disease_conditioned \
            -i "${IN_DIR}" -o "${OUT_COND}" -m "${MODEL_DIR}" -f 0
        mark_done "${KEY}"
    fi

    # (b2) plain (unconditioned) re-inference
    OUT_PLAIN="${PRED_BASE}/$(sk ${TRAINER})_plain_reinfer"
    KEY="b_plain_$(sk ${TRAINER})"
    if is_done "${KEY}"; then
        echo "[SKIP] ${KEY}"
    else
        echo "--- ${KEY}  ->  ${OUT_PLAIN} ---"
        mkdir -p "${OUT_PLAIN}"
        nnUNetv2_predict \
            -i "${IN_DIR}" -o "${OUT_PLAIN}" \
            -d ${DATASET_ID} -c ${FULLRES} -f 0 \
            -tr ${TRAINER} -p ${PLANS}
        mark_done "${KEY}"
    fi
done

# ════════════════════════════════════════════════════════════════
# PART A — new structural trainers: train fold 0, then infer
# ════════════════════════════════════════════════════════════════
echo "================================================================"
echo "Part A — Phase 1: Fullres training — ${#NEW_TRAINERS[@]} new trainers, fold 0, 200e"
echo "================================================================"
for TRAINER in "${NEW_TRAINERS[@]}"; do
    KEY="a_train_$(sk ${TRAINER})_fold0"
    if is_done "${KEY}"; then
        echo "[SKIP] ${KEY}"
    else
        echo "--- ${KEY} ---"
        nnUNetv2_train ${DATASET_ID} ${FULLRES} 0 \
            -tr ${TRAINER} -p ${PLANS} --npz
        mark_done "${KEY}"
    fi
done

echo "================================================================"
echo "Part A — Phase 1b: Inference — new trainers on imagesTs"
echo "================================================================"
for TRAINER in "${NEW_TRAINERS[@]}"; do
    OUT_NAME="$(sk ${TRAINER})"
    OUT_DIR="${PRED_BASE}/${OUT_NAME}"
    KEY="a_infer_${OUT_NAME}"
    if is_done "${KEY}"; then
        echo "[SKIP] ${KEY}"
    else
        echo "--- ${KEY} ---"
        mkdir -p "${OUT_DIR}"
        nnUNetv2_predict \
            -i "${IN_DIR}" -o "${OUT_DIR}" \
            -d ${DATASET_ID} -c ${FULLRES} -f 0 \
            -tr ${TRAINER} -p ${PLANS}
        mark_done "${KEY}"
    fi
done

# ════════════════════════════════════════════════════════════════
# PART C (OPTIONAL, commented) — retrain CrossAttn with the gamma fix
# ════════════════════════════════════════════════════════════════
# To validate the gamma-gate fix end-to-end you must train into a FRESH output
# dir (nnU-Net auto-resumes from checkpoint_latest.pth, so retraining the same
# trainer name would RESUME the old broken run). Either move the old results dir
# aside first (ask before any destructive move), or register a new trainer name.
# Then:
#   nnUNetv2_train 30 3d_fullres 0 -tr nnUNetTrainerDA5CrossAttn_200epochs -p ${PLANS} --npz
#   python -m nnunetv2.inference.predict_disease_conditioned -i "${IN_DIR}" \
#       -o "${PRED_BASE}/CrossAttn200e_fixed" \
#       -m "${nnUNet_results}/${DATASET_NAME}/nnUNetTrainerDA5CrossAttn_200epochs__${PLANS}__${FULLRES}" -f 0

# ─────────────────────────────────────────────
# END
# ─────────────────────────────────────────────
ELAPSED=$(( $(date +%s) - START_TS ))
echo ""
echo "================================================================"
echo "CHD_Dataset030_newmethods_and_crossattn.sh  — END"
echo "  Date     : $(date '+%Y-%m-%d %H:%M:%S')"
echo "  Elapsed  : $(( ELAPSED/3600 ))h $(( (ELAPSED%3600)/60 ))m $(( ELAPSED%60 ))s"
echo "  Resubmit this script to resume — completed steps are skipped."
echo ""
echo "  Next — topology-aware comparison vs DA5 baseline:"
echo "    python scripts/evaluate_topology_dataset030.py \\"
echo "       --gt ${nnUNet_preprocessed}/${DATASET_NAME}/gt_segmentations \\"
echo "       --pred DA5_baseline=${PRED_BASE}/DA5200e \\"
echo "              RegionScaffold=${PRED_BASE}/DA5RegionScaffold200e \\"
echo "              VesselFocusedTopo=${PRED_BASE}/DA5VesselFocusedTopo200e \\"
echo "              CenterlineAux=${PRED_BASE}/DA5CenterlineAux200e \\"
echo "              CrossAttn_cond=${PRED_BASE}/DA5CrossAttn200e_cond_reinfer \\"
echo "              CrossAttn_plain=${PRED_BASE}/DA5CrossAttn200e_plain_reinfer \\"
echo "       --dataset_json ${nnUNet_raw}/${DATASET_NAME}/dataset.json \\"
echo "       --out ${nnUNet_results}/${DATASET_NAME}/topology_eval"
echo "================================================================"
