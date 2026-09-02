#!/bin/bash
# FANWEI — JOB 5: reduce SeqSeg to ONE NIfTI per case/vessel on the CT grid, then STITCH it into the
# nnInteractive mask (vessels unioned; chambers/myo keep priority). Drops the VTK clutter.
#   sbatch scripts/CHD_refine_fanwei_step5_stitch.sh
#SBATCH --job-name=fw-stitch
#SBATCH --partition=bioe
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/users/sastocke/nnunet_CHD/logs/fw-stitch_%j.out
#SBATCH --error=/scratch/users/sastocke/nnunet_CHD/logs/fw-stitch_%j.err
set -euo pipefail
module purge
module load gcc/12.4.0 cuda/11.7.1 cmake/3.24.2
source /oak/stanford/groups/amarsden/sastocke/miniconda/etc/profile.d/conda.sh
REPO=/scratch/users/sastocke/nnunet_CHD; cd "$REPO"
source scripts/_provenance.sh

B=/scratch/users/sastocke/chd_refinement/out
SEQ="${1:-$B/seqseg_fanwei_bif1}"                  # pass .../seqseg_fanwei_bif2 for the deeper tree
NII="${SEQ}_nifti"
NNI="${2:-$B/nninteractive_fanwei/refined}"
MERGED="${3:-$B/refined_fanwei_merged}"
CT="$REPO/nnUNet_raw/Dataset012_Fanweidata/imagesTr"
rm -rf "$NII" "$MERGED"
stamp_provenance "fanwei-stitch" "$MERGED" "seqseg=$SEQ" "nni=$NNI" "ct=$CT"

echo "===== 1/2 SeqSeg -> one NIfTI per case/vessel (needs vtk: chd_seqseg310) ====="
conda activate /scratch/users/sastocke/conda_envs/chd_seqseg310; hash -r
python tools/seqseg_to_nifti.py --seqseg-dir "$SEQ" --ct-dir "$CT" --out-dir "$NII" --vessels Aorta,Pulmonary

echo ""
echo "===== 2/2 stitch nnInteractive + SeqSeg vessels ====="
conda activate /scratch/users/sastocke/conda_envs/nnunet310; hash -r
python tools/merge_refined.py --nninteractive-dir "$NNI" --seqseg-nifti-dir "$NII" \
    --out-dir "$MERGED" --vessels Aorta=6,Pulmonary=7

echo ""
echo "DONE."
echo "  vessel NIfTIs : $NII"
echo "  stitched final: $MERGED"
echo "  (the VTK tree in $SEQ is no longer needed once these look right)"
