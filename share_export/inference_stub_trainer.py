"""
Inference stub trainer for sharing case-weighted DA5 models to a STOCK nnU-Net / Slicer.

nnU-Net reads the trainer name from inside the checkpoint and must be able to import that class.
`nnUNetTrainerDA5CaseWeighted*` differs from stock DA5 ONLY in training-time case sampling and epoch
count — the network (stock ResEncUNet) and everything used at inference are identical. So a trivial
subclass of stock `nnUNetTrainerDA5` is enough to make the name resolvable and run the weights.

How to use on the collaborator's machine (no fork install):
  1. Copy this file anywhere under their nnU-Net trainer tree, e.g.
     <site-packages>/nnunetv2/training/nnUNetTrainer/  (nnU-Net recursively scans that folder), OR
     set the env var  nnUNet_extTrainer=<folder containing this file>.
  2. Predict as usual with the checkpoint AS-IS (no retag needed):
     nnUNetv2_predict -i IN -o OUT -d <ID> -c 3d_fullres \
        -tr nnUNetTrainerDA5CaseWeighted_500epochs -p nnUNetResEncUNetMPlans -f 0 -chk checkpoint_best.pth

Alternative that needs NO file on their side: retag the checkpoint's trainer_name to nnUNetTrainerDA5
(share_export/retag_checkpoint_to_stock.py). Either path works; pick one.
"""
from nnunetv2.training.nnUNetTrainer.variants.data_augmentation.nnUNetTrainerDA5 import nnUNetTrainerDA5


class nnUNetTrainerDA5CaseWeighted_500epochs(nnUNetTrainerDA5):
    """Inference-only stub: identical network to stock DA5 (weighting is train-time only)."""
    pass


class nnUNetTrainerDA5CaseWeighted_200epochs(nnUNetTrainerDA5):
    """Inference-only stub for the 200-epoch variant."""
    pass
