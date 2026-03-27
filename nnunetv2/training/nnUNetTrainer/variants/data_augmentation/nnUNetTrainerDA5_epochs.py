"""Epoch-count variants of nnUNetTrainerDA5.

Defined here (rather than modifying nnUNetTrainerDA5.py) to keep the base file
unchanged.  Only the 100-epoch variant is added; the 10-epoch variant already
exists in nnUNetTrainerDA5.py as ``nnUNetTrainerDA5_10epochs``.
"""
import torch

from nnunetv2.training.nnUNetTrainer.variants.data_augmentation.nnUNetTrainerDA5 import nnUNetTrainerDA5
from nnunetv2.training.nnUNetTrainer.variants.mixins._base import ComposableTrainerMixin


class nnUNetTrainerDA5_100epochs(ComposableTrainerMixin, nnUNetTrainerDA5):
    """DA5 trainer with 100 epochs.

    Inherits ComposableTrainerMixin so that perform_actual_validation uses
    the module-level _blosc2_init_worker in the spawn pool (fixes blosc2
    RuntimeError in spawned worker processes when writing predicted_next_stage
    .b2nd files).  All mixin hooks are no-ops, so training behaviour is
    identical to the plain DA5 trainer.
    """
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 100
