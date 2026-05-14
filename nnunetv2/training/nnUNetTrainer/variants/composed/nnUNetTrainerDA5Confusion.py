"""Composable DA5 + confusion-pair penalty trainer."""
import torch

from nnunetv2.training.nnUNetTrainer.variants.data_augmentation.nnUNetTrainerDA5 import nnUNetTrainerDA5
from nnunetv2.training.nnUNetTrainer.variants.mixins._base import ComposableTrainerMixin
from nnunetv2.training.nnUNetTrainer.variants.mixins.confusion_penalty import ConfusionPenaltyMixin


class nnUNetTrainerDA5Confusion(ConfusionPenaltyMixin, ComposableTrainerMixin, nnUNetTrainerDA5):
    """DA5 augmentation + targeted AO-PA confusion penalty.

    Auto-detects AO and PA class indices from dataset.json labels and
    adds a loss penalty when the model confuses them.  No disease
    conditioning — this is a standalone structural loss improvement.
    """
    pass


class nnUNetTrainerDA5Confusion_100epochs(nnUNetTrainerDA5Confusion):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 100


class nnUNetTrainerDA5Confusion_200epochs(nnUNetTrainerDA5Confusion):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 200
