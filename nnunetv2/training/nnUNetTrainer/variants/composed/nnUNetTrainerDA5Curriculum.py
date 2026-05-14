"""Composable DA5 + curriculum CE class weighting trainer."""
import torch

from nnunetv2.training.nnUNetTrainer.variants.data_augmentation.nnUNetTrainerDA5 import nnUNetTrainerDA5
from nnunetv2.training.nnUNetTrainer.variants.mixins._base import ComposableTrainerMixin
from nnunetv2.training.nnUNetTrainer.variants.mixins.curriculum_weights import CurriculumWeightsMixin


class nnUNetTrainerDA5Curriculum(CurriculumWeightsMixin, ComposableTrainerMixin, nnUNetTrainerDA5):
    """DA5 augmentation + curriculum CE weighting for AO/PA.

    Upweights AO/PA classes in CE loss during the first 20% of epochs
    (3x multiplier, step schedule). Dice loss unaffected.
    """
    pass


class nnUNetTrainerDA5Curriculum_100epochs(nnUNetTrainerDA5Curriculum):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 100


class nnUNetTrainerDA5Curriculum_200epochs(nnUNetTrainerDA5Curriculum):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 200


class nnUNetTrainerDA5Curriculum_1000epochs(nnUNetTrainerDA5Curriculum):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 1000
