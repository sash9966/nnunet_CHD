"""Composable DA5 + topology loss + curriculum CE weighting (no FiLM conditioning)."""
import torch

from nnunetv2.training.nnUNetTrainer.variants.data_augmentation.nnUNetTrainerDA5 import nnUNetTrainerDA5
from nnunetv2.training.nnUNetTrainer.variants.mixins._base import ComposableTrainerMixin
from nnunetv2.training.nnUNetTrainer.variants.mixins.curriculum_weights import CurriculumWeightsMixin
from nnunetv2.training.nnUNetTrainer.variants.mixins.topology_loss import TopologyLossMixin


class nnUNetTrainerDA5TopoCurriculum(
    TopologyLossMixin, CurriculumWeightsMixin, ComposableTrainerMixin, nnUNetTrainerDA5
):
    """DA5 + topology loss (soft-clDice on AO/PA) + curriculum CE weighting.

    No disease conditioning — pure structural loss combination.
    """
    pass


class nnUNetTrainerDA5TopoCurriculum_100epochs(nnUNetTrainerDA5TopoCurriculum):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 100


class nnUNetTrainerDA5TopoCurriculum_1000epochs(nnUNetTrainerDA5TopoCurriculum):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 1000
