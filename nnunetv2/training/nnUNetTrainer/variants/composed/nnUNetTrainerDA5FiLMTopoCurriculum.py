"""Composable DA5 + FiLM + topology loss + curriculum CE weighting."""
import torch

from nnunetv2.architectures.film_conditioned_unet import FiLMConditionedResEncUNet
from nnunetv2.training.nnUNetTrainer.variants.data_augmentation.nnUNetTrainerDA5 import nnUNetTrainerDA5
from nnunetv2.training.nnUNetTrainer.variants.mixins._base import ComposableTrainerMixin
from nnunetv2.training.nnUNetTrainer.variants.mixins.curriculum_weights import CurriculumWeightsMixin
from nnunetv2.training.nnUNetTrainer.variants.mixins.disease_conditioning import (
    DiseaseConditioningMixin,
    build_disease_conditioned_network,
)
from nnunetv2.training.nnUNetTrainer.variants.mixins.topology_loss import TopologyLossMixin


class nnUNetTrainerDA5FiLMTopoCurriculum(
    DiseaseConditioningMixin, TopologyLossMixin, CurriculumWeightsMixin,
    ComposableTrainerMixin, nnUNetTrainerDA5
):
    """DA5 + FiLM conditioning (bottleneck only) + topology loss + curriculum CE weighting.

    All three features compose cleanly:
    - FiLM: bottleneck-only network conditioning via disease_vec
    - Topology: extra loss term (soft-clDice on AO/PA)
    - Curriculum: CE class weighting schedule (AO/PA upweighted early)
    """
    disease_wrapper_class = FiLMConditionedResEncUNet
    disease_param_prefixes = {'disease_mlp', 'bottleneck_film'}

    def build_network_architecture(self, *a, **kw):
        return build_disease_conditioned_network(self, FiLMConditionedResEncUNet, *a, **kw)


class nnUNetTrainerDA5FiLMTopoCurriculum_100epochs(nnUNetTrainerDA5FiLMTopoCurriculum):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 100


class nnUNetTrainerDA5FiLMTopoCurriculum_500epochs(nnUNetTrainerDA5FiLMTopoCurriculum):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 500


class nnUNetTrainerDA5FiLMTopoCurriculum_1000epochs(nnUNetTrainerDA5FiLMTopoCurriculum):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 1000
