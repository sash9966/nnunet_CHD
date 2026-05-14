"""Composable DA5 + FiLM conditioning + disease-adaptive targets."""
import torch

from nnunetv2.architectures.film_conditioned_unet import FiLMConditionedResEncUNet
from nnunetv2.training.nnUNetTrainer.variants.data_augmentation.nnUNetTrainerDA5 import nnUNetTrainerDA5
from nnunetv2.training.nnUNetTrainer.variants.mixins._base import ComposableTrainerMixin
from nnunetv2.training.nnUNetTrainer.variants.mixins.adaptive_targets import DiseaseAdaptiveTargetMixin
from nnunetv2.training.nnUNetTrainer.variants.mixins.disease_conditioning import (
    DiseaseConditioningMixin,
    build_disease_conditioned_network,
)


class nnUNetTrainerDA5FiLMAdaptive(
    DiseaseConditioningMixin,
    DiseaseAdaptiveTargetMixin,
    ComposableTrainerMixin,
    nnUNetTrainerDA5,
):
    """FiLM disease conditioning + disease-adaptive training targets.

    Combines two complementary ideas:
    - FiLM modulation gives the network disease-specific feature transforms
      (bottleneck only, to avoid compound scaling instability)
    - Adaptive targets relax the loss at boundaries that are genuinely
      ambiguous for each disease (VSD → LV/RV, PuA → AO/PA)

    The model still predicts all classes; it's just not penalised for
    getting the ambiguous boundary wrong in affected cases.
    """
    disease_wrapper_class = FiLMConditionedResEncUNet
    disease_param_prefixes = {'disease_mlp', 'bottleneck_film'}

    def build_network_architecture(self, *a, **kw):
        return build_disease_conditioned_network(self, FiLMConditionedResEncUNet, *a, **kw)


class nnUNetTrainerDA5FiLMAdaptive_100epochs(nnUNetTrainerDA5FiLMAdaptive):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 100


class nnUNetTrainerDA5FiLMAdaptive_200epochs(nnUNetTrainerDA5FiLMAdaptive):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 200


class nnUNetTrainerDA5FiLMAdaptive_500epochs(nnUNetTrainerDA5FiLMAdaptive):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 500
