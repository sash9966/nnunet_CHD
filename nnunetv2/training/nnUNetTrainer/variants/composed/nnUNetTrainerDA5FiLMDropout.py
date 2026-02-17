"""Composable DA5 + FiLM conditioning with disease-vector dropout."""
import torch

from nnunetv2.architectures.film_conditioned_unet import FiLMConditionedResEncUNet
from nnunetv2.training.nnUNetTrainer.variants.data_augmentation.nnUNetTrainerDA5 import nnUNetTrainerDA5
from nnunetv2.training.nnUNetTrainer.variants.mixins._base import ComposableTrainerMixin
from nnunetv2.training.nnUNetTrainer.variants.mixins.disease_conditioning import (
    DiseaseConditioningMixin,
    build_disease_conditioned_network,
)


class nnUNetTrainerDA5FiLMDropout(DiseaseConditioningMixin, ComposableTrainerMixin, nnUNetTrainerDA5):
    """FiLM disease conditioning with 20% dropout (classifier-free guidance).

    Randomly drops the disease vector during training so the model learns
    to *need* the conditioning signal when available.
    """
    disease_wrapper_class = FiLMConditionedResEncUNet
    disease_param_prefixes = {'disease_mlp', 'bottleneck_film'}

    def mixin_init(self):
        super().mixin_init()
        self.disease_dropout_prob = 0.2

    def build_network_architecture(self, *a, **kw):
        return build_disease_conditioned_network(self, FiLMConditionedResEncUNet, *a, **kw)


class nnUNetTrainerDA5FiLMDropout_100epochs(nnUNetTrainerDA5FiLMDropout):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 100


class nnUNetTrainerDA5FiLMDropout_500epochs(nnUNetTrainerDA5FiLMDropout):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 500
