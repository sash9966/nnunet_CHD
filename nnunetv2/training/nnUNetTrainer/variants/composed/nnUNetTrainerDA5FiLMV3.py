"""Composable DA5 + FiLM disease-vector conditioning trainer — V3.

V3 changes vs V2
----------------
- FiLM at bottleneck only (removed per-decoder-stage FiLM that compounded
  to (1+γ)^N ≈ 2× feature distortion with N=7 stages).
- Removed auxiliary disease classifier and its competing BCE loss.
- disease_lr_multiplier reduced from 10.0 → 2.0.
- disease_K set to 8 to match the actual disease_map.json flag count.
"""
import torch

from nnunetv2.architectures.film_conditioned_unet import FiLMConditionedResEncUNet
from nnunetv2.training.nnUNetTrainer.variants.data_augmentation.nnUNetTrainerDA5 import nnUNetTrainerDA5
from nnunetv2.training.nnUNetTrainer.variants.mixins._base import ComposableTrainerMixin
from nnunetv2.training.nnUNetTrainer.variants.mixins.disease_conditioning import (
    DiseaseConditioningMixin,
    build_disease_conditioned_network,
)


class nnUNetTrainerDA5FiLMV3(DiseaseConditioningMixin, ComposableTrainerMixin, nnUNetTrainerDA5):
    disease_wrapper_class = FiLMConditionedResEncUNet
    disease_param_prefixes = {'disease_mlp', 'bottleneck_film'}

    def build_network_architecture(self, *a, **kw):
        return build_disease_conditioned_network(self, FiLMConditionedResEncUNet, *a, **kw)


class nnUNetTrainerDA5FiLMV3_100epochs(nnUNetTrainerDA5FiLMV3):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 100


class nnUNetTrainerDA5FiLMV3_500epochs(nnUNetTrainerDA5FiLMV3):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 500
