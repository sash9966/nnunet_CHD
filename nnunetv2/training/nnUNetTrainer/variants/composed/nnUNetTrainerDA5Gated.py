"""Composable DA5 + spatially-gated disease conditioning trainer."""
import torch

from nnunetv2.architectures.gated_conditioned_unet import GatedConditionedResEncUNet
from nnunetv2.training.nnUNetTrainer.variants.data_augmentation.nnUNetTrainerDA5 import nnUNetTrainerDA5
from nnunetv2.training.nnUNetTrainer.variants.mixins._base import ComposableTrainerMixin
from nnunetv2.training.nnUNetTrainer.variants.mixins.disease_conditioning import (
    DiseaseConditioningMixin,
    build_disease_conditioned_network,
)


class nnUNetTrainerDA5Gated(DiseaseConditioningMixin, ComposableTrainerMixin, nnUNetTrainerDA5):
    """Spatially-gated disease conditioning.

    Unlike FiLM (same gamma/beta everywhere), this wrapper predicts
    spatially-varying modulation conditioned on both the disease vector
    AND local image features.  The model can learn location-specific
    modulation, e.g. "at AO-PA junction + pulmonary_atresia, modulate
    differently."
    """
    disease_wrapper_class = GatedConditionedResEncUNet
    disease_param_prefixes = {'disease_mlp', 'bottleneck_gate', 'decoder_gates', 'disease_classifier'}

    def build_network_architecture(self, *a, **kw):
        return build_disease_conditioned_network(self, GatedConditionedResEncUNet, *a, **kw)


class nnUNetTrainerDA5Gated_100epochs(nnUNetTrainerDA5Gated):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 100
