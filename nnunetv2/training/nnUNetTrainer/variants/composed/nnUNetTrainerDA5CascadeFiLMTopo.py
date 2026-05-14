"""Cascade low-res trainer: DA5 + FiLM disease conditioning + topology loss.

Experiment B in the cascade ablation.  Adds soft-clDice topology loss on AO/PA
classes on top of Experiment A.  The topology term penalises topological
disconnections in the AO and PA predictions, which is the primary failure mode
in PuA and TGA cases where the network partially breaks vessel connectivity.

At the low-res stage the topology loss is especially effective because each patch
contains the full vessel tree, so the soft-skeleton supervision has global reach.

Loss schedule (from TopologyLossMixin defaults):
    epochs 0–10   : topo_weight = 0.0  (warm-up, classification loss only)
    epochs 10–30  : topo_weight ramps from 0.0 → topo_w_high (1.0)
    epochs 30+    : topo_weight decays toward topo_w_low (0.1)

MRO: DiseaseConditioningMixin → TopologyLossMixin → ComposableTrainerMixin
       → nnUNetTrainerDA5 → nnUNetTrainer
"""
import torch

from nnunetv2.architectures.film_conditioned_unet import FiLMConditionedResEncUNet
from nnunetv2.training.nnUNetTrainer.variants.data_augmentation.nnUNetTrainerDA5 import nnUNetTrainerDA5
from nnunetv2.training.nnUNetTrainer.variants.mixins._base import ComposableTrainerMixin
from nnunetv2.training.nnUNetTrainer.variants.mixins.disease_conditioning import (
    DiseaseConditioningMixin,
    build_disease_conditioned_network,
)
from nnunetv2.training.nnUNetTrainer.variants.mixins.topology_loss import TopologyLossMixin


class nnUNetTrainerDA5CascadeFiLMTopo(
    DiseaseConditioningMixin,
    TopologyLossMixin,
    ComposableTrainerMixin,
    nnUNetTrainerDA5,
):
    """Cascade-lowres trainer: DA5 + FiLM conditioning + soft-clDice topology loss."""

    disease_wrapper_class  = FiLMConditionedResEncUNet
    disease_param_prefixes = {'disease_mlp', 'bottleneck_film'}

    def build_network_architecture(self, *a, **kw):
        return build_disease_conditioned_network(self, FiLMConditionedResEncUNet, *a, **kw)


class nnUNetTrainerDA5CascadeFiLMTopo_100epochs(nnUNetTrainerDA5CascadeFiLMTopo):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 100


class nnUNetTrainerDA5CascadeFiLMTopo_200epochs(nnUNetTrainerDA5CascadeFiLMTopo):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 200
