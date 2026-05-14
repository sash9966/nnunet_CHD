"""Cascade low-res trainer: DA5 augmentation + FiLM disease conditioning.

Used as the ``3d_lowres`` stage of the cascade pipeline.  The low-res stage sees
the full cardiac anatomy in a single patch, providing the global context needed
for disease-informed vessel label assignment.  FiLM conditioning injects the
8-element disease vector at the bottleneck so the network learns topology
priors per disease (e.g., AO→RV / PA→LV for TGA; both vessels from RV for DORV).

Experiment A in the cascade ablation.

Cascade workflow
----------------
1. Plan with ExperimentPlannerForceLowRes  →  generates 3d_lowres + 3d_cascade_fullres
2. Train this trainer on 3d_lowres
3. Run nnUNetv2_predict (lowres) on training cases  →  produces prior segmentations
4. Set up symlinks with scripts/setup_cascade_predictions.py
5. Train nnUNetTrainerDA5CascadeFullresFiLM on 3d_cascade_fullres

MRO: DiseaseConditioningMixin → ComposableTrainerMixin → nnUNetTrainerDA5 → nnUNetTrainer
"""
import torch

from nnunetv2.architectures.film_conditioned_unet import FiLMConditionedResEncUNet
from nnunetv2.training.nnUNetTrainer.variants.data_augmentation.nnUNetTrainerDA5 import nnUNetTrainerDA5
from nnunetv2.training.nnUNetTrainer.variants.mixins._base import ComposableTrainerMixin
from nnunetv2.training.nnUNetTrainer.variants.mixins.disease_conditioning import (
    DiseaseConditioningMixin,
    build_disease_conditioned_network,
)


class nnUNetTrainerDA5CascadeFiLM(DiseaseConditioningMixin, ComposableTrainerMixin, nnUNetTrainerDA5):
    """Cascade-lowres trainer: DA5 + FiLM bottleneck disease conditioning."""

    disease_wrapper_class  = FiLMConditionedResEncUNet
    disease_param_prefixes = {'disease_mlp', 'bottleneck_film'}

    def build_network_architecture(self, *a, **kw):
        return build_disease_conditioned_network(self, FiLMConditionedResEncUNet, *a, **kw)


class nnUNetTrainerDA5CascadeFiLM_100epochs(nnUNetTrainerDA5CascadeFiLM):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 100


class nnUNetTrainerDA5CascadeFiLM_200epochs(nnUNetTrainerDA5CascadeFiLM):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 200
