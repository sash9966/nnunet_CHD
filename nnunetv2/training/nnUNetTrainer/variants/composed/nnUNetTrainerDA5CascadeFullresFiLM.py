"""Cascade high-res trainer: DA5 + FiLM disease conditioning.

Used as the ``3d_cascade_fullres`` stage.  At this stage nnU-Net's dataloader
concatenates the low-res predictions (one-hot, after random binary morphological
perturbation during training) as additional input channels.  The network receives
both the full-resolution image patch and the spatial prior from the low-res stage.

FiLM conditioning is retained at the high-res stage so the boundary refinement
is also disease-aware:
- VSD cases:  the low-res prior marks the general LV/RV region; the high-res
  stage sharpens the inter-ventricular boundary.  FiLM biases the decoder
  toward a VSD-compatible boundary (VSD flag active → LV–RV adjacency allowed).
- TGA cases:  the low-res prior already has the correct global vessel assignment
  (AO→RV, PA→LV); FiLM at the high-res stage preserves this during boundary
  refinement so the high-res decoder does not revert to normal-anatomy priors.

Architecture note
-----------------
``build_disease_conditioned_network`` rebuilds the network from the cascade
fullres configuration, which has a larger number of input channels than the
lowres configuration (num_classes extra channels for the one-hot prior).
``FiLMConditionedResEncUNet`` wraps the resulting ResidualEncoderUNet identically
regardless of input channel count.

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


class nnUNetTrainerDA5CascadeFullresFiLM(
    DiseaseConditioningMixin,
    ComposableTrainerMixin,
    nnUNetTrainerDA5,
):
    """Cascade-fullres trainer: DA5 + FiLM bottleneck conditioning.

    Intended for use with the 3d_cascade_fullres configuration after one of the
    DA5Cascade lowres trainers has produced predictions on the training cases.
    """

    disease_wrapper_class  = FiLMConditionedResEncUNet
    disease_param_prefixes = {'disease_mlp', 'bottleneck_film'}

    def build_network_architecture(self, *a, **kw):
        return build_disease_conditioned_network(self, FiLMConditionedResEncUNet, *a, **kw)


class nnUNetTrainerDA5CascadeFullresFiLM_100epochs(nnUNetTrainerDA5CascadeFullresFiLM):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 100


class nnUNetTrainerDA5CascadeFullresFiLM_200epochs(nnUNetTrainerDA5CascadeFullresFiLM):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 200
