"""Cascade-fullres trainer variants — one per lowres ablation.

Each class is the high-res counterpart for a specific low-res ablation stage.
All use DA5 augmentation; Topo variants apply soft-clDice on AO/PA at the
high-res stage in addition to whatever was applied at the low-res stage.

Ablation mapping:

  Low-res trainer                         | High-res trainer
  ----------------------------------------|-----------------------------------------
  nnUNetTrainerDA5 (baseline)             | nnUNetTrainerDA5CascadeFullresBaseline
  nnUNetTrainerDA5CascadeFiLM             | nnUNetTrainerDA5CascadeFullresFiLM (separate file)
  nnUNetTrainerDA5CascadeTopo             | nnUNetTrainerDA5CascadeFullresTopo
  nnUNetTrainerDA5CascadeFiLMTopo         | nnUNetTrainerDA5CascadeFullresFiLMTopo

Note: the fullres FiLM-only trainer lives in nnUNetTrainerDA5CascadeFullresFiLM.py.
"""
import torch

from nnunetv2.architectures.film_conditioned_unet import FiLMConditionedResEncUNet
from nnunetv2.training.nnUNetTrainer.variants.data_augmentation.nnUNetTrainerDA5 import nnUNetTrainerDA5
from nnunetv2.training.nnUNetTrainer.variants.data_augmentation.nnUNetTrainerDA5_epochs import (
    nnUNetTrainerDA5_100epochs,
    nnUNetTrainerDA5_200epochs,
)
from nnunetv2.training.nnUNetTrainer.variants.composed.nnUNetTrainerDA5CascadeFullresFiLM import (
    nnUNetTrainerDA5CascadeFullresFiLM_100epochs,
    nnUNetTrainerDA5CascadeFullresFiLM_200epochs,
)
from nnunetv2.training.nnUNetTrainer.variants.mixins._base import ComposableTrainerMixin
from nnunetv2.training.nnUNetTrainer.variants.mixins.disease_conditioning import (
    DiseaseConditioningMixin,
    build_disease_conditioned_network,
)
from nnunetv2.training.nnUNetTrainer.variants.mixins.topology_loss import TopologyLossMixin


# ---------------------------------------------------------------------------
# Baseline (no topology, no FiLM) — plain DA5, unique name for result folder
# ---------------------------------------------------------------------------

class nnUNetTrainerDA5CascadeFullresBaseline_100epochs(nnUNetTrainerDA5_100epochs):
    """Cascade-fullres counterpart for the plain DA5 baseline lowres stage."""
    pass


class nnUNetTrainerDA5CascadeFullresBaseline_200epochs(nnUNetTrainerDA5_200epochs):
    """Cascade-fullres counterpart for the plain DA5 baseline lowres stage."""
    pass


# ---------------------------------------------------------------------------
# Topology-only fullres variants
# ---------------------------------------------------------------------------

class nnUNetTrainerDA5CascadeFullresTopo(TopologyLossMixin, ComposableTrainerMixin, nnUNetTrainerDA5):
    """High-res cascade trainer: DA5 + soft-clDice topology loss on AO/PA.

    Intended as the fullres stage after nnUNetTrainerDA5CascadeTopo (lowres).
    Applies topology loss at both cascade stages end-to-end.
    """
    pass


class nnUNetTrainerDA5CascadeFullresTopo_100epochs(nnUNetTrainerDA5CascadeFullresTopo):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 100


class nnUNetTrainerDA5CascadeFullresTopo_200epochs(nnUNetTrainerDA5CascadeFullresTopo):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 200


# ---------------------------------------------------------------------------
# FiLM + Topology fullres variant
# ---------------------------------------------------------------------------

class nnUNetTrainerDA5CascadeFullresFiLMTopo(
    DiseaseConditioningMixin,
    TopologyLossMixin,
    ComposableTrainerMixin,
    nnUNetTrainerDA5,
):
    """High-res cascade trainer: DA5 + FiLM bottleneck conditioning + topology loss.

    Intended as the fullres stage after nnUNetTrainerDA5CascadeFiLMTopo (lowres).
    Applies both disease conditioning and AO/PA topology loss at the fullres stage.

    MRO: DiseaseConditioningMixin → TopologyLossMixin → ComposableTrainerMixin → DA5
    """

    disease_wrapper_class  = FiLMConditionedResEncUNet
    disease_param_prefixes = {'disease_mlp', 'bottleneck_film'}

    def build_network_architecture(self, *a, **kw):
        return build_disease_conditioned_network(self, FiLMConditionedResEncUNet, *a, **kw)


class nnUNetTrainerDA5CascadeFullresFiLMTopo_100epochs(nnUNetTrainerDA5CascadeFullresFiLMTopo):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 100


class nnUNetTrainerDA5CascadeFullresFiLMTopo_200epochs(nnUNetTrainerDA5CascadeFullresFiLMTopo):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 200
