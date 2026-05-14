"""Cascade low-res trainer: DA5 + topology loss (no FiLM conditioning).

Experiment D in the cascade ablation.  Ablates the FiLM conditioning to isolate
the contribution of the topology loss alone.  Useful for comparing:

    D  (topo only)   vs  A  (FiLM only)   →  which matters more for vessel topology?
    D  (topo only)   vs  B  (FiLM+topo)   →  does disease conditioning add on top of topo?

The soft-clDice topology loss targets AO (label 6) and PA (label 7) by default
(see TopologyLossMixin.topo_class_ids).  At the low-res stage each patch covers
the full vessel tree, so the skeleton-based supervision is globally effective.

MRO: TopologyLossMixin → ComposableTrainerMixin → nnUNetTrainerDA5 → nnUNetTrainer
"""
import torch

from nnunetv2.training.nnUNetTrainer.variants.data_augmentation.nnUNetTrainerDA5 import nnUNetTrainerDA5
from nnunetv2.training.nnUNetTrainer.variants.mixins._base import ComposableTrainerMixin
from nnunetv2.training.nnUNetTrainer.variants.mixins.topology_loss import TopologyLossMixin


class nnUNetTrainerDA5CascadeTopo(TopologyLossMixin, ComposableTrainerMixin, nnUNetTrainerDA5):
    """Cascade-lowres trainer: DA5 augmentation + soft-clDice topology loss on AO/PA."""
    pass


class nnUNetTrainerDA5CascadeTopo_100epochs(nnUNetTrainerDA5CascadeTopo):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 100


class nnUNetTrainerDA5CascadeTopo_200epochs(nnUNetTrainerDA5CascadeTopo):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 200
