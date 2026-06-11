"""DA5 + binary great-vessel (AO ∪ PA) soft-clDice topology loss.

TRAINER 2 of the "beat-the-baseline" experiment set.  Concentrates topology
pressure on great-vessel *continuity* by treating AO ∪ PA as a single connected
tubular structure (binary clDice), instead of the per-class topology loss used
by ``nnUNetTrainerDA5*Topo``.

The network is unchanged → standard ``nnUNetv2_predict`` works and outputs are
directly comparable to ``nnUNetTrainerDA5``.

MRO: VesselFocusedTopologyMixin → ComposableTrainerMixin → nnUNetTrainerDA5 → nnUNetTrainer
"""
import torch

from nnunetv2.training.nnUNetTrainer.variants.data_augmentation.nnUNetTrainerDA5 import nnUNetTrainerDA5
from nnunetv2.training.nnUNetTrainer.variants.mixins._base import ComposableTrainerMixin
from nnunetv2.training.nnUNetTrainer.variants.mixins.vessel_topo import VesselFocusedTopologyMixin


class nnUNetTrainerDA5VesselFocusedTopo(VesselFocusedTopologyMixin, ComposableTrainerMixin, nnUNetTrainerDA5):
    """DA5 augmentation + binary great-vessel soft-clDice (continuity-focused)."""
    pass


class nnUNetTrainerDA5VesselFocusedTopo_100epochs(nnUNetTrainerDA5VesselFocusedTopo):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 100


class nnUNetTrainerDA5VesselFocusedTopo_200epochs(nnUNetTrainerDA5VesselFocusedTopo):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 200


class nnUNetTrainerDA5VesselFocusedTopo_500epochs(nnUNetTrainerDA5VesselFocusedTopo):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 500
