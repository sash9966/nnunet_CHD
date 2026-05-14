"""Composable topology-loss-only trainer (no disease conditioning)."""
import torch

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.nnUNetTrainer.variants.mixins._base import ComposableTrainerMixin
from nnunetv2.training.nnUNetTrainer.variants.mixins.topology_loss import TopologyLossMixin


class nnUNetTrainerTopoLossV2(TopologyLossMixin, ComposableTrainerMixin, nnUNetTrainer):
    pass


class nnUNetTrainerTopoLossV2_100epochs(nnUNetTrainerTopoLossV2):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 100


class nnUNetTrainerTopoLossV2_200epochs(nnUNetTrainerTopoLossV2):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 200


class nnUNetTrainerTopoLossV2_1000epochs(nnUNetTrainerTopoLossV2):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 1000
