"""DA5 + hierarchical region-scaffold supervision.

TRAINER 1 of the "beat-the-baseline" experiment set.  Adds auxiliary region-level
losses (whole-heart, blood-pool, chambers, ventricles, atria, great-vessels,
myocardium) on soft probabilities to stabilise coarse anatomy and reduce the
class-competition flip-flopping that Dice barely penalises.

The network is unchanged → standard ``nnUNetv2_predict`` works and outputs are
directly comparable to ``nnUNetTrainerDA5``.

MRO: RegionScaffoldMixin → ComposableTrainerMixin → nnUNetTrainerDA5 → nnUNetTrainer
"""
import torch

from nnunetv2.training.nnUNetTrainer.variants.data_augmentation.nnUNetTrainerDA5 import nnUNetTrainerDA5
from nnunetv2.training.nnUNetTrainer.variants.mixins._base import ComposableTrainerMixin
from nnunetv2.training.nnUNetTrainer.variants.mixins.region_scaffold import RegionScaffoldMixin


class nnUNetTrainerDA5RegionScaffold(RegionScaffoldMixin, ComposableTrainerMixin, nnUNetTrainerDA5):
    """DA5 augmentation + soft hierarchical region-scaffold loss."""
    pass


class nnUNetTrainerDA5RegionScaffold_100epochs(nnUNetTrainerDA5RegionScaffold):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 100


class nnUNetTrainerDA5RegionScaffold_200epochs(nnUNetTrainerDA5RegionScaffold):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 200


class nnUNetTrainerDA5RegionScaffold_500epochs(nnUNetTrainerDA5RegionScaffold):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 500
