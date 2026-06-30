"""DA5 + disease-landmark auxiliary supervision.

For training on ``Dataset031_imageCHD_DiseaseLandmarks`` (anatomy labels +
conservative derived disease-proxy integer labels produced by the
``chd_landmarks`` package). Adds:
  * a soft-Dice term for the rare derived labels (vsd_orifice_proxy, ...),
    applied with positive supervision only (unverified absence is not treated
    as background), and
  * optional great-vessel clDice (aorta ∪ PA).

The network is unchanged → standard ``nnUNetv2_predict`` works and outputs are
directly comparable to ``nnUNetTrainerDA5`` trained on the same dataset.

MRO: DiseaseLandmarkMixin → ComposableTrainerMixin → nnUNetTrainerDA5 → nnUNetTrainer
"""
import torch

from nnunetv2.training.nnUNetTrainer.variants.data_augmentation.nnUNetTrainerDA5 import nnUNetTrainerDA5
from nnunetv2.training.nnUNetTrainer.variants.mixins._base import ComposableTrainerMixin
from nnunetv2.training.nnUNetTrainer.variants.mixins.disease_landmark import DiseaseLandmarkMixin


class nnUNetTrainerDA5DiseaseLandmark(DiseaseLandmarkMixin, ComposableTrainerMixin, nnUNetTrainerDA5):
    """DA5 augmentation + disease-landmark auxiliary loss."""
    pass


class nnUNetTrainerDA5DiseaseLandmark_100epochs(nnUNetTrainerDA5DiseaseLandmark):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 100


class nnUNetTrainerDA5DiseaseLandmark_200epochs(nnUNetTrainerDA5DiseaseLandmark):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 200


class nnUNetTrainerDA5DiseaseLandmark_500epochs(nnUNetTrainerDA5DiseaseLandmark):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 500
