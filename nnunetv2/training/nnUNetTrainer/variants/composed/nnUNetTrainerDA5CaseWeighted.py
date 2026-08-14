"""
DA5 + per-case sampling weights (for the all-data clinic model, Dataset100).

Draws trustworthy cases more often and noisy pseudo-labels less often, from a per-dataset
``case_weights.json`` = ``{case_id: weight}``. Example map: ground-truth 1x, expert-manual 3x,
QC'd pseudo 1x, auxiliary pseudo-labels 0.5x (the latter kept for FOV/volume diversity, not
label trust). If no case_weights.json is present the trainer samples uniformly (= plain DA5).

MRO: CaseSamplingWeightMixin -> ComposableTrainerMixin -> nnUNetTrainerDA5 -> nnUNetTrainer.
Train on fold 'all' for the deployment model; the weights only bias TRAIN sampling (val uniform).
"""
import torch

from nnunetv2.training.nnUNetTrainer.variants.data_augmentation.nnUNetTrainerDA5 import nnUNetTrainerDA5
from nnunetv2.training.nnUNetTrainer.variants.mixins._base import ComposableTrainerMixin
from nnunetv2.training.nnUNetTrainer.variants.mixins.case_weight import CaseSamplingWeightMixin


class nnUNetTrainerDA5CaseWeighted(CaseSamplingWeightMixin, ComposableTrainerMixin, nnUNetTrainerDA5):
    """DA5 with per-case sampling weights (reads case_weights.json; uniform if absent)."""
    pass


def _mk_epochs(base_cls, n):
    class _E(base_cls):
        def __init__(self, plans: dict, configuration: str, fold, dataset_json: dict,
                     device: torch.device = torch.device("cuda")):
            super().__init__(plans, configuration, fold, dataset_json, device)
            self.num_epochs = n
    _E.__name__ = f"{base_cls.__name__}_{n}epochs"
    _E.__qualname__ = _E.__name__
    return _E


nnUNetTrainerDA5CaseWeighted_200epochs = _mk_epochs(nnUNetTrainerDA5CaseWeighted, 200)
nnUNetTrainerDA5CaseWeighted_500epochs = _mk_epochs(nnUNetTrainerDA5CaseWeighted, 500)
