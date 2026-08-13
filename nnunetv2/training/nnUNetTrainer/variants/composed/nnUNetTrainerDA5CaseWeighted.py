"""
DA5 + per-case sampling weights (for the all-data clinic model, Dataset100).

Draws trustworthy cases more often and noisy pseudo-labels less often, from a
``case_weights.json`` written by tools/build_dataset100_finalclinic.py. Default D100 map:
ImageCHD GT 1x, Dataset080 expert 3x, QC'd promoted 1x, usable clinical pseudo 0.5x,
Fanwei pseudo 0.5x (kept for FOV/volume diversity, not label trust).

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
