"""DA5 septal-defect ABLATION arms (Dataset051, septal label id 8).

Isolates where a septal-detection gain comes from — sampling vs loss vs both:
  * nnUNetTrainerDA5SeptalOversample      — oversampling lever only
  * nnUNetTrainerDA5SeptalTversky         — FN-weighted Tversky loss lever only
  * nnUNetTrainerDA5SeptalOversampleTversky — both combined

Reference arms (already exist): nnUNetTrainerDA5 (anatomy+label8, no focus),
nnUNetTrainerDA5DiseaseLandmark (soft-Dice + clDice). All same fold-0 split.

MRO: <feature mixins> → ComposableTrainerMixin → nnUNetTrainerDA5 → nnUNetTrainer
"""
import torch

from nnunetv2.training.nnUNetTrainer.variants.data_augmentation.nnUNetTrainerDA5 import nnUNetTrainerDA5
from nnunetv2.training.nnUNetTrainer.variants.mixins._base import ComposableTrainerMixin
from nnunetv2.training.nnUNetTrainer.variants.mixins.septal_focus import (
    SeptalOversampleMixin, SeptalTverskyMixin)


def _mk_epochs(base_cls, n):
    class _E(base_cls):
        def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                     device: torch.device = torch.device('cuda')):
            super().__init__(plans, configuration, fold, dataset_json, device)
            self.num_epochs = n
    _E.__name__ = f"{base_cls.__name__}_{n}epochs"
    _E.__qualname__ = _E.__name__
    return _E


# ---- Arm 1: oversampling only -------------------------------------------------
class nnUNetTrainerDA5SeptalOversample(SeptalOversampleMixin, ComposableTrainerMixin, nnUNetTrainerDA5):
    """Septal-class foreground oversampling only."""
    pass


# ---- Arm 2: Tversky loss only -------------------------------------------------
class nnUNetTrainerDA5SeptalTversky(SeptalTverskyMixin, ComposableTrainerMixin, nnUNetTrainerDA5):
    """FN-weighted Tversky on the septal class only."""
    pass


# ---- Arm 3: combined ----------------------------------------------------------
class nnUNetTrainerDA5SeptalOversampleTversky(
        SeptalOversampleMixin, SeptalTverskyMixin, ComposableTrainerMixin, nnUNetTrainerDA5):
    """Septal oversampling + FN-weighted Tversky (combined)."""
    pass


nnUNetTrainerDA5SeptalOversample_200epochs = _mk_epochs(nnUNetTrainerDA5SeptalOversample, 200)
nnUNetTrainerDA5SeptalTversky_200epochs = _mk_epochs(nnUNetTrainerDA5SeptalTversky, 200)
nnUNetTrainerDA5SeptalOversampleTversky_200epochs = _mk_epochs(nnUNetTrainerDA5SeptalOversampleTversky, 200)
