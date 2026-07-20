"""DA5 septal-defect ABLATION arms (Dataset051, septal label id 8).

Isolates where a septal-detection gain comes from — sampling vs loss vs both:
  * nnUNetTrainerDA5SeptalOversample      — oversampling lever only
  * nnUNetTrainerDA5SeptalTversky         — FN-weighted Tversky loss lever only
  * nnUNetTrainerDA5SeptalOversampleTversky — both combined

Reference arms (already exist): nnUNetTrainerDA5 (anatomy+label8, no focus),
nnUNetTrainerDA5DiseaseLandmark (soft-Dice + clDice). All same fold-0 split.

FIXED Tversky arms (V2), after the Dataset070 ablation showed the original
weight=1.0 Tversky COLLAPSED class 8 to 0 predicted voxels (0/13 test cases vs
oversample-only 10/13). V2 = low weight (0.1) + warmup (term off until ep50) +
linear ramp (30 ep) + softened bias (alpha/beta 0.4/0.6):
  * nnUNetTrainerDA5SeptalTverskyV2          — fixed Tversky only
  * nnUNetTrainerDA5SeptalOversampleTverskyV2 — oversample + fixed Tversky
Original (collapsing) arms are kept unchanged for reproducibility of the finding.

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


# ---- Fixed Tversky (V2): low weight + warmup ramp + softened bias -------------
# Ablation on Dataset070 showed the weight=1.0 / 0.3-0.7 Tversky suppressed the
# septal class to zero. These arms apply the fix. Tune here in one place.
_TVERSKY_V2 = dict(
    septal_tversky_weight=0.1,
    septal_tversky_warmup_epochs=50,
    septal_tversky_ramp_epochs=30,
    septal_tversky_alpha=0.4,
    septal_tversky_beta=0.6,
)


class nnUNetTrainerDA5SeptalTverskyV2(
        SeptalTverskyMixin, ComposableTrainerMixin, nnUNetTrainerDA5):
    """FIXED Tversky only: low weight + warmup + linear ramp + softer FN bias."""
    septal_tversky_weight = _TVERSKY_V2["septal_tversky_weight"]
    septal_tversky_warmup_epochs = _TVERSKY_V2["septal_tversky_warmup_epochs"]
    septal_tversky_ramp_epochs = _TVERSKY_V2["septal_tversky_ramp_epochs"]
    septal_tversky_alpha = _TVERSKY_V2["septal_tversky_alpha"]
    septal_tversky_beta = _TVERSKY_V2["septal_tversky_beta"]


class nnUNetTrainerDA5SeptalOversampleTverskyV2(
        SeptalOversampleMixin, SeptalTverskyMixin, ComposableTrainerMixin, nnUNetTrainerDA5):
    """Oversample + FIXED Tversky (low weight + warmup + ramp + softer FN bias)."""
    septal_tversky_weight = _TVERSKY_V2["septal_tversky_weight"]
    septal_tversky_warmup_epochs = _TVERSKY_V2["septal_tversky_warmup_epochs"]
    septal_tversky_ramp_epochs = _TVERSKY_V2["septal_tversky_ramp_epochs"]
    septal_tversky_alpha = _TVERSKY_V2["septal_tversky_alpha"]
    septal_tversky_beta = _TVERSKY_V2["septal_tversky_beta"]


nnUNetTrainerDA5SeptalOversample_200epochs = _mk_epochs(nnUNetTrainerDA5SeptalOversample, 200)
nnUNetTrainerDA5SeptalTversky_200epochs = _mk_epochs(nnUNetTrainerDA5SeptalTversky, 200)
nnUNetTrainerDA5SeptalOversampleTversky_200epochs = _mk_epochs(nnUNetTrainerDA5SeptalOversampleTversky, 200)
nnUNetTrainerDA5SeptalTverskyV2_200epochs = _mk_epochs(nnUNetTrainerDA5SeptalTverskyV2, 200)
nnUNetTrainerDA5SeptalOversampleTverskyV2_200epochs = _mk_epochs(nnUNetTrainerDA5SeptalOversampleTverskyV2, 200)
