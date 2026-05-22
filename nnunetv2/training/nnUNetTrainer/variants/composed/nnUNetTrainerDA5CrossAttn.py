"""Composed trainers using cross-attention disease conditioning.

CrossAttentionConditioningMixin attaches a single-head cross-attention block
at each decoder stage, conditioning spatial features on a sequence of
per-disease token embeddings rather than a single global FiLM modulation.

These trainers are mutually exclusive with all FiLM trainers.
They require disease_map.json in the preprocessed dataset folder.

Trainer inventory
-----------------
3d_fullres — standalone cross-attention:
    nnUNetTrainerDA5CrossAttn
    nnUNetTrainerDA5CrossAttn_100epochs
    nnUNetTrainerDA5CrossAttn_200epochs

3d_fullres — cross-attention + topology loss:
    nnUNetTrainerDA5CrossAttnTopo
    nnUNetTrainerDA5CrossAttnTopo_100epochs
    nnUNetTrainerDA5CrossAttnTopo_200epochs

3d_fullres — aux diagnosis head + cross-attention (embedding reuse):
    nnUNetTrainerDA5AuxDiagCrossAttn
    nnUNetTrainerDA5AuxDiagCrossAttn_100epochs
    nnUNetTrainerDA5AuxDiagCrossAttn_200epochs

MRO for AuxDiag + CrossAttn:
    AuxiliaryDiagnosisMixin first so mixin_extra_loss computes the
    classification loss and sets diagnosis_embedding BEFORE the
    cross-attention mixin reads it via forward_kwargs.
"""
import torch

from nnunetv2.training.nnUNetTrainer.variants.data_augmentation.nnUNetTrainerDA5 import nnUNetTrainerDA5
from nnunetv2.training.nnUNetTrainer.variants.mixins._base import ComposableTrainerMixin
from nnunetv2.training.nnUNetTrainer.variants.mixins.aux_diagnosis_mixin import AuxiliaryDiagnosisMixin
from nnunetv2.training.nnUNetTrainer.variants.mixins.cross_attention_mixin import CrossAttentionConditioningMixin
from nnunetv2.training.nnUNetTrainer.variants.mixins.topology_loss import TopologyLossMixin


# ---------------------------------------------------------------------------
# Standalone cross-attention (no topology, no AuxDiag)
# ---------------------------------------------------------------------------

class nnUNetTrainerDA5CrossAttn(
    CrossAttentionConditioningMixin,
    ComposableTrainerMixin,
    nnUNetTrainerDA5,
):
    """DA5 augmentation + per-stage cross-attention disease conditioning."""
    pass


class nnUNetTrainerDA5CrossAttn_100epochs(nnUNetTrainerDA5CrossAttn):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 100


class nnUNetTrainerDA5CrossAttn_200epochs(nnUNetTrainerDA5CrossAttn):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 200


# ---------------------------------------------------------------------------
# Cross-attention + topology loss
# ---------------------------------------------------------------------------

class nnUNetTrainerDA5CrossAttnTopo(
    CrossAttentionConditioningMixin,
    TopologyLossMixin,
    ComposableTrainerMixin,
    nnUNetTrainerDA5,
):
    """DA5 + per-stage cross-attention + fixed-weight soft-clDice on AO/PA."""
    pass


class nnUNetTrainerDA5CrossAttnTopo_100epochs(nnUNetTrainerDA5CrossAttnTopo):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 100


class nnUNetTrainerDA5CrossAttnTopo_200epochs(nnUNetTrainerDA5CrossAttnTopo):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 200


# ---------------------------------------------------------------------------
# Auxiliary diagnosis head + cross-attention (embedding reuse)
# ---------------------------------------------------------------------------

class nnUNetTrainerDA5AuxDiagCrossAttn(
    AuxiliaryDiagnosisMixin,
    CrossAttentionConditioningMixin,
    ComposableTrainerMixin,
    nnUNetTrainerDA5,
):
    """DA5 + bottleneck aux classification head + per-stage cross-attention.

    MRO: AuxiliaryDiagnosisMixin before CrossAttentionConditioningMixin.
    - AuxDiag registers bottleneck hook and exposes diagnosis_embedding.
    - CrossAttn auto-detects AuxDiag presence and builds CrossAttentionConditionedResEncUNet
      with use_aux_embedding=True, so per-disease tokens come from K projected
      views of the 256-d diagnosis_embedding rather than the static table.
    - Both disease_vec (for PAD masking) and diagnosis_embedding (for aux_proj)
      are passed through forward_kwargs to the network.
    """
    use_aux_embedding = False  # AuxDiag flag: no FiLM to pass embedding to


class nnUNetTrainerDA5AuxDiagCrossAttn_100epochs(nnUNetTrainerDA5AuxDiagCrossAttn):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 100


class nnUNetTrainerDA5AuxDiagCrossAttn_200epochs(nnUNetTrainerDA5AuxDiagCrossAttn):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 200
