"""Composed trainers using the auxiliary diagnosis head mixin.

AuxiliaryDiagnosisMixin attaches a small MLP classification head to the
encoder bottleneck and trains it with BCEWithLogitsLoss against the 8-class
CHD diagnosis labels.  This regularises the encoder and exposes a 256-d
diagnosis_embedding that downstream conditioning mixins can reuse.

Trainer inventory
-----------------
3d_fullres (standalone, no FiLM):
    nnUNetTrainerDA5AuxDiag
    nnUNetTrainerDA5AuxDiag_100epochs
    nnUNetTrainerDA5AuxDiag_200epochs

3d_fullres + topology loss:
    nnUNetTrainerDA5AuxDiagTopo
    nnUNetTrainerDA5AuxDiagTopo_100epochs
    nnUNetTrainerDA5AuxDiagTopo_200epochs

3d_fullres + FiLM conditioning (embedding reuse):
    nnUNetTrainerDA5FiLMAuxDiag
    nnUNetTrainerDA5FiLMAuxDiag_100epochs
    nnUNetTrainerDA5FiLMAuxDiag_200epochs
"""
import torch

from nnunetv2.training.nnUNetTrainer.variants.data_augmentation.nnUNetTrainerDA5 import nnUNetTrainerDA5
from nnunetv2.training.nnUNetTrainer.variants.mixins._base import ComposableTrainerMixin
from nnunetv2.training.nnUNetTrainer.variants.mixins.aux_diagnosis_mixin import AuxiliaryDiagnosisMixin
from nnunetv2.training.nnUNetTrainer.variants.mixins.disease_conditioning import DiseaseConditioningMixin
from nnunetv2.training.nnUNetTrainer.variants.mixins.topology_loss import TopologyLossMixin


# ---------------------------------------------------------------------------
# Standalone auxiliary diagnosis head (no FiLM)
# ---------------------------------------------------------------------------

class nnUNetTrainerDA5AuxDiag(
    AuxiliaryDiagnosisMixin,
    ComposableTrainerMixin,
    nnUNetTrainerDA5,
):
    """DA5 + auxiliary diagnosis head at bottleneck.

    Requires disease_map.json in the preprocessed dataset folder.
    use_aux_embedding=False here — no FiLM to pass the embedding to.
    """
    use_aux_embedding = False


class nnUNetTrainerDA5AuxDiag_100epochs(nnUNetTrainerDA5AuxDiag):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 100


class nnUNetTrainerDA5AuxDiag_200epochs(nnUNetTrainerDA5AuxDiag):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 200


# ---------------------------------------------------------------------------
# Auxiliary diagnosis + topology loss
# ---------------------------------------------------------------------------

class nnUNetTrainerDA5AuxDiagTopo(
    AuxiliaryDiagnosisMixin,
    TopologyLossMixin,
    ComposableTrainerMixin,
    nnUNetTrainerDA5,
):
    """DA5 + aux diagnosis head + fixed-weight soft-clDice on AO/PA."""
    use_aux_embedding = False


class nnUNetTrainerDA5AuxDiagTopo_100epochs(nnUNetTrainerDA5AuxDiagTopo):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 100


class nnUNetTrainerDA5AuxDiagTopo_200epochs(nnUNetTrainerDA5AuxDiagTopo):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 200


# ---------------------------------------------------------------------------
# FiLM conditioning + auxiliary diagnosis head (embedding reuse)
# ---------------------------------------------------------------------------

class nnUNetTrainerDA5FiLMAuxDiag(
    AuxiliaryDiagnosisMixin,
    DiseaseConditioningMixin,
    ComposableTrainerMixin,
    nnUNetTrainerDA5,
):
    """DA5 + FiLM disease conditioning + aux diagnosis head at bottleneck.

    MRO order: AuxDiag first so mixin_prepare_forward can inject
    diagnosis_embedding AFTER DiseaseConditioningMixin has added disease_vec.
    use_aux_embedding=True exposes diagnosis_embedding in forward kwargs so
    FiLM-aware networks can consume it.

    Requires:
      - disease_map.json in preprocessed dataset folder
      - disease_wrapper_class / disease_param_prefixes set below
    """
    from nnunetv2.architectures.film_conditioned_unet import FiLMConditionedResEncUNet
    from nnunetv2.training.nnUNetTrainer.variants.mixins.disease_conditioning import (
        build_disease_conditioned_network,
    )

    disease_wrapper_class  = FiLMConditionedResEncUNet
    disease_param_prefixes = frozenset({'disease_mlp', 'bottleneck_film'})
    use_aux_embedding      = True

    def build_network_architecture(self, *args, **kwargs):
        from nnunetv2.architectures.film_conditioned_unet import FiLMConditionedResEncUNet
        from nnunetv2.training.nnUNetTrainer.variants.mixins.disease_conditioning import (
            build_disease_conditioned_network,
        )
        return build_disease_conditioned_network(self, FiLMConditionedResEncUNet, *args, **kwargs)


class nnUNetTrainerDA5FiLMAuxDiag_100epochs(nnUNetTrainerDA5FiLMAuxDiag):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 100


class nnUNetTrainerDA5FiLMAuxDiag_200epochs(nnUNetTrainerDA5FiLMAuxDiag):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 200
