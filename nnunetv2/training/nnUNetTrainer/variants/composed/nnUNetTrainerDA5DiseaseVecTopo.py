"""Composable DA5 + MLP disease conditioning + topology loss trainer."""
import torch

from nnunetv2.architectures.disease_conditioned_unet import DiseaseConditionedResEncUNet
from nnunetv2.training.nnUNetTrainer.variants.data_augmentation.nnUNetTrainerDA5 import nnUNetTrainerDA5
from nnunetv2.training.nnUNetTrainer.variants.mixins._base import ComposableTrainerMixin
from nnunetv2.training.nnUNetTrainer.variants.mixins.disease_conditioning import (
    DiseaseConditioningMixin,
    build_disease_conditioned_network,
)
from nnunetv2.training.nnUNetTrainer.variants.mixins.topology_loss import TopologyLossMixin


class nnUNetTrainerDA5DiseaseVecTopo(DiseaseConditioningMixin, TopologyLossMixin, ComposableTrainerMixin, nnUNetTrainerDA5):
    disease_wrapper_class = DiseaseConditionedResEncUNet
    disease_param_prefixes = {'disease_mlp', 'bottleneck_injector', 'decoder_injectors', 'disease_classifier'}

    def build_network_architecture(self, *a, **kw):
        return build_disease_conditioned_network(self, DiseaseConditionedResEncUNet, *a, **kw)


class nnUNetTrainerDA5DiseaseVecTopo_100epochs(nnUNetTrainerDA5DiseaseVecTopo):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 100


class nnUNetTrainerDA5DiseaseVecTopo_200epochs(nnUNetTrainerDA5DiseaseVecTopo):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 200
