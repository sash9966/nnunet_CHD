"""Cascade low-res trainer: DA5 + FiLM + spatial adjacency loss.

Experiment C in the cascade ablation.

STATUS: SpatialAdjacencyLossMixin is not yet implemented.
        This file is a forward-declaration placeholder.

When SpatialAdjacencyLossMixin is added to
    nnunetv2/training/nnUNetTrainer/variants/mixins/spatial_adjacency_loss.py

replace the stub below with the real class:

    from nnunetv2.training.nnUNetTrainer.variants.mixins.spatial_adjacency_loss import (
        SpatialAdjacencyLossMixin,
    )

    class nnUNetTrainerDA5CascadeFiLMAdjacency(
        DiseaseConditioningMixin,
        SpatialAdjacencyLossMixin,
        ComposableTrainerMixin,
        nnUNetTrainerDA5,
    ):
        disease_wrapper_class  = FiLMConditionedResEncUNet
        disease_param_prefixes = {'disease_mlp', 'bottleneck_film'}

        def build_network_architecture(self, *a, **kw):
            return build_disease_conditioned_network(self, FiLMConditionedResEncUNet, *a, **kw)

Design intent for SpatialAdjacencyLossMixin
-------------------------------------------
The spatial adjacency loss penalises predicted label pairs that violate known
cardiac anatomy constraints (e.g., LV adjacent to PA is forbidden in a normal
heart; RV adjacent to AO is forbidden except in DORV/TGA).  The loss is computed
from the predicted soft probabilities by checking the label-transition surface
against a disease-conditioned adjacency graph derived from NORMAL_ADJACENCY and
DISEASE_ADJACENCY_OVERRIDES in chd_postprocessing.config.

Suggested interface (mirror TopologyLossMixin):
    mixin_init()            — set adj_w_high, adj_w_low, adj_warmup_epochs
    mixin_initialize()      — build adjacency constraint matrix from dataset_json
    mixin_on_epoch_start()  — compute current weight via schedule
    mixin_extra_loss()      — return weighted adjacency violation term
"""
import torch

from nnunetv2.architectures.film_conditioned_unet import FiLMConditionedResEncUNet
from nnunetv2.training.nnUNetTrainer.variants.data_augmentation.nnUNetTrainerDA5 import nnUNetTrainerDA5
from nnunetv2.training.nnUNetTrainer.variants.mixins._base import ComposableTrainerMixin
from nnunetv2.training.nnUNetTrainer.variants.mixins.disease_conditioning import (
    DiseaseConditioningMixin,
    build_disease_conditioned_network,
)


# ---------------------------------------------------------------------------
# Placeholder — replace once SpatialAdjacencyLossMixin is implemented
# ---------------------------------------------------------------------------

class nnUNetTrainerDA5CascadeFiLMAdjacency(
    DiseaseConditioningMixin,
    ComposableTrainerMixin,
    nnUNetTrainerDA5,
):
    """Placeholder: DA5 + FiLM only (adjacency loss not yet wired in).

    Behaves identically to nnUNetTrainerDA5CascadeFiLM until
    SpatialAdjacencyLossMixin is implemented.
    """

    disease_wrapper_class  = FiLMConditionedResEncUNet
    disease_param_prefixes = {'disease_mlp', 'bottleneck_film'}

    def build_network_architecture(self, *a, **kw):
        return build_disease_conditioned_network(self, FiLMConditionedResEncUNet, *a, **kw)


class nnUNetTrainerDA5CascadeFiLMAdjacency_100epochs(nnUNetTrainerDA5CascadeFiLMAdjacency):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 100
