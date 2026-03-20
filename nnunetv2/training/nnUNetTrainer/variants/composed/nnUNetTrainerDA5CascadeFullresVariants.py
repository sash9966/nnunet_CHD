"""Thin cascade-fullres trainer variants — one per lowres ablation.

Each class is architecturally identical to its parent but has a unique name so
nnU-Net writes results to a distinct folder, allowing side-by-side comparison
of the four low-res ablation variants:

  Baseline  : DA5 lowres            → nnUNetTrainerDA5CascadeFullresBaseline_100epochs
  FiLM      : DA5CascadeFiLM lowres → nnUNetTrainerDA5CascadeFullresFiLM_100epochs  (already in separate file)
  Topo      : DA5CascadeTopo lowres → nnUNetTrainerDA5CascadeFullresTopo_100epochs
  FiLM+Topo : DA5CascadeFiLMTopo   → nnUNetTrainerDA5CascadeFullresFiLMTopo_100epochs
"""
import torch

from nnunetv2.training.nnUNetTrainer.variants.data_augmentation.nnUNetTrainerDA5_epochs import (
    nnUNetTrainerDA5_100epochs,
)
from nnunetv2.training.nnUNetTrainer.variants.composed.nnUNetTrainerDA5CascadeFullresFiLM import (
    nnUNetTrainerDA5CascadeFullresFiLM_100epochs,
)


class nnUNetTrainerDA5CascadeFullresBaseline_100epochs(nnUNetTrainerDA5_100epochs):
    """Cascade-fullres counterpart for the plain DA5 baseline lowres stage."""
    pass


class nnUNetTrainerDA5CascadeFullresTopo_100epochs(nnUNetTrainerDA5_100epochs):
    """Cascade-fullres counterpart for the topology-loss-only lowres stage."""
    pass


class nnUNetTrainerDA5CascadeFullresFiLMTopo_100epochs(nnUNetTrainerDA5CascadeFullresFiLM_100epochs):
    """Cascade-fullres counterpart for the FiLM + topology-loss lowres stage.

    Inherits FiLM disease conditioning from nnUNetTrainerDA5CascadeFullresFiLM;
    unique name ensures results land in a separate folder from the FiLM-only run.
    """
    pass
