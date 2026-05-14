"""Composed trainers using the scheduled soft-clDice topology loss.

All trainers in this file use ``TopologyLossScheduledMixin`` which ramps
the clDice weight from 0 → topo_w_high over the first ``topo_warmup_epochs``
epochs, holds it, then cosine-decays to ``topo_w_low`` by end of training.
This avoids destabilising early training before Dice/CE has converged.

Default schedule (all overridable via subclass attributes):
    topo_warmup_epochs  = 10
    topo_decay_start    = 40
    topo_w_high         = 0.5
    topo_w_low          = 0.05

Trainer inventory
-----------------
3d_fullres:
    nnUNetTrainerDA5TopoScheduled
    nnUNetTrainerDA5TopoScheduled_100epochs

3d_lowres (cascade low-res stage):
    nnUNetTrainerDA5CascadeTopoScheduled
    nnUNetTrainerDA5CascadeTopoScheduled_100epochs

3d_cascade_fullres (cascade high-res stage):
    nnUNetTrainerDA5CascadeFullresTopoScheduled_100epochs
"""
import torch

from nnunetv2.training.nnUNetTrainer.variants.data_augmentation.nnUNetTrainerDA5 import nnUNetTrainerDA5
from nnunetv2.training.nnUNetTrainer.variants.mixins._base import ComposableTrainerMixin
from nnunetv2.training.nnUNetTrainer.variants.mixins.topology_loss import TopologyLossScheduledMixin


# ---------------------------------------------------------------------------
# 3d_fullres — full-resolution single-stage training
# ---------------------------------------------------------------------------

class nnUNetTrainerDA5TopoScheduled(
    TopologyLossScheduledMixin,
    ComposableTrainerMixin,
    nnUNetTrainerDA5,
):
    """DA5 augmentation + scheduled soft-clDice on AO/PA (full-res, standard epochs)."""
    pass


class nnUNetTrainerDA5TopoScheduled_100epochs(nnUNetTrainerDA5TopoScheduled):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 100


class nnUNetTrainerDA5TopoScheduled_200epochs(nnUNetTrainerDA5TopoScheduled):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 200


# ---------------------------------------------------------------------------
# 3d_lowres — cascade low-res stage
# ---------------------------------------------------------------------------

class nnUNetTrainerDA5CascadeTopoScheduled(
    TopologyLossScheduledMixin,
    ComposableTrainerMixin,
    nnUNetTrainerDA5,
):
    """Cascade low-res: DA5 + scheduled soft-clDice on AO/PA.

    At the low-res stage the full cardiac structure fits in one patch,
    so skeleton-based supervision is globally effective.  The schedule
    prevents early epochs from over-constraining the coarse vessel shape
    before the segmentation has stabilised.
    """
    pass


class nnUNetTrainerDA5CascadeTopoScheduled_100epochs(nnUNetTrainerDA5CascadeTopoScheduled):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 100


class nnUNetTrainerDA5CascadeTopoScheduled_200epochs(nnUNetTrainerDA5CascadeTopoScheduled):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 200


# ---------------------------------------------------------------------------
# 3d_cascade_fullres — cascade high-res stage
# ---------------------------------------------------------------------------

class nnUNetTrainerDA5CascadeFullresTopoScheduled(
    TopologyLossScheduledMixin,
    ComposableTrainerMixin,
    nnUNetTrainerDA5,
):
    """Cascade high-res: DA5 + scheduled soft-clDice on AO/PA.

    Intended as the fullres stage after nnUNetTrainerDA5CascadeTopoScheduled
    (lowres).  At full resolution the skeleton better captures thin vessel
    walls, so the scheduled topology loss refines local connectivity on top
    of the low-res prior.
    """
    pass


class nnUNetTrainerDA5CascadeFullresTopoScheduled_100epochs(nnUNetTrainerDA5CascadeFullresTopoScheduled):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 100


class nnUNetTrainerDA5CascadeFullresTopoScheduled_200epochs(nnUNetTrainerDA5CascadeFullresTopoScheduled):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 200
