"""Versioned 200-epoch DA5 baseline for Dataset200 MRI (2026-09-18)."""
import torch
from nnunetv2.training.nnUNetTrainer.variants.data_augmentation.nnUNetTrainerDA5 import nnUNetTrainerDA5


class nnUNetTrainerDA5MRI200(nnUNetTrainerDA5):
    def __init__(self, plans: dict, configuration: str, fold: int,
                 dataset_json: dict, device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 200
