"""Backward-compatibility aliases for nnUNetTrainerDA5FiLMV2.

V2 has been superseded by V3 (bottleneck-only FiLM, no aux loss, K=8,
disease_lr_multiplier=2.0).  These aliases ensure existing checkpoints that
stored the V2 class name can still be loaded.
"""
from nnunetv2.training.nnUNetTrainer.variants.composed.nnUNetTrainerDA5FiLMV3 import (
    nnUNetTrainerDA5FiLMV3 as nnUNetTrainerDA5FiLMV2,
    nnUNetTrainerDA5FiLMV3_100epochs as nnUNetTrainerDA5FiLMV2_100epochs,
    nnUNetTrainerDA5FiLMV3_200epochs as nnUNetTrainerDA5FiLMV2_200epochs,
    nnUNetTrainerDA5FiLMV3_500epochs as nnUNetTrainerDA5FiLMV2_500epochs,
    nnUNetTrainerDA5FiLMV3_1000epochs as nnUNetTrainerDA5FiLMV2_1000epochs,
)

__all__ = [
    "nnUNetTrainerDA5FiLMV2",
    "nnUNetTrainerDA5FiLMV2_100epochs",
    "nnUNetTrainerDA5FiLMV2_200epochs",
    "nnUNetTrainerDA5FiLMV2_500epochs",
    "nnUNetTrainerDA5FiLMV2_1000epochs",
]
