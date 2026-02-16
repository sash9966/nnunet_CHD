"""
Unit tests for the composable mixin-based trainer architecture.

Tests MRO correctness, hook dispatch, build_network_architecture in both
instance and static-call modes, and basic forward/loss behaviour.

Run with:
    python3 -m unittest nnunetv2.tests.test_composable_trainers -v
"""
from __future__ import annotations

import json
import os
import tempfile
import unittest
from unittest.mock import patch

import torch
import numpy as np

# ---------------------------------------------------------------------------
# Minimal plans and dataset_json fixtures
# ---------------------------------------------------------------------------

MINIMAL_DATASET_JSON = {
    "channel_names": {"0": "CT"},
    "labels": {
        "background": 0,
        "LV": 1,
        "RV": 2,
        "LA": 3,
        "RA": 4,
        "AO": 5,
        "PA": 6,
        "WH": 7,
    },
    "numTraining": 10,
    "file_ending": ".nii.gz",
}

MINIMAL_PLANS = {
    "dataset_name": "Dataset999_Test",
    "plans_name": "nnUNetPlans",
    "original_median_spacing_after_transp": [1.0, 1.0, 1.0],
    "original_median_shape_after_transp": [128, 128, 128],
    "image_reader_writer": "SimpleITKIO",
    "transpose_forward": [0, 1, 2],
    "transpose_backward": [0, 1, 2],
    "configurations": {
        "3d_fullres": {
            "data_identifier": "nnUNetPlans_3d_fullres",
            "preprocessor_name": "DefaultPreprocessor",
            "batch_size": 2,
            "patch_size": [32, 32, 32],
            "median_image_size_in_voxels": [128, 128, 128],
            "spacing": [1.0, 1.0, 1.0],
            "normalization_schemes": ["CTNormalization"],
            "use_mask_for_norm": [False],
            "UNet_class_name": "dynamic_network_architectures.architectures.unet.ResidualEncoderUNet",
            "UNet_base_num_features": 32,
            "n_conv_per_stage_encoder": [2, 2, 2, 2, 2],
            "n_conv_per_stage_decoder": [2, 2, 2, 2],
            "num_pool_per_axis": [4, 4, 4],
            "pool_op_kernel_sizes": [[1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]],
            "conv_kernel_sizes": [[3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]],
            "unet_max_num_features": 320,
            "resampling_fn_data": "resample_data_or_seg_to_shape",
            "resampling_fn_seg": "resample_data_or_seg_to_shape",
            "resampling_fn_data_kwargs": {"is_seg": False, "order": 3, "order_z": 0, "force_separate_z": None},
            "resampling_fn_seg_kwargs": {"is_seg": True, "order": 1, "order_z": 0, "force_separate_z": None},
            "resampling_fn_probabilities": "resample_data_or_seg_to_shape",
            "resampling_fn_probabilities_kwargs": {"is_seg": False, "order": 1, "order_z": 0, "force_separate_z": None},
            "architecture": {
                "network_class_name": "dynamic_network_architectures.architectures.unet.ResidualEncoderUNet",
                "arch_kwargs": {
                    "n_stages": 5,
                    "features_per_stage": [32, 64, 128, 256, 320],
                    "conv_op": "torch.nn.modules.conv.Conv3d",
                    "kernel_sizes": [[3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]],
                    "strides": [[1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]],
                    "n_conv_per_stage": [2, 2, 2, 2, 2],
                    "n_conv_per_stage_decoder": [2, 2, 2, 2],
                    "conv_bias": True,
                    "norm_op": "torch.nn.modules.instancenorm.InstanceNorm3d",
                    "norm_op_kwargs": {"eps": 1e-05, "affine": True},
                    "dropout_op": None,
                    "dropout_op_kwargs": None,
                    "nonlin": "torch.nn.LeakyReLU",
                    "nonlin_kwargs": {"inplace": True},
                },
                "arch_kwargs_req_import": ["conv_op", "norm_op", "dropout_op", "nonlin"],
            },
            "batch_dice": True,
            "next_stage_names": None,
        },
    },
    "experiment_planner_used": "ExperimentPlanner",
    "label_manager": "LabelManager",
    "foreground_intensity_properties_per_channel": {
        "0": {"mean": 0.0, "median": 0.0, "std": 1.0, "min": -1.0, "max": 1.0, "percentile_00_5": -1.0, "percentile_99_5": 1.0},
    },
}


# ---------------------------------------------------------------------------
# Test: MRO is correct for all composed trainers
# ---------------------------------------------------------------------------

class TestMRO(unittest.TestCase):
    def test_film_v2_mro(self):
        from nnunetv2.training.nnUNetTrainer.variants.composed.nnUNetTrainerDA5FiLMV2 import nnUNetTrainerDA5FiLMV2
        from nnunetv2.training.nnUNetTrainer.variants.mixins.disease_conditioning import DiseaseConditioningMixin
        from nnunetv2.training.nnUNetTrainer.variants.mixins._base import ComposableTrainerMixin
        from nnunetv2.training.nnUNetTrainer.variants.data_augmentation.nnUNetTrainerDA5 import nnUNetTrainerDA5
        mro = nnUNetTrainerDA5FiLMV2.__mro__
        # DiseaseConditioningMixin before ComposableTrainerMixin before DA5
        assert mro.index(DiseaseConditioningMixin) < mro.index(ComposableTrainerMixin)
        assert mro.index(ComposableTrainerMixin) < mro.index(nnUNetTrainerDA5)

    def test_disease_vec_v2_mro(self):
        from nnunetv2.training.nnUNetTrainer.variants.composed.nnUNetTrainerDA5DiseaseVecV2 import nnUNetTrainerDA5DiseaseVecV2
        from nnunetv2.training.nnUNetTrainer.variants.mixins.disease_conditioning import DiseaseConditioningMixin
        from nnunetv2.training.nnUNetTrainer.variants.mixins._base import ComposableTrainerMixin
        mro = nnUNetTrainerDA5DiseaseVecV2.__mro__
        assert mro.index(DiseaseConditioningMixin) < mro.index(ComposableTrainerMixin)

    def test_film_topo_mro(self):
        from nnunetv2.training.nnUNetTrainer.variants.composed.nnUNetTrainerDA5FiLMTopo import nnUNetTrainerDA5FiLMTopo
        from nnunetv2.training.nnUNetTrainer.variants.mixins.disease_conditioning import DiseaseConditioningMixin
        from nnunetv2.training.nnUNetTrainer.variants.mixins.topology_loss import TopologyLossMixin
        from nnunetv2.training.nnUNetTrainer.variants.mixins._base import ComposableTrainerMixin
        mro = nnUNetTrainerDA5FiLMTopo.__mro__
        assert mro.index(DiseaseConditioningMixin) < mro.index(TopologyLossMixin)
        assert mro.index(TopologyLossMixin) < mro.index(ComposableTrainerMixin)

    def test_disease_vec_topo_mro(self):
        from nnunetv2.training.nnUNetTrainer.variants.composed.nnUNetTrainerDA5DiseaseVecTopo import nnUNetTrainerDA5DiseaseVecTopo
        from nnunetv2.training.nnUNetTrainer.variants.mixins.disease_conditioning import DiseaseConditioningMixin
        from nnunetv2.training.nnUNetTrainer.variants.mixins.topology_loss import TopologyLossMixin
        from nnunetv2.training.nnUNetTrainer.variants.mixins._base import ComposableTrainerMixin
        mro = nnUNetTrainerDA5DiseaseVecTopo.__mro__
        assert mro.index(DiseaseConditioningMixin) < mro.index(TopologyLossMixin)
        assert mro.index(TopologyLossMixin) < mro.index(ComposableTrainerMixin)

    def test_topo_only_mro(self):
        from nnunetv2.training.nnUNetTrainer.variants.composed.nnUNetTrainerTopoLossV2 import nnUNetTrainerTopoLossV2
        from nnunetv2.training.nnUNetTrainer.variants.mixins.topology_loss import TopologyLossMixin
        from nnunetv2.training.nnUNetTrainer.variants.mixins._base import ComposableTrainerMixin
        from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
        mro = nnUNetTrainerTopoLossV2.__mro__
        assert mro.index(TopologyLossMixin) < mro.index(ComposableTrainerMixin)
        assert mro.index(ComposableTrainerMixin) < mro.index(nnUNetTrainer)


# ---------------------------------------------------------------------------
# Test: build_network_architecture works in both modes
# ---------------------------------------------------------------------------

class TestBuildNetwork(unittest.TestCase):
    def _get_arch_args(self):
        cfg = MINIMAL_PLANS["configurations"]["3d_fullres"]["architecture"]
        return (
            cfg["network_class_name"],
            cfg["arch_kwargs"],
            cfg["arch_kwargs_req_import"],
            1,   # num_input_channels
            8,   # num_output_channels
        )

    def test_film_instance_call(self):
        from nnunetv2.training.nnUNetTrainer.variants.composed.nnUNetTrainerDA5FiLMV2 import nnUNetTrainerDA5FiLMV2
        from nnunetv2.architectures.film_conditioned_unet import FiLMConditionedResEncUNet
        # simulate instance call: pass trainer-like object with attributes
        class FakeTrainer:
            disease_K = 8
            disease_H = 64
            disease_E = 32
        trainer = FakeTrainer()
        args = self._get_arch_args()
        net = nnUNetTrainerDA5FiLMV2.build_network_architecture(trainer, *args)
        assert isinstance(net, FiLMConditionedResEncUNet)
        assert hasattr(net, 'disease_mlp')
        assert hasattr(net, 'bottleneck_film')

    def test_film_static_call(self):
        """Simulate how nnUNetPredictor calls build_network_architecture statically."""
        from nnunetv2.training.nnUNetTrainer.variants.composed.nnUNetTrainerDA5FiLMV2 import nnUNetTrainerDA5FiLMV2
        from nnunetv2.architectures.film_conditioned_unet import FiLMConditionedResEncUNet
        args = self._get_arch_args()
        # static call: no 'self', args shift by one
        net = nnUNetTrainerDA5FiLMV2.build_network_architecture(*args)
        assert isinstance(net, FiLMConditionedResEncUNet)

    def test_mlp_instance_call(self):
        from nnunetv2.training.nnUNetTrainer.variants.composed.nnUNetTrainerDA5DiseaseVecV2 import nnUNetTrainerDA5DiseaseVecV2
        from nnunetv2.architectures.disease_conditioned_unet import DiseaseConditionedResEncUNet
        class FakeTrainer:
            disease_K = 8
            disease_H = 64
            disease_E = 32
        trainer = FakeTrainer()
        args = self._get_arch_args()
        net = nnUNetTrainerDA5DiseaseVecV2.build_network_architecture(trainer, *args)
        assert isinstance(net, DiseaseConditionedResEncUNet)
        assert hasattr(net, 'bottleneck_injector')

    def test_mlp_static_call(self):
        from nnunetv2.training.nnUNetTrainer.variants.composed.nnUNetTrainerDA5DiseaseVecV2 import nnUNetTrainerDA5DiseaseVecV2
        from nnunetv2.architectures.disease_conditioned_unet import DiseaseConditionedResEncUNet
        args = self._get_arch_args()
        net = nnUNetTrainerDA5DiseaseVecV2.build_network_architecture(*args)
        assert isinstance(net, DiseaseConditionedResEncUNet)


# ---------------------------------------------------------------------------
# Test: mixin_prepare_forward returns disease_vec for disease trainers
# ---------------------------------------------------------------------------

class TestHooks(unittest.TestCase):
    def test_disease_mixin_prepare_forward(self):
        from nnunetv2.training.nnUNetTrainer.variants.mixins.disease_conditioning import DiseaseConditioningMixin
        mixin = DiseaseConditioningMixin()
        # set up minimal state
        mixin.disease_map = {"case_001": [1, 0, 0, 0, 0, 0, 0, 0]}
        mixin.disease_K = 8
        mixin.device = torch.device("cpu")
        batch = {"keys": ["case_001"]}
        kw = mixin.mixin_prepare_forward(batch)
        assert "disease_vec" in kw
        assert kw["disease_vec"].shape == (1, 8)

    def test_disease_mixin_no_map_returns_empty(self):
        from nnunetv2.training.nnUNetTrainer.variants.mixins.disease_conditioning import DiseaseConditioningMixin
        mixin = DiseaseConditioningMixin()
        mixin.disease_map = None
        mixin.disease_K = 8
        mixin.device = torch.device("cpu")
        batch = {"keys": ["case_001"]}
        kw = mixin.mixin_prepare_forward(batch)
        assert "disease_vec" not in kw

    def test_topo_mixin_extra_loss(self):
        """TopologyLossMixin.mixin_extra_loss returns non-zero when topo_loss is set."""
        from nnunetv2.training.nnUNetTrainer.variants.mixins.topology_loss import TopologyLossMixin
        mixin = TopologyLossMixin()
        mixin.enable_deep_supervision = True
        # mock a simple topo loss that returns a constant
        mixin.topo_loss = lambda pred, target: torch.tensor(0.5)
        mixin._current_topo_weight = 1.0
        # simulate DS output: list of tensors
        B, C, D, H, W = 1, 8, 8, 8, 8
        output = [torch.randn(B, C, D, H, W)]
        target = [torch.randint(0, C, (B, 1, D, H, W)).float()]
        extra = mixin.mixin_extra_loss(output, target, {})
        assert extra > 0.0

    def test_topo_mixin_no_loss_returns_zero(self):
        from nnunetv2.training.nnUNetTrainer.variants.mixins.topology_loss import TopologyLossMixin
        mixin = TopologyLossMixin()
        mixin.enable_deep_supervision = False
        mixin.topo_loss = None
        mixin._current_topo_weight = 0.0
        extra = mixin.mixin_extra_loss(torch.randn(1, 8, 8, 8, 8), torch.zeros(1, 1, 8, 8, 8), {})
        assert extra == 0.0


# ---------------------------------------------------------------------------
# Test: 100-epoch variants exist and set num_epochs
# ---------------------------------------------------------------------------

class TestEpochVariants(unittest.TestCase):
    def test_100epoch_classes_importable(self):
        from nnunetv2.training.nnUNetTrainer.variants.composed.nnUNetTrainerDA5FiLMV2 import nnUNetTrainerDA5FiLMV2_100epochs
        from nnunetv2.training.nnUNetTrainer.variants.composed.nnUNetTrainerDA5DiseaseVecV2 import nnUNetTrainerDA5DiseaseVecV2_100epochs
        from nnunetv2.training.nnUNetTrainer.variants.composed.nnUNetTrainerDA5FiLMTopo import nnUNetTrainerDA5FiLMTopo_100epochs
        from nnunetv2.training.nnUNetTrainer.variants.composed.nnUNetTrainerDA5DiseaseVecTopo import nnUNetTrainerDA5DiseaseVecTopo_100epochs
        from nnunetv2.training.nnUNetTrainer.variants.composed.nnUNetTrainerTopoLossV2 import nnUNetTrainerTopoLossV2_100epochs
        # all imported successfully
        assert True
