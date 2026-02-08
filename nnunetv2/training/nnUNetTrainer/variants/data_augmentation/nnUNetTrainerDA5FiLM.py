"""
DA5 trainer variant with FiLM disease-vector conditioning.

Loads ``disease_map.json`` from the preprocessed dataset folder and injects
a per-sample disease vector into the network via Feature-wise Linear
Modulation (FiLM) at every forward pass.  When the JSON is absent, training
falls back to exact baseline behaviour.
"""
from __future__ import annotations

import json
from os.path import isfile, join
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch import autocast, nn
from torch._dynamo import OptimizedModule

from nnunetv2.architectures.film_conditioned_unet import FiLMConditionedResEncUNet
from nnunetv2.training.loss.dice import get_tp_fp_fn_tn
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.nnUNetTrainer.variants.data_augmentation.nnUNetTrainerDA5 import nnUNetTrainerDA5
from nnunetv2.utilities.helpers import dummy_context


class nnUNetTrainerDA5FiLM(nnUNetTrainerDA5):
    """nnUNetTrainerDA5 + FiLM disease-vector conditioning.

    Configuration knobs (override in subclass ``__init__`` if needed):

    ============  ========  ===================================================
    Attribute     Default   Meaning
    ============  ========  ===================================================
    disease_K     8         Length of the binary disease flag vector
    disease_H     64        Hidden dimension of the disease MLP
    disease_E     32        Output embedding dimension of the disease MLP
    ============  ========  ===================================================
    """

    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plans, configuration, fold, dataset_json, device)
        # disease conditioning hyper-parameters
        self.disease_K: int = 8
        self.disease_H: int = 64
        self.disease_E: int = 32
        # populated in initialize()
        self.disease_map: Optional[Dict[str, List[int]]] = None

    # ------------------------------------------------------------------
    # Disease map loading
    # ------------------------------------------------------------------
    def _load_disease_map(self) -> Optional[Dict[str, List[int]]]:
        """Load ``disease_map.json`` from the preprocessed dataset base folder.

        Returns ``None`` if the file does not exist (baseline mode).
        """
        path = join(self.preprocessed_dataset_folder_base, "disease_map.json")
        if not isfile(path):
            self.print_to_log_file(
                f"WARNING: {path} not found. Disease conditioning DISABLED (baseline mode)."
            )
            return None
        with open(path) as f:
            disease_map = json.load(f)
        self.print_to_log_file(
            f"Loaded disease_map.json with {len(disease_map)} entries from {path}"
        )
        for case_id, vec in disease_map.items():
            assert isinstance(vec, list) and len(vec) == self.disease_K, (
                f"disease_map entry for '{case_id}' has length {len(vec)}, expected {self.disease_K}"
            )
        return disease_map

    # ------------------------------------------------------------------
    # Network building  (override the @staticmethod as an instance method)
    # ------------------------------------------------------------------
    def build_network_architecture(  # type: ignore[override]
        self,
        architecture_class_name: str,
        arch_init_kwargs: dict,
        arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
        num_input_channels: int,
        num_output_channels: int,
        enable_deep_supervision: bool = True,
    ) -> nn.Module:
        """Build the base network then wrap it with FiLM conditioning."""
        base_network = nnUNetTrainer.build_network_architecture(
            architecture_class_name,
            arch_init_kwargs,
            arch_init_kwargs_req_import,
            num_input_channels,
            num_output_channels,
            enable_deep_supervision,
        )

        wrapped = FiLMConditionedResEncUNet(
            base_network,
            disease_K=self.disease_K,
            disease_H=self.disease_H,
            disease_E=self.disease_E,
        )

        # He-init the disease MLP (FiLM heads are already zero-init)
        wrapped.disease_mlp.apply(FiLMConditionedResEncUNet.initialize)

        return wrapped

    # ------------------------------------------------------------------
    # Initialization (load disease map after parent sets up everything)
    # ------------------------------------------------------------------
    def initialize(self):
        super().initialize()
        self.disease_map = self._load_disease_map()

    # ------------------------------------------------------------------
    # Helper: build disease_vec tensor for a batch
    # ------------------------------------------------------------------
    def _build_disease_vec(self, batch_keys: list) -> Optional[torch.Tensor]:
        """Return (B, K) float tensor or ``None`` if conditioning is disabled."""
        if self.disease_map is None:
            return None
        vecs = []
        for case_id in batch_keys:
            if case_id not in self.disease_map:
                raise KeyError(
                    f"Case '{case_id}' not found in disease_map.json. "
                    "All training/validation cases must be present."
                )
            vecs.append(self.disease_map[case_id])
        return torch.tensor(vecs, dtype=torch.float32, device=self.device)

    # ------------------------------------------------------------------
    # Helper: get unwrapped network module
    # ------------------------------------------------------------------
    def _get_unwrapped_network(self) -> nn.Module:
        mod = self.network
        if self.is_ddp:
            mod = mod.module
        if isinstance(mod, OptimizedModule):
            mod = mod._orig_mod
        return mod

    # ------------------------------------------------------------------
    # Training step
    # ------------------------------------------------------------------
    def train_step(self, batch: dict) -> dict:
        data = batch["data"]
        target = batch["target"]

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)

        disease_vec = self._build_disease_vec(batch["keys"])

        self.optimizer.zero_grad(set_to_none=True)

        with autocast(self.device.type, enabled=True) if self.device.type == "cuda" else dummy_context():
            output = self.network(data, disease_vec=disease_vec)
            l = self.loss(output, target)

        if self.grad_scaler is not None:
            self.grad_scaler.scale(l).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            l.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.optimizer.step()

        return {"loss": l.detach().cpu().numpy()}

    # ------------------------------------------------------------------
    # Validation step  (DA5 overrides this, so we override it here too)
    # ------------------------------------------------------------------
    def validation_step(self, batch: dict) -> dict:
        data = batch["data"]
        target = batch["target"]

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)

        disease_vec = self._build_disease_vec(batch["keys"])

        with autocast(self.device.type, enabled=True) if self.device.type == "cuda" else dummy_context():
            output = self.network(data, disease_vec=disease_vec)
            del data
            l = self.loss(output, target)

        if self.enable_deep_supervision:
            output = output[0]
            target = target[0]

        axes = [0] + list(range(2, output.ndim))

        if self.label_manager.has_regions:
            predicted_segmentation_onehot = (torch.sigmoid(output) > 0.5).long()
        else:
            output_seg = output.argmax(1)[:, None]
            predicted_segmentation_onehot = torch.zeros(output.shape, device=output.device, dtype=torch.float32)
            predicted_segmentation_onehot.scatter_(1, output_seg, 1)
            del output_seg

        if self.label_manager.has_ignore_label:
            if not self.label_manager.has_regions:
                mask = target != self.label_manager.ignore_label
                target[target == self.label_manager.ignore_label] = 0
            else:
                if target.dtype == torch.bool:
                    mask = ~target[:, -1:]
                else:
                    mask = (1 - target[:, -1:]).bool()
                target = target[:, :-1].bool()
        else:
            mask = None

        tp, fp, fn, _ = get_tp_fp_fn_tn(predicted_segmentation_onehot, target, axes=axes, mask=mask)

        tp_hard = tp.detach().cpu().numpy()
        fp_hard = fp.detach().cpu().numpy()
        fn_hard = fn.detach().cpu().numpy()
        if not self.label_manager.has_regions:
            tp_hard = tp_hard[1:]
            fp_hard = fp_hard[1:]
            fn_hard = fn_hard[1:]

        return {"loss": l.detach().cpu().numpy(), "tp_hard": tp_hard, "fp_hard": fp_hard, "fn_hard": fn_hard}

    # ------------------------------------------------------------------
    # Checkpoint loading  (handle missing FiLM keys gracefully)
    # ------------------------------------------------------------------
    def load_checkpoint(self, filename_or_checkpoint: Union[dict, str]) -> None:
        if not self.was_initialized:
            self.initialize()

        if isinstance(filename_or_checkpoint, str):
            checkpoint = torch.load(filename_or_checkpoint, map_location=self.device, weights_only=False)
        else:
            checkpoint = filename_or_checkpoint

        new_state_dict = {}
        for k, value in checkpoint["network_weights"].items():
            key = k
            if key not in self.network.state_dict().keys() and key.startswith("module."):
                key = key[7:]
            new_state_dict[key] = value

        self.my_init_kwargs = checkpoint["init_args"]
        self.current_epoch = checkpoint["current_epoch"]
        self.logger.load_checkpoint(checkpoint["logging"])
        self._best_ema = checkpoint["_best_ema"]
        self.inference_allowed_mirroring_axes = checkpoint.get(
            "inference_allowed_mirroring_axes", self.inference_allowed_mirroring_axes
        )

        mod = self._get_unwrapped_network()
        missing, unexpected = mod.load_state_dict(new_state_dict, strict=False)
        if missing:
            self.print_to_log_file(
                f"load_checkpoint: missing keys (will use init weights): {missing}"
            )
        if unexpected:
            self.print_to_log_file(
                f"load_checkpoint: unexpected keys (ignored): {unexpected}"
            )

        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        if self.grad_scaler is not None:
            if checkpoint["grad_scaler_state"] is not None:
                self.grad_scaler.load_state_dict(checkpoint["grad_scaler_state"])


class nnUNetTrainerDA5FiLM_100epochs(nnUNetTrainerDA5FiLM):
    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 100
