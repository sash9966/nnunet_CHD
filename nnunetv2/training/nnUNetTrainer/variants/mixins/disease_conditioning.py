"""
Disease conditioning mixin for composable trainers.

Provides ``DiseaseConditioningMixin`` which implements all disease-vector
related hooks:

- Loading ``disease_map.json`` and copying it to the model output folder.
- Building per-batch disease vectors for the forward pass.
- Extra optimizer param groups with a higher LR for disease components.
- Per-case disease vector setup during validation.
- Gradient diagnostic logging.

The concrete trainer must set the class attribute ``disease_wrapper_class``
to the appropriate wrapper (``FiLMConditionedResEncUNet`` or
``DiseaseConditionedResEncUNet``).

A shared helper ``build_disease_conditioned_network`` handles the
``isinstance(self, str)`` arg-shifting needed for both instance and static
(predictor) calls.
"""
from __future__ import annotations

import json
import pydoc
import shutil
from os.path import isfile, join
from typing import Dict, List, Optional, Type

import torch
from torch import nn

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.nnUNetTrainer.variants.mixins._base import TrainerMixin


# =========================================================================
# Shared helper for build_network_architecture
# =========================================================================

def build_disease_conditioned_network(
    self_or_arch_name,
    wrapper_cls,
    architecture_class_name=None,
    arch_init_kwargs=None,
    arch_init_kwargs_req_import=None,
    num_input_channels=None,
    num_output_channels=None,
    enable_deep_supervision: bool = True,
) -> nn.Module:
    """Build base network and wrap with disease conditioning.

    Handles the ``isinstance(self, str)`` arg-shifting for both instance
    calls (during training) and static calls (from nnUNetPredictor).

    Parameters
    ----------
    self_or_arch_name : trainer instance or str
        When called as an instance method, this is the trainer.
        When called statically by the predictor, this is the first
        positional arg (architecture_class_name).
    wrapper_cls : type
        The wrapper class (FiLMConditionedResEncUNet or
        DiseaseConditionedResEncUNet).
    """
    if isinstance(self_or_arch_name, str):
        # static call from predictor: self_or_arch_name is architecture_class_name
        (architecture_class_name, arch_init_kwargs, arch_init_kwargs_req_import,
         num_input_channels, num_output_channels) = (
            self_or_arch_name, architecture_class_name, arch_init_kwargs,
            arch_init_kwargs_req_import, num_input_channels
        )
        disease_K, disease_H, disease_E = 8, 64, 32
        trainer_instance = None
    else:
        trainer_instance = self_or_arch_name
        disease_K = trainer_instance.disease_K
        disease_H = trainer_instance.disease_H
        disease_E = trainer_instance.disease_E

    # 1) build vanilla network via the standard utility
    base_network = nnUNetTrainer.build_network_architecture(
        architecture_class_name,
        arch_init_kwargs,
        arch_init_kwargs_req_import,
        num_input_channels,
        num_output_channels,
        enable_deep_supervision,
    )

    # 2) check if wrapper needs conv/norm/nonlin ops (MLP-style wrapper)
    import inspect
    wrapper_init_params = inspect.signature(wrapper_cls.__init__).parameters
    needs_conv_ops = "conv_op" in wrapper_init_params

    extra_kwargs = {}
    if needs_conv_ops:
        resolved_kwargs = dict(**arch_init_kwargs)
        for key in arch_init_kwargs_req_import:
            if resolved_kwargs[key] is not None:
                resolved_kwargs[key] = pydoc.locate(resolved_kwargs[key])
        extra_kwargs = dict(
            conv_op=resolved_kwargs["conv_op"],
            norm_op=resolved_kwargs["norm_op"],
            norm_op_kwargs=resolved_kwargs.get("norm_op_kwargs", {"eps": 1e-5, "affine": True}),
            nonlin=resolved_kwargs["nonlin"],
            nonlin_kwargs=resolved_kwargs.get("nonlin_kwargs", {"inplace": True}),
        )

    # 3) wrap with disease conditioning
    wrapped = wrapper_cls(
        base_network,
        disease_K=disease_K,
        disease_H=disease_H,
        disease_E=disease_E,
        **extra_kwargs,
    )

    # 4) init disease modules
    wrapped.disease_mlp.apply(wrapper_cls.initialize)
    if hasattr(wrapped, "bottleneck_injector"):
        wrapped.bottleneck_injector.apply(wrapper_cls.initialize)
    if hasattr(wrapped, "decoder_injectors"):
        wrapped.decoder_injectors.apply(wrapper_cls.initialize)
    # Note: SpatialGateLayer (bottleneck_gate, decoder_gates) has its own
    # zero-init scheme in __init__ — do NOT apply He init to those modules.

    return wrapped


# =========================================================================
# DiseaseConditioningMixin
# =========================================================================

class DiseaseConditioningMixin(TrainerMixin):
    """Mixin that adds disease-vector conditioning to a composable trainer.

    The concrete trainer class must set:
        ``disease_wrapper_class`` — the wrapper class to use
        ``disease_param_prefixes`` — set of param name prefixes for the
            disease components (used for LR splitting and grad diagnostics)
    """

    # These are defaults; subclasses can override.
    disease_wrapper_class = None  # must be set on concrete class
    disease_param_prefixes: set = frozenset()  # set on concrete class

    def mixin_init(self):
        super().mixin_init()
        self.disease_K: int = 8
        self.disease_H: int = 64
        self.disease_E: int = 32
        self.disease_lr_multiplier: float = 10.0
        self.aux_disease_loss_weight: float = 0.1
        self.disease_map: Optional[Dict[str, List[int]]] = None
        self.disease_dropout_prob: float = 0.0  # classifier-free guidance dropout

    def mixin_initialize(self):
        super().mixin_initialize()
        self.disease_map = self._load_disease_map()

    # ------------------------------------------------------------------
    # Disease map loading
    # ------------------------------------------------------------------
    def _load_disease_map(self) -> Optional[Dict[str, List[int]]]:
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
    # Build disease_vec tensor for a batch
    # ------------------------------------------------------------------
    def _build_disease_vec(self, batch_keys: list) -> Optional[torch.Tensor]:
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
    # Hook: prepare forward kwargs
    # ------------------------------------------------------------------
    def mixin_prepare_forward(self, batch: dict) -> dict:
        kw = super().mixin_prepare_forward(batch)
        disease_vec = self._build_disease_vec(batch["keys"])
        if disease_vec is not None:
            # Classifier-free guidance dropout: randomly drop the disease
            # vector during training so the model learns to *need* it.
            if (self.disease_dropout_prob > 0
                    and self.network.training
                    and torch.rand(1).item() < self.disease_dropout_prob):
                disease_vec = None
        if disease_vec is not None:
            kw["disease_vec"] = disease_vec
        return kw

    # ------------------------------------------------------------------
    # Hook: extra loss (auxiliary disease classification)
    # ------------------------------------------------------------------
    def mixin_extra_loss(self, output, target, batch, **forward_kwargs) -> float:
        extra = super().mixin_extra_loss(output, target, batch, **forward_kwargs)
        aux_logits = forward_kwargs.get("aux_logits")
        disease_vec = forward_kwargs.get("disease_vec")
        if aux_logits is not None and disease_vec is not None:
            aux_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                aux_logits, disease_vec
            )
            extra = extra + self.aux_disease_loss_weight * aux_loss
        return extra

    # ------------------------------------------------------------------
    # Hook: extra optimizer param groups
    # ------------------------------------------------------------------
    def mixin_param_groups(self) -> List[dict]:
        groups = super().mixin_param_groups()
        prefixes = self.disease_param_prefixes
        if not prefixes:
            return groups

        disease_params = []
        for name, param in self.network.named_parameters():
            if any(name.startswith(prefix) or f'.{prefix}' in name for prefix in prefixes):
                disease_params.append(param)

        if disease_params:
            groups.append({
                "params": disease_params,
                "lr": self.initial_lr * self.disease_lr_multiplier,
            })
        return groups

    # ------------------------------------------------------------------
    # Hook: fix LR after scheduler
    # ------------------------------------------------------------------
    def mixin_fix_lr_after_scheduler(self):
        super().mixin_fix_lr_after_scheduler()
        # PolyLRScheduler sets all param groups to the same LR.
        # Re-apply the multiplier to group index 1 (the disease param group).
        if len(self.optimizer.param_groups) > 1:
            self.optimizer.param_groups[1]["lr"] *= self.disease_lr_multiplier

    # ------------------------------------------------------------------
    # Hook: on_train_start (copy disease_map.json)
    # ------------------------------------------------------------------
    def mixin_on_train_start(self):
        super().mixin_on_train_start()
        src = join(self.preprocessed_dataset_folder_base, "disease_map.json")
        if isfile(src):
            dst = join(self.output_folder_base, "disease_map.json")
            shutil.copy2(src, dst)
            self.print_to_log_file(f"Copied disease_map.json to {dst}")

    # ------------------------------------------------------------------
    # Hook: gradient diagnostics
    # ------------------------------------------------------------------
    def mixin_on_train_epoch_end(self, train_outputs):
        super().mixin_on_train_epoch_end(train_outputs)
        mod = self._get_unwrapped_network()

        # -- gradient diagnostics (every 10 epochs) --
        if self.current_epoch % 10 == 0:
            prefixes = self.disease_param_prefixes
            disease_grad = 0.0
            total_grad = 0.0
            for name, param in mod.named_parameters():
                if param.grad is not None:
                    g = param.grad.norm().item()
                    total_grad += g
                    if any(name.startswith(prefix) for prefix in prefixes):
                        disease_grad += g
            if total_grad > 0:
                pct = 100 * disease_grad / total_grad
                self.print_to_log_file(
                    f"[Grad diagnostic] Disease components: {disease_grad:.4f} / "
                    f"Total: {total_grad:.2f} = {pct:.2f}%"
                )

        # -- FiLM modulation magnitude logging (every epoch, FiLM only) --
        if hasattr(mod, 'get_film_magnitudes'):
            magnitudes = mod.get_film_magnitudes()
            if magnitudes:
                self.logger.log('film_magnitudes', magnitudes, self.current_epoch)
                # text log every 10 epochs
                if self.current_epoch % 10 == 0:
                    parts = []
                    for layer_name, vals in sorted(magnitudes.items()):
                        parts.append(
                            f"{layer_name}: |γ|={vals['gamma']:.4f} |β|={vals['beta']:.4f}"
                        )
                    self.print_to_log_file(
                        f"[FiLM magnitudes] {', '.join(parts)}"
                    )

    # ------------------------------------------------------------------
    # Validation hooks: set/clear disease vec per case
    # ------------------------------------------------------------------
    def before_validation_case(self, case_id: str):
        super().before_validation_case(case_id)
        mod = self._get_unwrapped_network()
        if self.disease_map is not None and case_id in self.disease_map:
            vec = torch.tensor([self.disease_map[case_id]], dtype=torch.float32, device=self.device)
            mod.set_disease_vec(vec)
        else:
            mod.clear_disease_vec()

    def after_validation_case(self, case_id: str):
        super().after_validation_case(case_id)
        self._get_unwrapped_network().clear_disease_vec()
