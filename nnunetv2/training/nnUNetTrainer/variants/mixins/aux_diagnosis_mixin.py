"""
Auxiliary diagnosis head mixin for composable trainers.

Provides ``AuxiliaryDiagnosisMixin`` which attaches a small MLP classification
head to the encoder bottleneck features and trains it to predict CHD diagnosis
labels alongside the main segmentation objective.

Two-benefit rationale
---------------------
1. **Encoder regularisation**: auxiliary supervision at the bottleneck forces
   the encoder to produce features that are predictive of CHD diagnosis type,
   making representations more semantically meaningful and acting as a
   regulariser against overfitting on small clinical datasets.

2. **Embedding reuse**: the penultimate 256-d hidden state (``diagnosis_embedding``)
   is a richer learned encoding of disease type than the raw K=8 one-hot input
   vector.  When combined with ``DiseaseConditioningMixin`` and
   ``use_aux_embedding=True``, this embedding is injected into forward kwargs
   so the FiLM / conditioning network can consume it directly rather than
   re-encoding from scratch.

Usage
-----
Standalone (no FiLM)::

    class nnUNetTrainerDA5AuxDiag(
        AuxiliaryDiagnosisMixin, ComposableTrainerMixin, nnUNetTrainerDA5
    ):
        pass

Combined with FiLM::

    class nnUNetTrainerDA5FiLMAuxDiag(
        AuxiliaryDiagnosisMixin, DiseaseConditioningMixin,
        ComposableTrainerMixin, nnUNetTrainerDA5
    ):
        disease_wrapper_class  = FiLMConditionedResEncUNet
        disease_param_prefixes = {'disease_mlp', 'bottleneck_film'}
        use_aux_embedding      = True   # pass diagnosis_embedding in forward kwargs
"""
from __future__ import annotations

import json
from os.path import isfile, join
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from nnunetv2.training.nnUNetTrainer.variants.mixins._base import TrainerMixin


class AuxiliaryDiagnosisMixin(TrainerMixin):
    """Bottleneck classification head with BCEWithLogitsLoss auxiliary supervision.

    Class attributes (override on concrete trainer)
    ------------------------------------------------
    use_aux_embedding : bool
        When True and ``DiseaseConditioningMixin`` is also in the MRO, the
        lagged ``diagnosis_embedding`` (256-d) is injected into forward kwargs
        under the key ``"diagnosis_embedding"`` so the conditioning network can
        use it instead of re-encoding the raw disease vector.
    aux_loss_weight : float
        Scalar weight applied to the auxiliary BCE loss before adding to the
        segmentation loss.  Default: 0.1.
    aux_lr_multiplier : float
        Learning rate multiplier for the diagnosis head parameters relative to
        the main backbone LR.  Default: 3.0.
    aux_num_classes : int
        Number of disease classes.  Defaults to ``self.disease_K`` if available
        (set by DiseaseConditioningMixin), otherwise 8.
    """

    # ── Class-level defaults (override on concrete class) ────────────────
    use_aux_embedding: bool = True
    aux_loss_weight: float = 0.1
    aux_lr_multiplier: float = 3.0
    aux_num_classes: int = 8   # overridden by disease_K when DiseaseConditioningMixin present

    # ── Hook: initialise mixin state ─────────────────────────────────────
    def mixin_init(self):
        super().mixin_init()
        self._diagnosis_head: Optional[nn.Module] = None
        self._hook_handle = None
        self._bottleneck_features: Optional[torch.Tensor] = None
        # Exposed for downstream mixins / logging
        self.diagnosis_embedding: Optional[torch.Tensor] = None
        # Accumulated per epoch for logging
        self._aux_loss_accum: List[float] = []
        self._aux_pred_probs_accum: List[torch.Tensor] = []
        # Separate disease map (only loaded when DiseaseConditioningMixin absent)
        self._aux_disease_map: Optional[Dict[str, List[int]]] = None

    # ── Hook: build head + hook after network is ready ───────────────────
    def mixin_initialize(self):
        super().mixin_initialize()

        K = getattr(self, 'disease_K', self.aux_num_classes)
        mod = self._get_unwrapped_network()
        bottleneck_ch = self._get_bottleneck_channels(mod)

        # ── Classification head: bottleneck_ch → 256 → K ─────────────────
        self._diagnosis_head = nn.Sequential(
            nn.Linear(bottleneck_ch, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, K),
        ).to(self.device)

        # Kaiming init (He) for both linear layers, zeros for biases
        for layer in self._diagnosis_head:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
                nn.init.zeros_(layer.bias)

        # ── Forward hook on encoder bottleneck ───────────────────────────
        self._register_bottleneck_hook(mod)

        # ── Disease map (fallback when DiseaseConditioningMixin absent) ───
        if not getattr(self, 'disease_map', None):
            self._aux_disease_map = self._load_aux_disease_map()

        self.print_to_log_file(
            f"[AuxDiagMixin] head: {bottleneck_ch} → 256 → {K}  "
            f"| aux_weight={self.aux_loss_weight}  lr_mult={self.aux_lr_multiplier}x  "
            f"| use_aux_embedding={self.use_aux_embedding}"
        )

    # ── Hook: inject diagnosis_embedding into forward kwargs ─────────────
    def mixin_prepare_forward(self, batch: dict) -> dict:
        kw = super().mixin_prepare_forward(batch)
        # Inject the lagged (previous-step) embedding when use_aux_embedding is
        # enabled.  Downstream networks and mixins can consume
        # kwargs["diagnosis_embedding"] if they support it.
        if self.use_aux_embedding and self.diagnosis_embedding is not None:
            kw["diagnosis_embedding"] = self.diagnosis_embedding
        return kw

    # ── Hook: auxiliary BCE loss ──────────────────────────────────────────
    def mixin_extra_loss(self, output, target, batch: dict, **forward_kwargs) -> float:
        extra = super().mixin_extra_loss(output, target, batch, **forward_kwargs)

        if self._bottleneck_features is None or self._diagnosis_head is None:
            return extra

        disease_vec = self._resolve_disease_vec(batch["keys"])
        if disease_vec is None:
            return extra

        # ── Spatial average pool: (B, C, [D,] H, W) → (B, C) ─────────────
        feats = self._bottleneck_features.float()  # ensure fp32 under AMP
        if feats.dim() > 2:
            feats = feats.mean(dim=list(range(2, feats.dim())))  # (B, C)

        # ── Forward through head in two steps to capture penultimate state ─
        hidden = self._diagnosis_head[0](feats)   # Linear → (B, 256)
        hidden = self._diagnosis_head[1](hidden)  # ReLU   → (B, 256)
        logits = self._diagnosis_head[2](hidden)  # Linear → (B, K)

        # Expose 256-d embedding for downstream mixin reuse (detached)
        self.diagnosis_embedding = hidden.detach()

        # ── Auxiliary BCE loss (multi-label) ──────────────────────────────
        aux_loss = F.binary_cross_entropy_with_logits(logits, disease_vec)
        weighted = self.aux_loss_weight * aux_loss

        self._aux_loss_accum.append(aux_loss.item())

        # Accumulate per-class mean sigmoid probability for progress plot
        with torch.no_grad():
            probs = torch.sigmoid(logits.detach().float()).mean(dim=0).cpu()
        self._aux_pred_probs_accum.append(probs)

        return extra + weighted

    # ── Hook: extra param group for the diagnosis head ───────────────────
    def mixin_param_groups(self) -> List[dict]:
        groups = super().mixin_param_groups()
        if self._diagnosis_head is not None:
            groups.append({
                "params": list(self._diagnosis_head.parameters()),
                "lr": self.initial_lr * self.aux_lr_multiplier,
            })
        return groups

    # ── Hook: re-apply LR multiplier after PolyLR scheduler step ─────────
    def mixin_fix_lr_after_scheduler(self):
        super().mixin_fix_lr_after_scheduler()
        if self._diagnosis_head is None:
            return
        # Match by param ids — avoids hardcoded group indices so it composes
        # correctly when DiseaseConditioningMixin adds its own extra group.
        head_ids = {id(p) for p in self._diagnosis_head.parameters()}
        main_lr = self.optimizer.param_groups[0]["lr"]
        for group in self.optimizer.param_groups[1:]:
            if {id(p) for p in group["params"]} == head_ids:
                group["lr"] = main_lr * self.aux_lr_multiplier
                break

    # ── Hook: log aux loss summary at epoch end ───────────────────────────
    def mixin_on_train_epoch_end(self, train_outputs):
        super().mixin_on_train_epoch_end(train_outputs)
        if self._aux_loss_accum:
            avg_aux = sum(self._aux_loss_accum) / len(self._aux_loss_accum)
            avg_seg = float(
                sum(o["loss"] for o in train_outputs) / len(train_outputs)
            ) if train_outputs else float("nan")
            self.print_to_log_file(
                f"[AuxDiag] epoch {self.current_epoch}  "
                f"seg_loss={avg_seg:.4f}  "
                f"aux_bce={avg_aux:.4f}  "
                f"aux_weighted={self.aux_loss_weight * avg_aux:.4f}"
            )

            # Log to logger for progress.png panel
            mean_probs = (
                torch.stack(self._aux_pred_probs_accum).mean(dim=0).tolist()
                if self._aux_pred_probs_accum else []
            )
            self.logger.log(
                'aux_diagnosis',
                {'bce': avg_aux, 'probs': mean_probs},
                self.current_epoch,
            )

            self._aux_loss_accum.clear()
            self._aux_pred_probs_accum.clear()

    # ── Private helpers ───────────────────────────────────────────────────

    def _get_bottleneck_channels(self, mod: nn.Module) -> int:
        """Return the number of channels at the encoder bottleneck."""
        # Handle FiLM / Gated wrappers that expose a .network attribute
        inner = getattr(mod, 'network', mod)
        encoder = getattr(inner, 'encoder', None)
        if encoder is not None and hasattr(encoder, 'output_channels'):
            return int(encoder.output_channels[-1])
        # Fallback with a warning
        default = 320
        self.print_to_log_file(
            f"[AuxDiagMixin] Could not determine bottleneck channels from "
            f"{type(mod).__name__}; defaulting to {default}.  "
            f"Override aux_bottleneck_ch on the trainer if incorrect."
        )
        return default

    def _register_bottleneck_hook(self, mod: nn.Module):
        """Register a forward hook on the last encoder stage."""
        inner = getattr(mod, 'network', mod)
        encoder = getattr(inner, 'encoder', None)
        if encoder is None:
            self.print_to_log_file(
                "[AuxDiagMixin] No encoder found; bottleneck hook NOT registered."
            )
            return

        # dynamic-network-architectures ResidualEncoder uses self.stages
        stages = getattr(encoder, 'stages', None)
        if stages is None or len(stages) == 0:
            self.print_to_log_file(
                "[AuxDiagMixin] encoder.stages not found; hook NOT registered."
            )
            return

        def _hook(module, input, output):
            self._bottleneck_features = output

        self._hook_handle = stages[-1].register_forward_hook(_hook)

    def _load_aux_disease_map(self) -> Optional[Dict[str, List[int]]]:
        """Load disease_map.json when DiseaseConditioningMixin is absent."""
        path = join(self.preprocessed_dataset_folder_base, "disease_map.json")
        if not isfile(path):
            self.print_to_log_file(
                f"[AuxDiagMixin] {path} not found; auxiliary loss DISABLED."
            )
            return None
        with open(path) as f:
            dm = json.load(f)
        self.print_to_log_file(
            f"[AuxDiagMixin] Loaded disease_map.json ({len(dm)} entries) from {path}"
        )
        return dm

    def _resolve_disease_vec(self, batch_keys: list) -> Optional[torch.Tensor]:
        """Build disease vector tensor for a batch.

        Prefers ``self.disease_map`` (from DiseaseConditioningMixin) over the
        locally loaded ``self._aux_disease_map``.
        """
        dm = getattr(self, 'disease_map', None) or self._aux_disease_map
        if dm is None:
            return None
        K = getattr(self, 'disease_K', self.aux_num_classes)
        vecs = []
        for case_id in batch_keys:
            case_id = case_id.removesuffix("_image")  # blosc2 identifiers include _image suffix
            if case_id not in dm:
                return None  # skip batch if any key is missing
            vecs.append(dm[case_id][:K])
        return torch.tensor(vecs, dtype=torch.float32, device=self.device)
