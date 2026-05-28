"""
Cross-attention disease conditioning mixin for composable trainers.

Provides ``CrossAttentionConditioningMixin`` which conditions the decoder at
every resolution stage via cross-attention between spatial decoder features
(queries) and a set of per-disease token embeddings (keys + values).

Design rationale
----------------
FiLM applies a single global scale-and-shift at the bottleneck.  This is a
spatially-uniform modulation: every voxel in a cardiac structure gets the
same conditioning signal regardless of its local context.  For CHD the
pathology is often localised (e.g. ASD affects only the atrial septum; HLHS
affects only the left-side structures), so global conditioning can be
insufficient.

Cross-attention naturally addresses this "spatial-locality failure mode"
identified in the INSIDE paper: each spatial query position attends
*selectively* to the disease tokens that are relevant to its local context.
The attention entropy (logged every 50 training steps) is the main diagnostic:
low entropy → the model has learnt which diseases matter for that decoder stage;
high entropy → the conditioning is still uniform (FiLM-like).

This mixin is **mutually exclusive** with ``DiseaseConditioningMixin``.
It **composes cleanly** with ``AuxiliaryDiagnosisMixin``: when both are
active, per-disease tokens are derived by projecting the 256-d
``diagnosis_embedding`` through K learned per-disease linear heads
(data-adaptive tokens) instead of looking up a static embedding table.

Network
-------
``CrossAttentionConditionedResEncUNet`` wraps a plain ResidualEncoderUNet.
It absorbs ``encoder`` and ``decoder`` (identical state_dict keys) and adds:

  disease_embed   — nn.Embedding (K, d_model=64) learnable disease tokens
  pad_token       — nn.Parameter (d_model,) used for inactive diseases
  cross_attn_blocks — nn.ModuleDict {str(stage_idx): CrossAttnBlock}
  aux_proj        — nn.ModuleList[K × Linear(256, 64)] (aux-embedding path only)

``CrossAttnBlock`` per decoder stage:
  Q = q_proj(spatial features)  ∈ (B, N, 64)
  K = V = disease tokens         ∈ (B, 8, 64)
  Attention (single head, scale 1/√64):
      attn = softmax(Q @ K^T / √64)   ∈ (B, N, 8)
      out  = attn @ V                  ∈ (B, N, 64)
  out = out_proj(out)  →  (B, N, C_stage)
  out = LayerNorm(x_flat + out)
  return reshape to (B, C_stage, *spatial)

All cross-attention weights are xavier_uniform_ initialised; biases are zero.
"""
from __future__ import annotations

import json
import math
from os.path import isfile, join
from typing import Dict, List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.nnUNetTrainer.variants.mixins._base import TrainerMixin


# =============================================================================
# CrossAttnBlock — single cross-attention block for one decoder stage
# =============================================================================

class CrossAttnBlock(nn.Module):
    """Lightweight single-head cross-attention block for one decoder stage.

    Queries: spatial decoder features (B, C, *spatial) projected to d_model.
    Keys/Values: disease token sequence (B, num_tokens, d_model).
    Output projected back to C, then residual + LayerNorm.
    """

    def __init__(self, feat_channels: int, d_model: int = 64, num_tokens: int = 8):
        super().__init__()
        self.d_model = d_model
        self.scale = d_model ** -0.5

        self.q_proj   = nn.Linear(feat_channels, d_model)
        self.out_proj  = nn.Linear(d_model, feat_channels)
        self.norm      = nn.LayerNorm(feat_channels)

        # Xavier init for all projections, zero biases
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.zeros_(self.q_proj.bias)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

        # Cache for entropy diagnostics (set during forward, detached)
        self._last_attn: Optional[torch.Tensor] = None

    def forward(
        self,
        x: torch.Tensor,
        disease_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, C, *spatial) decoder feature map.
        disease_tokens : (B, T, d_model) disease token sequence.

        Returns
        -------
        (B, C, *spatial) modulated feature map.
        """
        B, C, *spatial = x.shape
        N = math.prod(spatial)

        # Flatten spatial dims: (B, N, C)
        x_flat = x.reshape(B, N, C)

        # Compute queries
        q = self.q_proj(x_flat)                         # (B, N, d_model)

        # Single-head attention (K=8 tokens → attention matrix is tiny)
        k = v = disease_tokens                           # (B, T, d_model)
        attn = torch.softmax(
            q @ k.transpose(-2, -1) * self.scale,       # (B, N, T)
            dim=-1,
        )

        # Store for entropy diagnostics (no extra memory cost — detached)
        self._last_attn = attn.detach()

        out = attn @ v                                   # (B, N, d_model)
        out = self.out_proj(out)                         # (B, N, C)

        # Residual + LayerNorm
        out = self.norm(x_flat + out)                    # (B, N, C)

        return out.reshape(B, C, *spatial)


# =============================================================================
# CrossAttentionConditionedResEncUNet — wrapper network
# =============================================================================

class CrossAttentionConditionedResEncUNet(nn.Module):
    """ResidualEncoderUNet with cross-attention disease conditioning at every
    decoder stage.

    Absorbs ``encoder`` and ``decoder`` from a pre-built base network so that
    state_dict keys stay identical to vanilla ResidualEncoderUNet.  New
    parameters live under ``disease_embed``, ``pad_token``,
    ``cross_attn_blocks``, and optionally ``aux_proj``.

    Parameters
    ----------
    base_network : nn.Module
        A fully-constructed ResidualEncoderUNet.
    disease_K : int
        Number of disease classes (length of multi-hot vector).  Default 8.
    d_model : int
        Attention dimension.  Default 64.
    use_aux_embedding : bool
        If True, per-disease tokens are produced by projecting a 256-d
        ``diagnosis_embedding`` through K learned Linear(256, d_model) heads
        rather than looking up the static embedding table.  Set automatically
        by CrossAttentionConditioningMixin when AuxiliaryDiagnosisMixin is
        present.
    """

    def __init__(
        self,
        base_network: nn.Module,
        disease_K: int = 8,
        d_model: int = 64,
        use_aux_embedding: bool = False,
    ):
        super().__init__()

        # Absorb encoder + decoder (preserves state_dict key names)
        self.encoder = base_network.encoder
        self.decoder = base_network.decoder

        self.disease_K        = disease_K
        self.d_model          = d_model
        self.use_aux_embedding = use_aux_embedding

        # ── Disease token embeddings ──────────────────────────────────────
        # Learnable table (K, d_model) + one [PAD] token for inactive diseases
        self.disease_embed = nn.Embedding(disease_K, d_model)
        self.pad_token     = nn.Parameter(torch.zeros(d_model))

        nn.init.xavier_uniform_(self.disease_embed.weight)

        # ── Cross-attention block for each decoder stage ──────────────────
        # Decoder stage s processes features with encoder.output_channels[-(s+2)] channels
        n_stages = len(self.decoder.stages)
        stage_channels = [
            int(self.encoder.output_channels[-(s + 2)])
            for s in range(n_stages)
        ]
        self.cross_attn_blocks = nn.ModuleDict({
            str(s): CrossAttnBlock(stage_channels[s], d_model, disease_K)
            for s in range(n_stages)
        })

        # ── Aux embedding projection (optional) ──────────────────────────
        # K separate Linear(256, d_model) — one per disease, giving data-adaptive
        # per-disease tokens from the bottleneck-derived 256-d representation.
        if use_aux_embedding:
            aux_embed_dim = 256  # matches AuxiliaryDiagnosisMixin head hidden size
            self.aux_proj = nn.ModuleList([
                nn.Linear(aux_embed_dim, d_model) for _ in range(disease_K)
            ])
            for layer in self.aux_proj:
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

        # ── Inference attribute (set via set_disease_vec) ─────────────────
        self._current_disease_vec: Optional[torch.Tensor] = None

    # ------------------------------------------------------------------
    # Disease token construction
    # ------------------------------------------------------------------
    def _build_disease_tokens(
        self,
        disease_vec: torch.Tensor,
        diagnosis_embedding: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Build (B, K, d_model) token sequence from multi-hot disease vector.

        Active disease i → embed_table[i] (or aux_proj[i](diagnosis_embedding))
        Inactive disease i → pad_token (learned)
        """
        B, K = disease_vec.shape
        device = disease_vec.device
        mask = disease_vec.float().unsqueeze(-1)       # (B, K, 1)

        if self.use_aux_embedding and diagnosis_embedding is not None:
            # Data-adaptive tokens: project 256-d embedding with K distinct heads
            tokens = torch.stack(
                [self.aux_proj[i](diagnosis_embedding) for i in range(K)],
                dim=1,
            )                                           # (B, K, d_model)
        else:
            # Static embedding table lookup
            tokens = self.disease_embed.weight          # (K, d_model)
            tokens = tokens.unsqueeze(0).expand(B, -1, -1)  # (B, K, d_model)

        # Replace inactive disease slots with the learned PAD token
        pad = self.pad_token.unsqueeze(0).unsqueeze(0).expand(B, K, -1)  # (B, K, d_model)
        tokens = mask * tokens + (1.0 - mask) * pad
        return tokens                                   # (B, K, d_model)

    # ------------------------------------------------------------------
    # Inference helpers (same API as FiLMConditionedResEncUNet)
    # ------------------------------------------------------------------
    def set_disease_vec(self, vec: Optional[torch.Tensor]) -> None:
        self._current_disease_vec = vec

    def clear_disease_vec(self) -> None:
        self._current_disease_vec = None

    # ------------------------------------------------------------------
    # Deep-supervision property
    # ------------------------------------------------------------------
    @property
    def deep_supervision(self) -> bool:
        return self.decoder.deep_supervision

    @deep_supervision.setter
    def deep_supervision(self, val: bool) -> None:
        self.decoder.deep_supervision = val

    # ------------------------------------------------------------------
    # Initialization (mirrors ResidualEncoderUNet.initialize)
    # ------------------------------------------------------------------
    @staticmethod
    def initialize(module: nn.Module) -> None:
        from dynamic_network_architectures.initialization.weight_init import InitWeights_He
        InitWeights_He(1e-2)(module)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(
        self,
        x: torch.Tensor,
        disease_vec: Optional[torch.Tensor] = None,
        diagnosis_embedding: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        dv = disease_vec if disease_vec is not None else self._current_disease_vec

        if dv is None:
            # Baseline path — no disease tokens, no extra compute
            skips = self.encoder(x)
            return self.decoder(skips)

        # Build per-disease token sequence: (B, K, d_model)
        disease_tokens = self._build_disease_tokens(dv.float(), diagnosis_embedding)

        # Encoder (AuxDiag bottleneck hook fires here when both mixins active)
        skips = list(self.encoder(x))

        # Manual decoder forward with cross-attention injection after each stage
        lres_input   = skips[-1]
        seg_outputs: List[torch.Tensor] = []

        for s in range(len(self.decoder.stages)):
            x_dec = self.decoder.transpconvs[s](lres_input)          # upsample
            x_dec = torch.cat((x_dec, skips[-(s + 2)]), dim=1)       # skip concat
            x_dec = self.decoder.stages[s](x_dec)                    # conv block
            x_dec = self.cross_attn_blocks[str(s)](x_dec, disease_tokens)  # ← cross-attn
            if self.decoder.deep_supervision:
                seg_outputs.append(self.decoder.seg_layers[s](x_dec))
            elif s == len(self.decoder.stages) - 1:
                seg_outputs.append(self.decoder.seg_layers[-1](x_dec))
            lres_input = x_dec

        seg_outputs = seg_outputs[::-1]
        return seg_outputs if self.decoder.deep_supervision else seg_outputs[0]


# =============================================================================
# Shared build helper (handles static predictor call + instance trainer call)
# =============================================================================

def build_cross_attention_conditioned_network(
    self_or_arch_name,
    architecture_class_name=None,
    arch_init_kwargs=None,
    arch_init_kwargs_req_import=None,
    num_input_channels=None,
    num_output_channels=None,
    enable_deep_supervision: bool = True,
) -> nn.Module:
    """Build base network and wrap with CrossAttentionConditionedResEncUNet.

    Handles the ``isinstance(self, str)`` arg-shifting for both instance
    calls (during training) and static calls (from nnUNetPredictor).
    """
    if isinstance(self_or_arch_name, str):
        # Static call from nnUNetPredictor: positional args shifted by 1
        (architecture_class_name, arch_init_kwargs, arch_init_kwargs_req_import,
         num_input_channels, num_output_channels) = (
            self_or_arch_name, architecture_class_name, arch_init_kwargs,
            arch_init_kwargs_req_import, num_input_channels,
        )
        disease_K         = 8
        d_model           = 64
        use_aux_embedding = False
    else:
        trainer           = self_or_arch_name
        disease_K         = getattr(trainer, 'disease_K', 8)
        d_model           = getattr(trainer, 'ca_d_model', 64)
        use_aux_embedding = getattr(trainer, '_has_aux_diagnosis', False)

    base = nnUNetTrainer.build_network_architecture(
        architecture_class_name,
        arch_init_kwargs,
        arch_init_kwargs_req_import,
        num_input_channels,
        num_output_channels,
        enable_deep_supervision,
    )

    wrapped = CrossAttentionConditionedResEncUNet(
        base,
        disease_K=disease_K,
        d_model=d_model,
        use_aux_embedding=use_aux_embedding,
    )
    return wrapped


# =============================================================================
# CrossAttentionConditioningMixin
# =============================================================================

class CrossAttentionConditioningMixin(TrainerMixin):
    """Mixin that adds cross-attention disease conditioning at every decoder stage.

    The mixin is mutually exclusive with ``DiseaseConditioningMixin`` (global
    FiLM bottleneck conditioning) — they cannot both be active in the same MRO.

    When ``AuxiliaryDiagnosisMixin`` is also in the MRO, per-disease tokens are
    produced by projecting ``self.diagnosis_embedding`` (256-d) through K
    learned per-disease linear heads rather than looking up the static embedding
    table.  Auto-detected in ``mixin_init`` — no manual flag required.

    Class attributes (override on concrete trainer)
    ------------------------------------------------
    ca_d_model : int
        Attention dimension.  Default 64.
    ca_lr_multiplier : float
        LR multiplier for cross-attention and disease embedding parameters.
        Default 2.0.
    disease_K : int
        Number of disease classes.  Default 8.
    """

    ca_d_model:       int   = 64
    ca_lr_multiplier: float = 2.0
    disease_K:        int   = 8

    # ------------------------------------------------------------------
    # Hook: mixin_init
    # ------------------------------------------------------------------
    def mixin_init(self):
        # ── Mutual exclusivity guard ──────────────────────────────────
        from nnunetv2.training.nnUNetTrainer.variants.mixins.disease_conditioning import (
            DiseaseConditioningMixin,
        )
        if DiseaseConditioningMixin in type(self).__mro__:
            raise ValueError(
                "CrossAttentionConditioningMixin and DiseaseConditioningMixin "
                "are mutually exclusive — they both condition the decoder but "
                "via incompatible mechanisms.  Use one or the other."
            )

        # ── Auto-detect AuxiliaryDiagnosisMixin ──────────────────────
        try:
            from nnunetv2.training.nnUNetTrainer.variants.mixins.aux_diagnosis_mixin import (
                AuxiliaryDiagnosisMixin,
            )
            self._has_aux_diagnosis: bool = AuxiliaryDiagnosisMixin in type(self).__mro__
        except ImportError:
            self._has_aux_diagnosis = False

        super().mixin_init()

        self.disease_map: Optional[Dict[str, List[int]]] = None
        # Step counter for attention entropy logging (every 50 steps)
        self._ca_step_count: int = 0
        # Per-epoch per-stage entropy accumulator (stage_idx -> list[float])
        self._ca_entropy_accum: Dict[str, List[float]] = {}

    # ------------------------------------------------------------------
    # Hook: mixin_initialize  (disease map + network wrap)
    # ------------------------------------------------------------------
    def mixin_initialize(self):
        super().mixin_initialize()
        self.disease_map = self._load_disease_map()

        if self._has_aux_diagnosis:
            self.print_to_log_file(
                f"[CrossAttnMixin] AuxiliaryDiagnosisMixin detected — "
                f"using data-adaptive per-disease tokens (aux_proj path)"
            )

        self.print_to_log_file(
            f"[CrossAttnMixin] d_model={self.ca_d_model}  "
            f"disease_K={self.disease_K}  "
            f"use_aux_embedding={self._has_aux_diagnosis}  "
            f"lr_mult={self.ca_lr_multiplier}x"
        )

    # ------------------------------------------------------------------
    # Override build_network_architecture (wraps base net with cross-attn)
    # ------------------------------------------------------------------
    def build_network_architecture(self, *args, **kwargs):
        return build_cross_attention_conditioned_network(self, *args, **kwargs)

    # ------------------------------------------------------------------
    # Hook: mixin_prepare_forward
    # ------------------------------------------------------------------
    def mixin_prepare_forward(self, batch: dict) -> dict:
        kw = super().mixin_prepare_forward(batch)
        disease_vec = self._build_disease_vec(batch["keys"])
        if disease_vec is not None:
            kw["disease_vec"] = disease_vec
        return kw

    # ------------------------------------------------------------------
    # Hook: mixin_extra_loss  (attention entropy logging only, no extra loss)
    # ------------------------------------------------------------------
    def mixin_extra_loss(self, output, target, batch: dict, **forward_kwargs) -> float:
        extra = super().mixin_extra_loss(output, target, batch, **forward_kwargs)

        self._ca_step_count += 1
        # Always accumulate per-epoch (cheap; entropy uses cached detached attn)
        self._accumulate_attention_entropy()
        # Periodic text log retained for live runs
        if self._ca_step_count % 50 == 0:
            self._log_attention_entropy()

        return extra

    # ------------------------------------------------------------------
    # Hook: extra optimizer param groups
    # ------------------------------------------------------------------
    def mixin_param_groups(self) -> List[dict]:
        groups = super().mixin_param_groups()
        mod = self._get_unwrapped_network()

        ca_params = []
        # Cross-attention blocks
        if hasattr(mod, 'cross_attn_blocks'):
            ca_params.extend(mod.cross_attn_blocks.parameters())
        # Disease embedding table + PAD token
        if hasattr(mod, 'disease_embed'):
            ca_params.extend(mod.disease_embed.parameters())
        if hasattr(mod, 'pad_token'):
            ca_params.append(mod.pad_token)
        # Aux projection heads (when AuxDiag is combined)
        if hasattr(mod, 'aux_proj'):
            ca_params.extend(mod.aux_proj.parameters())

        if ca_params:
            groups.append({
                "params": ca_params,
                "lr": self.initial_lr * self.ca_lr_multiplier,
            })
        return groups

    # ------------------------------------------------------------------
    # Hook: fix LR after scheduler
    # ------------------------------------------------------------------
    def mixin_fix_lr_after_scheduler(self):
        super().mixin_fix_lr_after_scheduler()
        # Match group by checking for cross_attn_block params
        mod = self._get_unwrapped_network()
        if not hasattr(mod, 'cross_attn_blocks'):
            return

        ca_ids = {id(p) for p in mod.cross_attn_blocks.parameters()}
        main_lr = self.optimizer.param_groups[0]["lr"]
        for group in self.optimizer.param_groups[1:]:
            # Identify this group by checking if any of its params are cross-attn params
            group_ids = {id(p) for p in group["params"]}
            if group_ids & ca_ids:
                group["lr"] = main_lr * self.ca_lr_multiplier
                break

    # ------------------------------------------------------------------
    # Hook: on_train_start (copy disease_map.json to output folder)
    # ------------------------------------------------------------------
    def mixin_on_train_start(self):
        super().mixin_on_train_start()
        import shutil
        src = join(self.preprocessed_dataset_folder_base, "disease_map.json")
        if isfile(src):
            dst = join(self.output_folder_base, "disease_map.json")
            shutil.copy2(src, dst)
            self.print_to_log_file(f"[CrossAttnMixin] Copied disease_map.json to {dst}")

    # ------------------------------------------------------------------
    # Validation hooks: set/clear disease_vec per case
    # ------------------------------------------------------------------
    def before_validation_case(self, case_id: str):
        super().before_validation_case(case_id)
        mod = self._get_unwrapped_network()
        if self.disease_map is not None and case_id in self.disease_map:
            vec = torch.tensor(
                [self.disease_map[case_id]], dtype=torch.float32, device=self.device
            )
            mod.set_disease_vec(vec)
        else:
            mod.clear_disease_vec()

    def after_validation_case(self, case_id: str):
        super().after_validation_case(case_id)
        self._get_unwrapped_network().clear_disease_vec()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------
    def _load_disease_map(self) -> Optional[Dict[str, List[int]]]:
        path = join(self.preprocessed_dataset_folder_base, "disease_map.json")
        if not isfile(path):
            self.print_to_log_file(
                f"[CrossAttnMixin] WARNING: {path} not found. "
                "Cross-attention conditioning DISABLED (baseline path)."
            )
            return None
        with open(path) as f:
            dm = json.load(f)
        K = self.disease_K
        for case_id, vec in dm.items():
            if len(vec) < K:
                dm[case_id] = vec + [0] * (K - len(vec))
        self.print_to_log_file(
            f"[CrossAttnMixin] Loaded disease_map.json with {len(dm)} entries"
        )
        return dm

    def _build_disease_vec(self, batch_keys: list) -> Optional[torch.Tensor]:
        if self.disease_map is None:
            return None
        vecs = []
        for case_id in batch_keys:
            case_id = case_id.removesuffix("_image")  # blosc2 identifiers include _image suffix
            if case_id not in self.disease_map:
                raise KeyError(
                    f"Case '{case_id}' not found in disease_map.json."
                )
            vecs.append(self.disease_map[case_id])
        return torch.tensor(vecs, dtype=torch.float32, device=self.device)

    def _accumulate_attention_entropy(self):
        """Append per-stage entropy of the most recent forward to the epoch accumulator."""
        mod = self._get_unwrapped_network()
        if not hasattr(mod, 'cross_attn_blocks'):
            return
        for stage_idx, block in mod.cross_attn_blocks.items():
            if block._last_attn is None:
                continue
            with torch.no_grad():
                attn = block._last_attn.float()             # (B, N, K)
                entropy = -(attn * (attn + 1e-8).log()).sum(dim=-1).mean().item()
            self._ca_entropy_accum.setdefault(stage_idx, []).append(entropy)

    def _log_attention_entropy(self):
        """Print the most recent step's per-stage attention entropy."""
        mod = self._get_unwrapped_network()
        if not hasattr(mod, 'cross_attn_blocks'):
            return
        parts = []
        for stage_idx, block in mod.cross_attn_blocks.items():
            if block._last_attn is None:
                continue
            attn = block._last_attn.float()
            entropy = -(attn * (attn + 1e-8).log()).sum(dim=-1).mean().item()
            parts.append(f"s{stage_idx}={entropy:.3f}")
        if parts:
            self.print_to_log_file(
                f"[CrossAttn] step={self._ca_step_count}  "
                f"attn_entropy(nats) [{', '.join(parts)}]  "
                f"(max={math.log(self.disease_K):.2f} = uniform)"
            )

    # ------------------------------------------------------------------
    # Hook: per-epoch entropy aggregation + logger record
    # ------------------------------------------------------------------
    def mixin_on_train_epoch_end(self, train_outputs):
        super().mixin_on_train_epoch_end(train_outputs)
        if not self._ca_entropy_accum:
            # Loud warning: cross-attention was never invoked this epoch
            self.print_to_log_file(
                f"[CrossAttn][WARNING] epoch {self.current_epoch}  "
                f"no attention entropy collected — disease_vec may not be reaching the network."
            )
            self.logger.log(
                'cross_attention',
                {'per_stage_entropy': {}, 'max_entropy': math.log(self.disease_K)},
                self.current_epoch,
            )
            return

        per_stage_mean = {
            s: float(sum(v) / len(v)) for s, v in self._ca_entropy_accum.items()
        }
        max_ent = math.log(self.disease_K)
        # Text log
        parts = [f"s{s}={e:.3f}" for s, e in sorted(per_stage_mean.items())]
        self.print_to_log_file(
            f"[CrossAttn] epoch={self.current_epoch}  "
            f"mean_attn_entropy(nats) [{', '.join(parts)}]  "
            f"(max={max_ent:.2f} = uniform; lower = more selective)"
        )
        self.logger.log(
            'cross_attention',
            {'per_stage_entropy': per_stage_mean, 'max_entropy': max_ent},
            self.current_epoch,
        )
        self._ca_entropy_accum.clear()
