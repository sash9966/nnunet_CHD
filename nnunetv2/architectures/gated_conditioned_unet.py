"""
Spatially-gated disease-conditioned ResidualEncoderUNet for nnU-Net v2.

Unlike FiLM (which applies the same gamma/beta everywhere), this wrapper
predicts **spatially-varying** modulation that depends on BOTH the disease
vector AND the local image features:

    interaction = x * embed_proj(e)      # content x disease
    gamma_spatial = gate_conv(interaction)  # lightweight bottleneck conv
    y = (1 + gamma_spatial) * x            # spatially-varying FiLM

This lets the model learn: "at spatial locations that look like AO-PA
junction AND disease=pulmonary_atresia, modulate differently."

When no disease vector is provided the forward pass is identical to the
vanilla network (exact baseline, zero extra compute).
"""
from __future__ import annotations

from typing import List, Optional, Type, Union

import torch
import torch.nn as nn
from dynamic_network_architectures.initialization.weight_init import InitWeights_He


class SpatialGateLayer(nn.Module):
    """Spatially-varying gating conditioned on disease embedding AND local features.

    Architecture::

        embed_proj: Linear(E -> C)            # project embedding to channel space
        gate_conv:  Conv1x1(C -> C//r) -> ReLU -> Conv1x1(C//r -> C)  # lightweight

    Last conv is zero-initialised -> identity at start.
    """

    def __init__(
        self,
        feature_channels: int,
        embed_dim: int,
        conv_op: Type[nn.Module] = nn.Conv3d,
        reduction: int = 8,
    ):
        super().__init__()
        self.embed_proj = nn.Linear(embed_dim, feature_channels)
        mid = max(feature_channels // reduction, 4)

        # Use 1x1(x1) convolution — works for 2D or 3D
        self.gate_conv = nn.Sequential(
            conv_op(feature_channels, mid, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            conv_op(mid, feature_channels, kernel_size=1, bias=True),
        )

        # Small-random init for embed_proj so gradients flow immediately
        nn.init.normal_(self.embed_proj.weight, mean=0, std=0.01)
        nn.init.zeros_(self.embed_proj.bias)

        # Zero-init the last conv layer -> identity at start
        nn.init.zeros_(self.gate_conv[-1].weight)
        nn.init.zeros_(self.gate_conv[-1].bias)

    def forward(self, x: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, C, *spatial) feature tensor.
        e : (B, E) disease embedding vector.
        """
        e_proj = self.embed_proj(e)  # (B, C)
        # broadcast to spatial dims
        shape = [x.shape[0], x.shape[1]] + [1] * (x.ndim - 2)
        e_proj = e_proj.view(*shape)  # (B, C, 1, 1[, 1])

        interaction = x * e_proj  # (B, C, *spatial) — content x disease
        gamma_spatial = self.gate_conv(interaction)  # (B, C, *spatial)

        # Store magnitude for diagnostics (detached, cheap)
        self._last_gamma_abs_mean = gamma_spatial.detach().abs().mean().item()

        return (1 + gamma_spatial) * x


class GatedConditionedResEncUNet(nn.Module):
    """ResidualEncoderUNet with spatially-gated disease conditioning.

    Structure mirrors ``FiLMConditionedResEncUNet`` so that trainers,
    logging, and inference code work identically. The key difference is
    that gating is spatially-varying rather than channel-global.

    Parameters
    ----------
    base_network : nn.Module
        A fully constructed ``ResidualEncoderUNet`` instance.
    disease_K : int
        Length of the binary disease flag vector (default 8).
    disease_H : int
        Hidden dim of the disease MLP (default 64).
    disease_E : int
        Output embedding dim of the disease MLP (default 32).
    conv_op : type
        Convolution class (nn.Conv2d or nn.Conv3d).
    """

    def __init__(
        self,
        base_network: nn.Module,
        disease_K: int = 8,
        disease_H: int = 64,
        disease_E: int = 32,
        conv_op: Type[nn.Module] = nn.Conv3d,
        # Accept but ignore these — passed by build_disease_conditioned_network
        norm_op=None,
        norm_op_kwargs=None,
        nonlin=None,
        nonlin_kwargs=None,
    ):
        super().__init__()

        # ---- absorb encoder & decoder (preserves state_dict key names) ----
        self.encoder = base_network.encoder
        self.decoder = base_network.decoder

        # ---- disease embedding MLP ----
        self.disease_mlp = nn.Sequential(
            nn.Linear(disease_K, disease_H),
            nn.ReLU(inplace=True),
            nn.Linear(disease_H, disease_E),
        )

        # ---- bottleneck spatial gate ----
        bottleneck_channels = self.encoder.output_channels[-1]
        self.bottleneck_gate = SpatialGateLayer(
            bottleneck_channels, disease_E, conv_op=conv_op,
        )

        # ---- one spatial gate per decoder stage ----
        decoder_stage_channels: List[int] = []
        n_decoder_stages = len(self.decoder.stages)
        for s in range(n_decoder_stages):
            decoder_stage_channels.append(self.encoder.output_channels[-(s + 2)])

        self.decoder_gates = nn.ModuleList([
            SpatialGateLayer(ch, disease_E, conv_op=conv_op)
            for ch in decoder_stage_channels
        ])

        # ---- auxiliary disease classifier (from bottleneck features) ----
        self.disease_classifier = nn.Linear(bottleneck_channels, disease_K)

        # ---- inference attribute (set via set_disease_vec) ----
        self._current_disease_vec: Optional[torch.Tensor] = None

    # ------------------------------------------------------------------
    # Inference helpers
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
    # Initialization
    # ------------------------------------------------------------------
    @staticmethod
    def initialize(module: nn.Module) -> None:
        InitWeights_He(1e-2)(module)

    # ------------------------------------------------------------------
    # FiLM-compatible magnitude diagnostics
    # ------------------------------------------------------------------
    def get_film_magnitudes(self) -> dict:
        """Return gate magnitudes in the same format as FiLMConditionedResEncUNet.

        Keys: ``"bottleneck"``, ``"dec_0"``, ``"dec_1"``, etc.
        Values: ``{"gamma": float, "beta": 0.0}`` (no beta in gated model,
        kept for logger compatibility).
        """
        result = {}
        if hasattr(self.bottleneck_gate, '_last_gamma_abs_mean'):
            result["bottleneck"] = {
                "gamma": self.bottleneck_gate._last_gamma_abs_mean,
                "beta": 0.0,
            }
        for i, gate in enumerate(self.decoder_gates):
            if hasattr(gate, '_last_gamma_abs_mean'):
                result[f"dec_{i}"] = {
                    "gamma": gate._last_gamma_abs_mean,
                    "beta": 0.0,
                }
        return result

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(
        self,
        x: torch.Tensor,
        disease_vec: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, List[torch.Tensor], tuple]:
        dv = disease_vec if disease_vec is not None else self._current_disease_vec

        if dv is None:
            # ---- exact baseline path (no extra compute) ----
            skips = self.encoder(x)
            return self.decoder(skips)

        # ---- spatially-gated conditioned path ----
        e = self.disease_mlp(dv.float())  # (B, E)

        skips = self.encoder(x)
        skips = list(skips)

        # Auxiliary disease classification from bottleneck (training only)
        aux_disease_logits = None
        if self.training:
            pooled = skips[-1].mean(dim=list(range(2, skips[-1].ndim)))
            aux_disease_logits = self.disease_classifier(pooled)

        # Spatial gate at bottleneck
        skips[-1] = self.bottleneck_gate(skips[-1], e)

        # Decoder forward with per-stage spatial gating
        lres_input = skips[-1]
        seg_outputs: List[torch.Tensor] = []
        for s in range(len(self.decoder.stages)):
            x_dec = self.decoder.transpconvs[s](lres_input)
            x_dec = torch.cat((x_dec, skips[-(s + 2)]), dim=1)
            x_dec = self.decoder.stages[s](x_dec)
            # Spatially-gated modulation after decoder conv block
            x_dec = self.decoder_gates[s](x_dec, e)
            if self.decoder.deep_supervision:
                seg_outputs.append(self.decoder.seg_layers[s](x_dec))
            elif s == (len(self.decoder.stages) - 1):
                seg_outputs.append(self.decoder.seg_layers[-1](x_dec))
            lres_input = x_dec

        seg_outputs = seg_outputs[::-1]
        if not self.decoder.deep_supervision:
            seg_out = seg_outputs[0]
        else:
            seg_out = seg_outputs

        if aux_disease_logits is not None:
            return seg_out, aux_disease_logits
        return seg_out
