"""
FiLM-conditioned ResidualEncoderUNet for nnU-Net v2.

Wraps a standard ResidualEncoderUNet and applies Feature-wise Linear
Modulation (FiLM) at the bottleneck and every decoder stage:

    y = (1 + gamma) * x + beta

where gamma and beta are predicted from a learned disease embedding.
When no disease vector is provided the forward pass is identical to the
vanilla network (exact baseline, zero extra compute).
"""
from __future__ import annotations

from typing import List, Optional, Tuple, Type, Union

import torch
import torch.nn as nn
from dynamic_network_architectures.initialization.weight_init import InitWeights_He


class FiLMLayer(nn.Module):
    """Predict per-channel gamma and beta from a disease embedding and apply FiLM.

    FiLM formula::

        y = (1 + gamma) * x + beta

    Both ``gamma_head`` and ``beta_head`` are zero-initialised so that the
    default output is the identity:  ``y = x``  when the disease embedding is
    all-zeros (or just after init).
    """

    def __init__(self, feature_channels: int, embed_dim: int):
        super().__init__()
        self.gamma_head = nn.Linear(embed_dim, feature_channels)
        self.beta_head = nn.Linear(embed_dim, feature_channels)

        # Zero-init so FiLM starts as identity (gamma=0, beta=0 → y=x)
        nn.init.zeros_(self.gamma_head.weight)
        nn.init.zeros_(self.gamma_head.bias)
        nn.init.zeros_(self.beta_head.weight)
        nn.init.zeros_(self.beta_head.bias)

    def forward(self, x: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, C, *spatial) feature tensor.
        e : (B, E) disease embedding vector.
        """
        gamma = self.gamma_head(e)  # (B, C)
        beta = self.beta_head(e)  # (B, C)
        # broadcast to spatial dims
        shape = [x.shape[0], x.shape[1]] + [1] * (x.ndim - 2)
        gamma = gamma.view(*shape)
        beta = beta.view(*shape)
        return (1 + gamma) * x + beta


class FiLMConditionedResEncUNet(nn.Module):
    """ResidualEncoderUNet with optional FiLM disease-vector conditioning.

    The encoder and decoder are *absorbed* from a pre-built base network so
    that the ``state_dict`` keys (``encoder.*``, ``decoder.*``) stay identical
    to vanilla ResidualEncoderUNet.  New parameters live under
    ``disease_mlp.*``, ``bottleneck_film.*``, and ``decoder_films.*``.

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
    """

    def __init__(
        self,
        base_network: nn.Module,
        disease_K: int = 8,
        disease_H: int = 64,
        disease_E: int = 32,
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

        # ---- bottleneck FiLM ----
        bottleneck_channels = self.encoder.output_channels[-1]
        self.bottleneck_film = FiLMLayer(bottleneck_channels, disease_E)

        # ---- one FiLM layer per decoder stage ----
        #  decoder stage s operates on features with channel count
        #  encoder.output_channels[-(s+2)]  (the skip channel count at that level)
        decoder_stage_channels: List[int] = []
        n_decoder_stages = len(self.decoder.stages)
        for s in range(n_decoder_stages):
            decoder_stage_channels.append(self.encoder.output_channels[-(s + 2)])

        self.decoder_films = nn.ModuleList([
            FiLMLayer(ch, disease_E) for ch in decoder_stage_channels
        ])

        # ---- inference attribute (set via set_disease_vec) ----
        self._current_disease_vec: Optional[torch.Tensor] = None

    # ------------------------------------------------------------------
    # Inference helpers
    # ------------------------------------------------------------------
    def set_disease_vec(self, vec: Optional[torch.Tensor]) -> None:
        """Store a disease vector for use in ``forward`` when the arg is None."""
        self._current_disease_vec = vec

    def clear_disease_vec(self) -> None:
        self._current_disease_vec = None

    # ------------------------------------------------------------------
    # Deep-supervision property (keeps set_deep_supervision_enabled working)
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
        InitWeights_He(1e-2)(module)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(
        self,
        x: torch.Tensor,
        disease_vec: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        # resolve effective disease_vec
        dv = disease_vec if disease_vec is not None else self._current_disease_vec

        if dv is None:
            # ---- exact baseline path (no extra compute) ----
            skips = self.encoder(x)
            return self.decoder(skips)

        # ---- FiLM-conditioned path ----
        e = self.disease_mlp(dv.float())  # (B, E)

        skips = self.encoder(x)
        skips = list(skips)  # make mutable

        # FiLM at bottleneck
        skips[-1] = self.bottleneck_film(skips[-1], e)

        # decoder forward with per-stage FiLM
        lres_input = skips[-1]
        seg_outputs: List[torch.Tensor] = []
        for s in range(len(self.decoder.stages)):
            x_dec = self.decoder.transpconvs[s](lres_input)
            x_dec = torch.cat((x_dec, skips[-(s + 2)]), dim=1)
            x_dec = self.decoder.stages[s](x_dec)
            # FiLM modulation after decoder conv block
            x_dec = self.decoder_films[s](x_dec, e)
            if self.decoder.deep_supervision:
                seg_outputs.append(self.decoder.seg_layers[s](x_dec))
            elif s == (len(self.decoder.stages) - 1):
                seg_outputs.append(self.decoder.seg_layers[-1](x_dec))
            lres_input = x_dec

        seg_outputs = seg_outputs[::-1]
        if not self.decoder.deep_supervision:
            return seg_outputs[0]
        return seg_outputs
