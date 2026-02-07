"""
Disease-conditioned ResidualEncoderUNet for nnU-Net v2.

Wraps a standard ResidualEncoderUNet and injects a learned disease embedding
at the bottleneck and at every decoder stage.  When no disease vector is
provided the forward pass is identical to the vanilla network (exact baseline).
"""
from __future__ import annotations

from typing import List, Optional, Tuple, Type, Union

import torch
import torch.nn as nn
from dynamic_network_architectures.initialization.weight_init import InitWeights_He


class DiseaseInjector(nn.Module):
    """Fuses a disease embedding *e* into a spatial feature map *f*.

    Steps
    -----
    1. Broadcast *e* to the spatial dimensions of *f*.
    2. Concatenate along the channel axis  -> (B, C+E, *spatial).
    3. 1×1(×1) conv  -> (B, C, *spatial), followed by norm + nonlin.
    """

    def __init__(
        self,
        feature_channels: int,
        embed_dim: int,
        conv_op: Type[nn.Module],
        norm_op: Type[nn.Module],
        norm_op_kwargs: dict,
        nonlin: Type[nn.Module],
        nonlin_kwargs: dict,
    ):
        super().__init__()
        self.conv = conv_op(feature_channels + embed_dim, feature_channels, kernel_size=1, bias=True)
        self.norm = norm_op(feature_channels, **norm_op_kwargs)
        self.nonlin = nonlin(**nonlin_kwargs)

    def forward(self, f: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        f : (B, C, *spatial) feature tensor.
        e : (B, E) disease embedding vector.
        """
        spatial_dims = f.shape[2:]
        e_broadcast = e.view(e.shape[0], e.shape[1], *([1] * len(spatial_dims)))
        e_broadcast = e_broadcast.expand(-1, -1, *spatial_dims)
        z = torch.cat([f, e_broadcast], dim=1)
        return self.nonlin(self.norm(self.conv(z)))


class DiseaseConditionedResEncUNet(nn.Module):
    """ResidualEncoderUNet with optional disease-vector conditioning.

    The encoder and decoder are *absorbed* from a pre-built base network so
    that the ``state_dict`` keys (``encoder.*``, ``decoder.*``) stay identical
    to vanilla ResidualEncoderUNet.  New parameters live under
    ``disease_mlp.*``, ``bottleneck_injector.*``, and ``decoder_injectors.*``.

    Parameters
    ----------
    base_network : nn.Module
        A fully constructed ``ResidualEncoderUNet`` instance.
    conv_op, norm_op, norm_op_kwargs, nonlin, nonlin_kwargs
        Must match the ops used inside *base_network* (taken from plans).
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
        conv_op: Type[nn.Module],
        norm_op: Type[nn.Module],
        norm_op_kwargs: dict,
        nonlin: Type[nn.Module],
        nonlin_kwargs: dict,
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

        # ---- bottleneck injector ----
        bottleneck_channels = self.encoder.output_channels[-1]
        self.bottleneck_injector = DiseaseInjector(
            bottleneck_channels, disease_E, conv_op, norm_op, norm_op_kwargs, nonlin, nonlin_kwargs,
        )

        # ---- one injector per decoder stage ----
        #  decoder stage s operates on features with channel count
        #  encoder.output_channels[-(s+2)]  (the skip channel count at that level)
        decoder_stage_channels: List[int] = []
        n_decoder_stages = len(self.decoder.stages)
        for s in range(n_decoder_stages):
            decoder_stage_channels.append(self.encoder.output_channels[-(s + 2)])

        self.decoder_injectors = nn.ModuleList([
            DiseaseInjector(
                ch, disease_E, conv_op, norm_op, norm_op_kwargs, nonlin, nonlin_kwargs,
            )
            for ch in decoder_stage_channels
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
        # init_last_bn_before_add_to_0 is handled by the base encoder;
        # the disease modules use standard He init which is fine.

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

        # ---- disease-conditioned path ----
        e = self.disease_mlp(dv.float())  # (B, E)

        skips = self.encoder(x)
        skips = list(skips)  # make mutable

        # inject at bottleneck
        skips[-1] = self.bottleneck_injector(skips[-1], e)

        # decoder forward with per-stage injection
        lres_input = skips[-1]
        seg_outputs: List[torch.Tensor] = []
        for s in range(len(self.decoder.stages)):
            x_dec = self.decoder.transpconvs[s](lres_input)
            x_dec = torch.cat((x_dec, skips[-(s + 2)]), dim=1)
            x_dec = self.decoder.stages[s](x_dec)
            # disease injection after decoder conv block
            x_dec = self.decoder_injectors[s](x_dec, e)
            if self.decoder.deep_supervision:
                seg_outputs.append(self.decoder.seg_layers[s](x_dec))
            elif s == (len(self.decoder.stages) - 1):
                seg_outputs.append(self.decoder.seg_layers[-1](x_dec))
            lres_input = x_dec

        seg_outputs = seg_outputs[::-1]
        if not self.decoder.deep_supervision:
            return seg_outputs[0]
        return seg_outputs
