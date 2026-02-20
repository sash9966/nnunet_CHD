"""
FiLM-conditioned ResidualEncoderUNet for nnU-Net v2.

Wraps a standard ResidualEncoderUNet and applies Feature-wise Linear
Modulation (FiLM) at the **bottleneck only**:

    y = (1 + gamma) * x + beta

where gamma and beta are predicted from a learned disease embedding.

Design rationale
----------------
Applying FiLM at every decoder stage compounds the scaling:
  (1+γ)^N with N≈7 layers means even γ=0.1 gives 1.95× total distortion.
Bottleneck-only injection is sufficient for disease conditioning — the
decoder naturally propagates the modulated representation through all skip
connections without compounding instability.

When no disease vector is provided the forward pass is identical to the
vanilla network (exact baseline, zero extra compute).
"""
from __future__ import annotations

from typing import List, Optional, Union

import torch
import torch.nn as nn
from dynamic_network_architectures.initialization.weight_init import InitWeights_He


class FiLMLayer(nn.Module):
    """Predict per-channel gamma and beta from a disease embedding and apply FiLM.

    FiLM formula::

        y = (1 + gamma) * x + beta

    Weights are initialised with small random values (near-identity) so that
    gradient flow to the disease MLP is immediate.  Biases are zero-init.
    """

    def __init__(self, feature_channels: int, embed_dim: int):
        super().__init__()
        self.gamma_head = nn.Linear(embed_dim, feature_channels)
        self.beta_head = nn.Linear(embed_dim, feature_channels)

        # Small random init: near-identity (gamma≈0, beta≈0) but W≠0 so
        # gradients flow through to disease_mlp from the very first step.
        # Zero-init blocks gradient flow: dL/de = W^T @ dL/dgamma = 0 when W=0.
        nn.init.normal_(self.gamma_head.weight, mean=0, std=0.01)
        nn.init.zeros_(self.gamma_head.bias)
        nn.init.normal_(self.beta_head.weight, mean=0, std=0.01)
        nn.init.zeros_(self.beta_head.bias)

    def forward(self, x: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, C, *spatial) feature tensor.
        e : (B, E) disease embedding vector.
        """
        gamma = self.gamma_head(e)  # (B, C)
        beta = self.beta_head(e)    # (B, C)

        # store magnitudes for diagnostics (tensor, no .item() to keep torch.compile happy)
        self._last_gamma_abs_mean = gamma.detach().abs().mean()
        self._last_beta_abs_mean = beta.detach().abs().mean()

        # broadcast to spatial dims
        shape = [x.shape[0], x.shape[1]] + [1] * (x.ndim - 2)
        gamma = gamma.view(*shape)
        beta = beta.view(*shape)
        return (1 + gamma) * x + beta


class FiLMConditionedResEncUNet(nn.Module):
    """ResidualEncoderUNet with optional FiLM disease-vector conditioning.

    FiLM is applied at the **bottleneck only**.  Applying it at every decoder
    stage compounds the multiplicative scaling by N layers — with N≈7 and
    γ=0.1 the total distortion is (1+0.1)^7 ≈ 1.95×, which destabilises
    training.  Bottleneck injection is sufficient: the modulated
    representation naturally propagates through all decoder skip connections.

    The encoder and decoder are *absorbed* from a pre-built base network so
    that the ``state_dict`` keys (``encoder.*``, ``decoder.*``) stay identical
    to vanilla ResidualEncoderUNet.  New parameters live under
    ``disease_mlp.*`` and ``bottleneck_film.*``.

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

        # ---- bottleneck FiLM (single injection point) ----
        bottleneck_channels = self.encoder.output_channels[-1]
        self.bottleneck_film = FiLMLayer(bottleneck_channels, disease_E)

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
    # FiLM magnitude diagnostics
    # ------------------------------------------------------------------
    def get_film_magnitudes(self) -> dict:
        """Return last-seen |gamma| and |beta| mean for the bottleneck FiLM layer.

        Returns a dict like::

            {"bottleneck": {"gamma": 0.012, "beta": 0.003}}

        Values are floats (already detached).  Returns empty dict if no
        forward pass has been made yet.
        """
        if hasattr(self.bottleneck_film, '_last_gamma_abs_mean'):
            return {
                "bottleneck": {
                    "gamma": self.bottleneck_film._last_gamma_abs_mean.item(),
                    "beta": self.bottleneck_film._last_beta_abs_mean.item(),
                }
            }
        return {}

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

        skips = list(self.encoder(x))     # make mutable

        # FiLM at bottleneck only — modulate the deepest representation
        # before passing to the decoder.  The decoder propagates this through
        # all skip connections naturally.
        skips[-1] = self.bottleneck_film(skips[-1], e)

        return self.decoder(skips)
