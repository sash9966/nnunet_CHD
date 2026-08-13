"""
Per-case sampling-weight mixin.

Bias the TRAINING dataloader to draw trustworthy cases (expert / GT) more often and
noisy pseudo-labels less often — a per-case importance weight, applied at the *sampling*
level (not the loss). Reads a weight map ``case_weights.json`` = ``{case_id: weight}`` from
the dataset folder and sets the train loader's ``sampling_probabilities`` (normalized weights,
aligned to ``loader.indices``). Validation sampling is left uniform.

Mechanism (same shared-path override as ``SeptalOversampleMixin``, flagged intentionally):
during ``get_dataloaders`` we monkeypatch DA5's ``nnUNetDataLoader`` to a subclass that, on
the FIRST instantiation (the train loader — DA5 builds train before val), computes
``sampling_probabilities`` from ``self.indices`` + the weight map. Set at construction, so it
is pickled to the augmentation worker processes. Restored in a ``finally``.

Self-disables (uniform sampling = base behaviour) if no ``case_weights.json`` is found, so the
trainer is safe on datasets without a weight map.

Rationale / literature: Ren et al. 2018 (reweight noisy examples), Karimi et al. 2020 (noisy
labels in medical image analysis), Cui et al. 2019 (effective number of samples — why scarce
cases should not be up-weighted without bound). See docs/project_overview.html → Literature.
"""
from __future__ import annotations

import json
import os

import numpy as np

from nnunetv2.paths import nnUNet_raw
from nnunetv2.training.nnUNetTrainer.variants.mixins._base import TrainerMixin


class CaseSamplingWeightMixin(TrainerMixin):
    """Weight per-case training sampling probability from ``case_weights.json``."""

    case_weights_filename: str = "case_weights.json"

    def mixin_init(self):
        super().mixin_init()
        self._case_weights = None

    def mixin_initialize(self):
        super().mixin_initialize()
        self._case_weights = self._load_case_weights()
        if self._case_weights:
            vals = list(self._case_weights.values())
            self.print_to_log_file(
                f"[CaseWeight] loaded {len(self._case_weights)} case weights "
                f"(min {min(vals):.2f}, max {max(vals):.2f})")
        else:
            self.print_to_log_file("[CaseWeight] no case_weights.json found -> uniform sampling")

    def _load_case_weights(self):
        cands = []
        base = getattr(self, "preprocessed_dataset_folder_base", None)
        if base:
            cands.append(os.path.join(base, self.case_weights_filename))
        try:
            dsn = self.dataset_json.get("name")
        except Exception:
            dsn = None
        if dsn:
            cands.append(os.path.join(nnUNet_raw, dsn, self.case_weights_filename))
        for p in cands:
            if p and os.path.isfile(p):
                try:
                    raw = json.loads(open(p).read())
                    self.print_to_log_file(f"[CaseWeight] using {p}")
                    return {str(k): float(v) for k, v in raw.items()}
                except Exception as e:
                    self.print_to_log_file(f"[CaseWeight] failed to read {p}: {e!r}")
        return None

    def get_dataloaders(self):
        if not self._case_weights:
            return super().get_dataloaders()

        import nnunetv2.training.nnUNetTrainer.variants.data_augmentation.nnUNetTrainerDA5 as da5mod
        Base = da5mod.nnUNetDataLoader
        weights = self._case_weights
        logf = self.print_to_log_file
        state = {"first": True}   # only the FIRST loader built (train) is weighted; val stays uniform

        class _WeightedLoader(Base):
            def __init__(self, data, *a, **kw):
                super().__init__(data, *a, **kw)
                if not state["first"]:
                    return
                state["first"] = False
                keys = list(self.indices)
                w = np.array([weights.get(str(k), 1.0) for k in keys], dtype=np.float64)
                if not np.isfinite(w).all() or w.sum() <= 0:
                    logf("[CaseWeight] invalid weights -> leaving uniform"); return
                self.sampling_probabilities = w / w.sum()
                miss = [k for k in keys if str(k) not in weights]
                logf(f"[CaseWeight] TRAIN sampling weighted over {len(keys)} cases "
                     f"(sum {w.sum():.1f}, max case share {w.max()/w.sum():.3f}"
                     f"{'; ' + str(len(miss)) + ' unmapped->1.0' if miss else ''})")

        da5mod.nnUNetDataLoader = _WeightedLoader
        try:
            return super().get_dataloaders()
        finally:
            da5mod.nnUNetDataLoader = Base
