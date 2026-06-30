"""
chd_landmarks
=============

Disease-landmark and derived-region preprocessing for the nnU-Net CHD
segmentation project.

This package is a *research preprocessing and training-support* tool. It is
NOT a diagnostic tool: it assumes CHD diagnosis flags are already provided
(from imageCHD metadata) and uses them, together with existing ground-truth
anatomy segmentations, to derive disease-specific landmark / region labels.

It is fully isolated from the upstream nnU-Net code and the existing datasets:
  * it reads source datasets read-only,
  * it only ever writes NEW datasets (e.g. Dataset031_*),
  * it never mutates existing dataset.json / labels / trainers.

Modules
-------
io                       NIfTI + config I/O (affine-preserving).
labels                   Label-map resolution by name (never hardcoded ids).
metadata                 Disease-flag parsing + per-case annotation status.
disease_rules            Loads chd_disease_rules.yaml; decides active rules.
geometry                 Voxel/physical geometry, distances, radii, bboxes.
topology                 Connected components, skeletons, Betti (optional libs).
derived_regions          The per-region builders (VSD proxy, stenosis, ...).
derived_label_builder    Orchestrates per-case derivation (conservative merge).
nnunet_dataset_builder   Builds a NEW nnU-Net raw dataset (Dataset031).
region_based_dataset_json Region-based dataset.json generation.
metrics                  Disease-aware metrics beyond Dice.
cli                      `python -m chd_landmarks.cli ...`
"""

__all__ = [
    "io",
    "labels",
    "metadata",
    "disease_rules",
    "geometry",
    "topology",
    "derived_regions",
    "derived_label_builder",
    "nnunet_dataset_builder",
    "region_based_dataset_json",
    "metrics",
]

__version__ = "0.1.0"
