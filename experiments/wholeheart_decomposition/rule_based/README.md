# Approach D — Rule-based heuristic decomposition

> Anatomy, soft topology priors, disease-conditioned overrides, and the AO/PA
> feature list live in [`../STAGE2_SPEC.md`](../STAGE2_SPEC.md) +
> [`../anatomy_priors.yaml`](../anatomy_priors.yaml). Encode disease rules
> (TGA/DORV/ToF/PuA/HLHS) by reading the YAML, not by hand-wiring `if
> disease["TGA"]:` branches.

**Goal.** Hand-coded labelling driven by anatomy + topology. Slow to
build *correctly*, but always interpretable and useful both as a baseline
and as a feature for the learned methods (Approach C in particular).

## Heuristics to encode

1. **Chambers first**. Take the binary heart mask, erode by ~5 mm,
   keep the 4-6 largest connected components. The largest by volume that
   sits inferior + left = LV-BP; superior + left + smaller = LA; symmetric
   right side = RV-BP / RA. Disambiguate via geodesic distance from the
   inferior vena cava entry (which is also detectable).
2. **Myocardium**. The shell between the chamber blobs and the heart-mask
   boundary; assign by morphological dilation of the chamber assignment.
3. **Aorta**. The largest *vessel-shaped* connected component that exits
   superiorly from the LV side. Vessel-shape = high eccentricity +
   bounded radius along the EDT skeleton.
4. **PA**. Same but exiting from the RV side. If the AO/PA share a base
   (DORV, TGA, PuA), use diagnosis metadata to flip the assignment.

## Geodesic propagation

For every voxel inside the binary mask, compute geodesic distance to the
labelled chamber blobs. Each voxel's chamber label is the argmin, *but
only* if the geodesic path doesn't cross a high-curvature constriction
(prevents AO label from spilling into PA via shared connective tissue).

## Files to add

- `chamber_heuristics.py` — detect chambers from binary mask.
- `vessel_heuristics.py` — detect AO and PA tracks.
- `geodesic_propagate.py` — `scipy`-based geodesic distance + assignment.

## Why ship this

Even at 60% Dice this baseline gives **interpretable error analysis**:
"chamber detection was correct on N cases; vessel disambiguation failed
on M cases". That breakdown is hard to extract from a black-box network.

## How it composes with disease metadata

`scripts/make_disease_map.py` provides per-case K=8 disease flags.
HLHS, TGA, DORV, PuA all directly affect which chamber connects to AO/PA.
Encode these as explicit rule branches:

```python
if disease["TGA"]:
    swap_assignments("AO", "PA")
if disease["HLHS"]:
    LV_volume_threshold *= 0.2  # LV may be vestigial
```
