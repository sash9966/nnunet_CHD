# Approach G — Human-in-the-loop via 3D Slicer Segment Editor

> Arterial-outflow review priorities, QC overlays to ship to Slicer
> (uncertainty, components, AO/PA disagreement), and the active-learning
> capture format are specified in
> [`../STAGE2_SPEC.md`](../STAGE2_SPEC.md) §5. The disease-flag overlays come
> from [`../anatomy_priors.yaml`](../anatomy_priors.yaml).

**Goal.** Make the Stage-1 + Stage-2 outputs trivially loadable into
3D Slicer for clinician review and correction. Targets the actual
end-user workflow (CFD preprocessing); makes the pipeline useful
even when Stage-2 isn't fully reliable.

## What "compatible with Slicer" means here

- Predictions saved as **NIfTI** with correct affine — already true for
  all `nnUNetv2_predict` output.
- A small `.mrml` scene file (Slicer's project format) bundles:
  - The CT volume
  - The binary heart mask (Segmentation node, single segment "Heart")
  - The 7-class decomposition (Segmentation node, 7 segments coloured
    per Dataset030 convention)
- Optionally a `.json` metadata file with case ID, predicted disease
  flags (from `disease_map.json`), and the model identifiers used.

## Manual correction loop

1. Clinician opens the scene in Slicer.
2. Inspects the binary mask first; corrects gross localisation errors
   using **Segment Editor → Paint / Erase / Threshold**.
3. The corrected binary mask is then fed back through a chosen Stage-2
   method (e.g. Approach A re-inference with the corrected channel-1 input).
4. The corrected 7-class output is re-opened in Slicer for fine-tuning.

## What to build

- `export_to_slicer.py` — converts a (CT, binary mask, 7-class pred,
  disease flags) tuple into a `.mrml` + accompanying NIfTIs in a
  single zipped folder per case.
- `import_corrections.py` — given a directory of corrected `.seg.nrrd`
  files from Slicer, convert back into nnU-Net-compatible NIfTIs
  for retraining.
- `correction_dashboard.md` — a Markdown checklist the clinician can
  fill in per case (a 5-question form: chamber correct? AO/PA correct?
  septum correct? vessel continuity correct? notes).

## Why this approach is always on

Whatever automatic method wins, you'll still need to hand-correct edge
cases for clinical CFD preprocessing. Investing in the round-trip
Slicer↔nnU-Net plumbing now pays off forever.

## Integration with MONAI Label (Approach E)

`medsam_slicer/` and this folder are complementary:
- `medsam_slicer/` is *active assistance*: the model proposes
  segmentations on prompts.
- `human_in_the_loop/` is *passive review*: the model is silent unless
  invoked.

Both share the same `.mrml` export format, so a clinician can switch
between them in the same Slicer session.
