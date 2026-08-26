# Promptable Refinement Flywheel (workstream D)

> **Extends the clinical pseudo-label flywheel (workstream A).** Take the clean, native-geometry
> **LCC-postprocessed seed labels** (`predictions/*__grid2native_lcc/`) and use them to *prompt*
> promptable / interactive foundation models — to refine the whole heart and, especially, the
> **vessels (PA, aorta, and PA/RA that voxel nnU-Net gets wrong)** — then feed the improved labels
> back into training. Conceptually DeepEdit's seed+edit idea, but with newer 3D foundation models and
> CHD-specific vessel/centerline logic. Name alternatives considered: "Multi-Stage Segmentation
> Adaptation", "Seed-to-Prompt Refinement", "Extendable Pipeline".

## Why
- The LCC seed labels are already good on the big chambers but weak on **thin/branching vessels**
  (PA/RA), which is also where volumetric Dice is misleading. Promptable models (esp. SAM2-style with
  slice propagation, and centerline/vessel-specific models) are strong exactly there.
- We already have per-case seed masks → we can **auto-generate prompts** from them (box/points/centerline)
  with no manual clicking, run refinement at scale, expert-verify, and recycle into the flywheel.
- Native geometry: the LCC labels are on each case's native grid/spacing (not the 512³ ImageCHD grid),
  which is what these models expect — no resampling needed to prompt them.

## Prompt derivation — turning an LCC label into each model's prompt
Per case, per target structure (component of the LCC mask). Implement in a `tools/label_to_prompts.py`:
- **3D / per-slice bounding box** — min/max voxel extent of the component (or per-axial-slice bbox).
  → SegVol, nnInteractive, SAM-family box prompts. Robust, trivial to compute.
- **Foreground point(s)** — distance-transform peak (most-interior voxel) + N sampled from the eroded
  core (avoid boundary). → SAM-Med3D, MedSAM2, Medical-SAM2, SAM. One robust click per structure.
- **Background/negative points** — sample from *adjacent* structures (e.g. for LV, negatives in RV/LA)
  to disambiguate touching chambers.
- **Centerline points (vessels)** — skeletonize the aorta/PA component (`skimage.morphology.skeletonize_3d`
  or `kimimaro`) → ordered centerline; endpoints = start/stop planes (aligns with the TCV/ImageCHD root
  cutoffs). → SeqSeg tracing seeds, centerline-guided vessel models, and slice-propagation seeds.
- **Scribbles** — sample along the centerline or across the component. → ScribblePrompt / nnInteractive.
- **Key-slice mask (video models)** — MedSAM2 / Medical-SAM2 treat the volume as video: give one key
  slice's mask/box/points (straight from the LCC label) and propagate through slices via memory attention.

## Models to evaluate (prompt type → how we drive it from the LCC label)
| Model | Dim | Prompt | Derive from LCC | Notes |
|---|---|---|---|---|
| **MedSAM2** (2504.03600) | 3D/video | key-slice box+points, propagate | key-slice bbox/centroid | fine-tuned on 455k 3D masks; SAM2 memory attn; human-in-loop −85% effort |
| **Medical SAM 2** (2408.00874) | 3D-as-video | key-slice prompt | same | alt SAM2-medical impl |
| **SAM-Med3D** (2310.15161) | native 3D | few 3D point prompts | fg core points + negatives | fully-3D encoder; good 3D spatial context |
| **SegVol** | 3D | bounding box (+text) | 3D bbox | zoom-out-zoom-in; box-only, no click correction |
| **nnInteractive** | 3D (2D-slice prompts) | box/points/scribble | per-slice bbox / scribbles | nnU-Net-based promptable + auto-zoom |
| **DeepEdit** (2305.10655) | 3D | click editing on an initial seg | seed = LCC mask, then clicks | your reference; MONAI Label / 3D Slicer loop |
| **SeqSeg** (2501.15712, Marsden lab) | 3D vessels | centerline seed point + tracing | vessel skeleton endpoints | local-segment vascular model construction; ideal for PA/aorta |
| **HiPaS** (Nat Commun 2025) | 3D | — (auto) | (compare) | pulmonary artery-vein, non-contrast+CTPA, Dice 91.8% — vessel benchmark |
| one-stage vessel+centerline (MDPI Imaging 2025) | 3D | — | (compare) | joint seg+centerline w/ topology loss |

## Metrics
- Segmentation: Dice + **NSD/surface Dice** (better for thin vessels) per structure; whole-heart Dice.
- Vessels specifically: centerline overlap / topology (connected components, branch completeness), not just voxel Dice.
- Loop value: fraction of cases where promptable refinement beats the raw LCC seed (per structure),
  and expert-correction time saved.

## Feedback into the flywheel
Refined + expert-verified outputs (especially vessels) → corrected labels → back into the
workstream-A training pool (Dataset09x/10x). This is effectively "flywheel stage 2": nnU-Net for the
bulk anatomy, promptable/vessel models to fix the hard structures, expert sign-off, recycle.

## Papers (verified links)
- DeepEdit — interactive editable 3D seg: https://arxiv.org/abs/2305.10655
- MedSAM2 — 3D images & videos: https://arxiv.org/abs/2504.03600
- Medical SAM 2 — segment medical images as video: https://arxiv.org/abs/2408.00874
- SAM-Med3D — general-purpose 3D promptable: https://arxiv.org/abs/2310.15161
- SeqSeg — local segments for vascular models (Marsden lab): https://arxiv.org/abs/2501.15712
- HiPaS pulmonary artery-vein: https://www.nature.com/articles/s41467-025-56505-6
- Robust vessel seg + centerline (one-stage): https://www.mdpi.com/2313-433x/11/7/209
- Dynamic prompt generation for interactive 3D training: https://arxiv.org/abs/2510.03189
- RAPS-3D efficient interactive 3D seg: https://arxiv.org/abs/2507.07730
- (verify refs) SegVol, nnInteractive, ScribblePrompt — cite exact arXiv when picked.

## Open questions
- **"zigzag"** — no well-known segmentation model by that name found; likely a mis-transcription.
  Candidates the user may mean: SegVol (zoom), a scribble/zig-zag interaction, or another tool — CONFIRM.
- Contrast vs non-contrast CHD CT: which promptable models hold up (HiPaS handles both).
- 2D-slice prompts (nnInteractive) vs true-3D prompts (SAM-Med3D) for branching vessels.
- Which model per structure: chambers (SAM-Med3D/MedSAM2) vs vessels (SeqSeg/centerline) — likely a mix.
