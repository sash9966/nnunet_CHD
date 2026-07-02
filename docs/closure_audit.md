# Myocardial-closure audit (Dataset052 Phase 0) — ⚠ HALT

Tool: `tools/audit_myocardial_closure.py` (read-only). Sample: **32 held-out test
labels** (`…/SegmentationDetailStandard/Dataset030/labelsTs`). The full 110 `labelsTr`
live on the cluster (`/scratch/users/sastocke/…`) and are **not on this dev box**, so
this is a representative sample, not the full cohort — rerun there for final counts.
No `gudhi`/`cripser` on disk → topology via Euler number + 26-connectivity + fill-holes.

## Results (n=32)
| Metric | Value | Read |
|---|---|---|
| Myo present | 26/32 (6 missing, ~19%) | spec expected ~12% missing; higher here |
| **Myo connected components** | **median 5.5, max 29; only 5/26 = 1 component** | Myo is **fragmented**, not one shell |
| **Myo encloses volume** (fill-holes − Myo) | **median 3 vox; 11/26 enclose 0** | Myo is **not a watertight wall** around chambers |
| VSD orifice ringed by Myo (`ring_LV_RV`) | median 0.51 | orifice only ~half-bounded by wall |
| Outflow orifices ringed by Myo | LV–AO 0.28, RV–AO 0.34, RV–PA 0.29 | outlets weakly bounded (~30%) |
| AO has ventricular orifice | 32/32 | aortic outlet derivable |
| PA has ventricular orifice | 27/32 (5 missing) | ~16% exclusion candidates (atresia/discontinuous) |
| LV–RV adjacency (VSD) | median 234 vox, nonzero 24/32 | localized, plausible |
| LA–RA / LV–LA / RV–RA adjacency | median 3224 / 3880 / 4774 | huge — **no atrial walls / AV-valve planes labeled** |
| AO–PA adjacency | median 544, nonzero 28/32 | arterial contact common (future primitive) |

## Verdict: the strict "closed myocardial shell" premise does NOT hold on ImageCHD
The method as specified requires `Myo ∪ separators` to be a **closed 2-manifold that
bounds every compartment**. On the GT:
1. Myo is **missing** in ~19% and **fragmented** (median 5.5 components) in the rest.
2. Myo **encloses ~zero volume** (11/26 enclose nothing) — it is a thin, open, LV-centric
   sheet, not a wall enclosing all four chambers + outflows.
3. Orifices are only **~30–50% ringed** by Myo.
4. The **atria have no myocardial wall at all** (LA–RA/LV–LA/RV–RA adjacencies are huge),
   consistent with the earlier septal-defect finding: the atrial septum is unlabeled.

⇒ Per the Phase-0 STOP rule ("if Myo is not closable / not present in a large fraction,
halt"), **we halt on the literal formulation.** `Myo ∪ septal_defect ∪ outflow_cap`
cannot be certified a closed shell, and `validate_closure.myo_closed` (2-manifold) is
not an achievable invariant on this data.

## What IS achievable and still valuable (recommended re-scope)
The *useful* core survives: making the **separators explicit positive labels**
(`septal_defect`, `outflow_cap`) still directly attacks LV/RV confusion and Ao↔PA
swapping — the actual failure modes — **without** needing a globally closed manifold.
Recommend replacing the "closed 2-manifold" invariant with an achievable
**compartment-separation contract**:
- each blood pool is a **single 26-connected component** (flag-gated for single-ventricle), AND
- **no two should-be-separated pools are 26-adjacent** unless a separator label lies between them
  (VSD → septal_defect; outflow → outflow_cap), AND
- **completeness**: after inserting a separator, the two pools are verified 26-disconnected.

This drops `myo_closed`/Betti-manifold checks (unprovable here) but keeps
`pools_single_component`, `all_orifices_capped`, and completeness — which is what makes
each compartment a bounded container for CFD.

## Exclusions (seed for the future artery/vein step)
- ~5/32 (~16%) have a present PA with **no ventricular orifice** → `out_of_scope/arteries_todo/`.
- TGA-IVS (TGA ∧ no LV–RV gap) → `out_of_scope/tga_ivs/` (needs diagnosis join; Phase 3).
- 6/32 missing Myo → the ventricular-septum separator can't be myo-anchored; likely also excluded or handled by direct LV–RV adjacency only.

**Recommendation:** proceed, but with the re-scoped compartment-separation contract
(not manifold closure), and rerun this audit on the full 110 `labelsTr` on the cluster
to finalize exclusion counts before generating Dataset052.
