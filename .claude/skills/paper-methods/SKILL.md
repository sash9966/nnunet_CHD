---
name: paper-methods
description: Turn an experimental workstream into a publication-ready Methods section. Use when the user says "write this up as a paper", "methods section", "document the science", or is preparing work for publication. Produces a grounded, step-by-step Methods writeup + a prior-work basis.
---

# Paper Methods Writer

When invoked, produce a Methods section for the named workstream by doing the following:

1. **Load context.** Read the workstream's memory charter (`memory/project_*.md`) and any repo docs it
   points to (`docs/*.md`), plus the git log / provenance manifest (`logs/run_manifest.jsonl`) for the
   exact commits, scripts, and parameters used.

2. **Assemble the section in this order:**
   - *Scientific grounding* — every prior method the approach builds on, with author + venue/arXiv id.
     Cite only papers verified in-context or fetched; never invent a citation or its ID.
   - *Rationale* — the problem and why the approach should work, in 3–5 sentences.
   - *Method, step by step* — each pipeline stage as a numbered step: input → operation → output, naming
     the exact tool/script and key parameters (thresholds, models, coordinate conventions).
   - *Validation* — datasets, metrics, ground-truth source, and how each metric is interpreted.
   - *Findings & caveats* — quantitative results with deltas, plus honest limitations/risks.
   - *Reproducibility* — commit hashes, env, and where provenance is logged.

3. **Rules.**
   - Ground every claim in either a cited paper or a specific artifact (script/commit); flag anything
     unverified as "to confirm".
   - Prefer precise, reproducible phrasing (exact voxel/spacing/threshold values) over vague description.
   - Keep it method-agnostic and reusable — this template applies to any of the user's workstreams.
   - Output as markdown suitable to paste into the workstream memory under a "## Paper — Methods" heading
     and later into a manuscript.
