# Shared agent instructions

Read `CLAUDE.md`, `docs/FEATURES.md`, `docs/CONVENTIONS.md` and `docs/REFINEMENT_MEMORY.md` before changing the clinical refinement workflow. Claude and Codex share those records; update them when findings or decisions change.

For the current accepted-50 comparison, follow `docs/refinement_four_arm.md`. Keep observed historical failures separate from proposed fixes and GPU-validated outcomes. Do not silently change the cohort, seed source, training plans, evaluation route or bifurcation-depth semantics between arms. Preserve reproducible prompts and final-mask provenance.
