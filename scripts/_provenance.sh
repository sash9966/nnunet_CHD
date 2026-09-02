#!/bin/bash
# Sourced by run scripts. stamp_provenance records HOW a run/model was made:
#   git commit (the "version"), branch, clean/dirty, timestamp, SLURM job, host, conda env, params.
# Writes a central JSONL manifest (logs/run_manifest.jsonl) + a human PROVENANCE.txt next to the output.
#   usage:  source scripts/_provenance.sh
#           stamp_provenance <label> <outdir> [KEY=VAL ...]
stamp_provenance () {
  # never let logging abort a run
  { local label="$1" outdir="$2"; shift 2 || true
    local repo="${REPO:-/scratch/users/sastocke/nnunet_CHD}"
    local commit branch state ts host job env params
    commit=$(git -C "$repo" rev-parse --short HEAD 2>/dev/null || true)
    branch=$(git -C "$repo" rev-parse --abbrev-ref HEAD 2>/dev/null || true)
    if [ -z "$commit" ]; then                 # git binary often absent on compute nodes -> read .git directly
      local head; head=$(cat "$repo/.git/HEAD" 2>/dev/null || true)
      if [ "${head:0:5}" = "ref: " ]; then
        branch="${head#ref: refs/heads/}"; commit=$(cut -c1-12 "$repo/.git/${head#ref: }" 2>/dev/null || true)
      else commit="${head:0:12}"; fi
    fi
    [ -z "$commit" ] && commit=NA; [ -z "$branch" ] && branch=NA
    git -C "$repo" diff --quiet 2>/dev/null && state=clean || state=DIRTY
    ts=$(date -u +%Y-%m-%dT%H:%M:%SZ); host=$(hostname); job=${SLURM_JOB_ID:-none}; env=${CONDA_PREFIX:-none}
    params="$*"
    mkdir -p "$repo/logs" "$outdir" 2>/dev/null || true
    printf '{"ts":"%s","label":"%s","commit":"%s","branch":"%s","state":"%s","job":"%s","host":"%s","env":"%s","params":"%s","outdir":"%s"}\n' \
      "$ts" "$label" "$commit" "$branch" "$state" "$job" "$host" "$env" "$params" "$outdir" \
      >> "$repo/logs/run_manifest.jsonl" 2>/dev/null || true
    { echo "== provenance =="; echo "when:   $ts"; echo "label:  $label"; echo "commit: $commit ($branch, $state)";
      echo "job:    $job on $host"; echo "env:    $env"; echo "params: $params"; } | tee "$outdir/PROVENANCE.txt" 2>/dev/null || true
  } || true
}
