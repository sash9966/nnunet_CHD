# Approach C — Graph / topology-based decomposition

**Goal.** Decompose the binary heart mask into a graph (chambers + vessel
branches) and assign anatomical labels at the *branch* level rather than
the voxel level.

## Why a graph

A voxel-wise classifier can flip-flop within a single anatomical branch
because it has no incentive to be globally consistent. A branch-level
classifier sees the whole structure at once: "this branch connects the
RV to a thin tube exiting the thorax → it's the PA". The graph also
exposes topology directly so we can add hard or soft constraints like
"exactly one Aorta", "Aorta and PA do not share a parent".

## Construction pipeline (sketch)

```
1. Extract centreline skeleton  →  skimage.morphology.skeletonize_3d
2. Detect chambers              →  largest 4-5 connected components of
                                   (binary_mask - skeleton_dilated)
3. Build graph:
     nodes  = {chamber blobs, skeleton branch points, skeleton endpoints}
     edges  = skeleton branches
     features per node:
       - centroid coordinates (mm)
       - voxel volume
       - mean CT intensity inside the node region
       - distance to each chamber
       - vessel radius (EDT max along branch)
4. Optional: connect adjacent chambers in the graph (atrial septum, etc.)
5. Save as PyTorch Geometric Data object  +  NetworkX .gpickle for inspection
```

## Classifier candidates

- **GCN / GraphSAGE** — small, fast, easy first model.
- **Graph Attention (GAT)** — useful if "this branch labels depend on
  neighbouring branches' labels" matters (it does for AO vs PA).
- **Graph transformer** — overkill for ~10-20 nodes per case.

Output is a per-node label probability that we paint back onto the voxels
by majority-vote over the skeleton voxels associated with each node.

## Files to add

- `build_graph.py` — extracts the graph from a binary mask, exports
  `.gpickle` (NetworkX) + `.pt` (PyG).
- `train_branch_classifier.py` — GNN training loop, takes graphs +
  Dataset030 GT labels propagated to nodes.
- `assign_labels.py` — runs trained classifier, paints labels onto voxels.

## When this approach wins

When the dominant error mode after Approach A is *semantic confusion of a
continuous branch* (every voxel in the branch gets the wrong label
together). Voxel-wise networks can't fix this; branch-wise can.

## Risks

- 73 training cases produces ~5-15 branches per case → ~1000 training
  nodes total. Modest dataset. Strong inductive bias (small GNN, heavy
  regularisation, diagnosis-conditioning) is mandatory.
- Skeleton fragility: skeletonization of thin vessels often breaks.
  Pre-filter the skeleton with `prune_short_branches`.
