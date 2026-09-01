#!/usr/bin/env python3
"""
run_nninteractive_refine.py
===========================

Refine an LCC pseudo-label with nnInteractive (Isensee 2025, DKFZ) by auto-prompting from the label.
Per structure: pick the key axial slice (largest cross-section) and pass the filled slice region as a
**lasso** (positive), plus **negative points** inside adjacent structures so touching chambers /
septal defects don't merge. Writes a refined multi-label mask.

API used (from the nnInteractive readme — verify against the installed version on first run):
  from nnInteractive.inference.inference_session import nnInteractiveInferenceSession
  from nnInteractive.model_management import ensure_model_available, get_default_model_id
  session.set_image(img[1,x,y,z]); session.set_target_buffer(uint8[x,y,z])
  session.add_lasso_interaction(crop[di,dj,1], include_interaction=True, interaction_bbox=[[i0,i1],[j0,j1],[z,z+1]])
  session.add_point_interaction((x,y,z), include_interaction=False)   # negatives
  result = target_buffer.cpu().numpy(); session.reset_interactions()

Usage:
  python tools/run_nninteractive_refine.py --image CT.nii.gz --label LCC.nii.gz --out refined.nii.gz \
      [--structures LV-BP,RV-BP,LA,RA] [--prompt-mode lasso|points|both] [--save-prompts prompts.json]
"""
import argparse, json, os, sys
from pathlib import Path

import numpy as np
import nibabel as nib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from label_to_prompts import interior_points, centerline, DEFAULT_STRUCTURES, CHAMBERS, VESSELS  # reuse verified helpers


def log(*a):
    print(*a, flush=True)


def key_slice_lasso(mask):
    """Return (crop[di,dj,1] uint8, interaction_bbox) for the largest-area axial slice, or None."""
    areas = mask.sum(axis=(0, 1))
    if areas.max() == 0:
        return None
    z = int(np.argmax(areas))
    sl = mask[:, :, z]
    ij = np.argwhere(sl)
    i0, j0 = ij.min(0); i1, j1 = ij.max(0) + 1
    crop = sl[i0:i1, j0:j1].astype(np.uint8)[..., None]
    bbox = [[int(i0), int(i1)], [int(j0), int(j1)], [int(z), int(z) + 1]]
    return crop, bbox


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image", required=True, help="CT volume (nnInteractive input)")
    ap.add_argument("--label", required=True, help="LCC pseudo-label (source of prompts)")
    ap.add_argument("--out", required=True, help="refined multi-label output .nii.gz")
    ap.add_argument("--structures", default=None, help="comma-separated (default: all present)")
    ap.add_argument("--prompt-mode", default="lasso", choices=["lasso", "points", "both"])
    ap.add_argument("--n-fg", type=int, default=3)
    ap.add_argument("--n-vessel-pts", type=int, default=8, help="positive points sampled along a vessel centerline")
    ap.add_argument("--save-prompts", default=None, help="optional JSON dump of the prompts used")
    args = ap.parse_args()

    import torch
    from nnInteractive.inference.inference_session import nnInteractiveInferenceSession
    from nnInteractive.model_management import ensure_model_available, get_default_model_id

    structs = dict(DEFAULT_STRUCTURES)
    if args.structures:
        want = {s.strip() for s in args.structures.split(",")}
        structs = {k: v for k, v in structs.items() if k in want}

    log("[nnI] loading model (NNINTERACTIVE_MODEL_DIR=%s)" % os.environ.get("NNINTERACTIVE_MODEL_DIR", "~/.nninteractive"))
    model_path = ensure_model_available(get_default_model_id())
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    sess = nnInteractiveInferenceSession(device=dev)
    sess.initialize_from_trained_model_folder(str(model_path))

    ct = nib.load(args.image)
    arr = np.asanyarray(ct.dataobj).astype(np.float32)
    aff = ct.affine
    sess.set_image(arr[None])                                   # (1, x, y, z)

    lab = np.rint(np.asanyarray(nib.load(args.label).dataobj)).astype(np.int16)
    if lab.shape != arr.shape:
        sys.exit("ERROR: image %s and label %s shapes differ" % (arr.shape, lab.shape))
    out = np.zeros(arr.shape, dtype=np.int16)
    present = {int(v) for v in np.unique(lab)} - {0}
    spacing = tuple(float(z) for z in ct.header.get_zooms()[:3])
    prompts_used = {}

    for name, sid in structs.items():
        if sid not in present:
            continue
        mask = lab == sid
        tgt = torch.zeros(arr.shape, dtype=torch.uint8)
        sess.set_target_buffer(tgt)
        sess.reset_interactions()
        rec = {"id": sid, "positive": None, "negatives": []}

        did_pos = False
        want_lasso = args.prompt_mode in ("lasso", "both") and name in CHAMBERS
        if want_lasso:
            kl = key_slice_lasso(mask)
            if kl is not None:
                crop, bbox = kl
                try:
                    sess.add_lasso_interaction(crop, include_interaction=True, interaction_bbox=bbox)
                    rec["positive"] = {"type": "lasso", "bbox": bbox}
                    did_pos = True
                except Exception as e:
                    log("  [%s] lasso failed (%r) -> falling back to points" % (name, e))
        if not did_pos and name in VESSELS:                     # tubular -> positive points along the centerline
            cl, _ = centerline(mask, spacing)
            if len(cl) >= 2:
                k = max(1, len(cl) // max(1, args.n_vessel_pts))
                pts = cl[::k][:args.n_vessel_pts]
                for p in pts:
                    sess.add_point_interaction(tuple(int(v) for v in p), include_interaction=True)
                rec["positive"] = {"type": "centerline_points", "n": len(pts)}
                did_pos = True
        if not did_pos:                                         # points fallback / points mode
            fg = interior_points(mask, spacing, args.n_fg)
            for p in fg:
                sess.add_point_interaction(tuple(int(v) for v in p), include_interaction=True)
            rec["positive"] = {"type": "points", "points": fg}

        adj = (lab != 0) & (lab != sid)                        # negatives from adjacent structures
        if adj.any():
            for p in interior_points(adj, spacing, 3):
                sess.add_point_interaction(tuple(int(v) for v in p), include_interaction=False)
                rec["negatives"].append(list(map(int, p)))

        res = tgt.cpu().numpy()
        n_new = int(((res > 0) & (out == 0)).sum())
        out[(res > 0) & (out == 0)] = sid                      # first-writer wins (avoid overlaps)
        prompts_used[name] = rec
        log("  [%s] refined voxels=%d (prompt=%s)" % (name, n_new, rec["positive"]["type"]))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(out, aff), args.out)
    log("[nnI] wrote %s" % args.out)
    if args.save_prompts:
        Path(args.save_prompts).write_text(json.dumps(prompts_used, indent=1))
        log("[nnI] prompts -> %s" % args.save_prompts)


if __name__ == "__main__":
    main()
