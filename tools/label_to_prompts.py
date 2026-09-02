#!/usr/bin/env python3
"""
label_to_prompts.py
===================

Turn native-geometry LCC seed labels into prompts for promptable / interactive segmentation models
(workstream D — Promptable Refinement Flywheel; see docs/promptable_refinement.md).

Per case, per structure, emits:
  * bbox_voxel                     3D bounding box (min/max voxel)              -> SegVol / box prompts
  * interior_point_voxel + fg_points_voxel   distance-transform peak + eroded-core samples
                                                                                -> SAM-Med3D / MedSAM2 points
  * neg_points_voxel               points inside ADJACENT structures            -> disambiguation negatives
  * centerline_voxel + endpoints_voxel   (VESSELS: Aorta, Pulmonary) skeleton diameter path
                                                                                -> SeqSeg trace seeds / root cut-planes
  * lasso_slices                   (CHAMBERS) per-axial-slice positive contour points with
                                     GEOMETRY-adaptive density (uniform + proximity-to-adjacent + curvature)
                                                                                -> nnInteractive pos lasso
All points also given in world/physical coords (via the NIfTI affine). Memory-safe I/O (nibabel).

Usage:
  python tools/label_to_prompts.py --labels-dir <dir_of_label.nii.gz> --out-dir <out> \
      [--cases c1,c2] [--structures LV-BP,RV-BP,Aorta,Pulmonary] \
      [--n-fg 3] [--lasso-k-mm 6] [--lasso-band-mm 4] [--lasso-curv-deg 35] [--write-qc]
"""
import argparse, json, os, sys
from pathlib import Path

import numpy as np
import nibabel as nib
from scipy import ndimage as ndi

# default CHD 7-class scheme (override via --label-json if a dataset differs)
DEFAULT_STRUCTURES = {"LV-BP": 1, "RV-BP": 2, "LA": 3, "RA": 4, "Myo": 5, "Aorta": 6, "Pulmonary": 7}
VESSELS = {"Aorta", "Pulmonary"}                 # tubular -> centerline
CHAMBERS = {"LV-BP", "RV-BP", "LA", "RA"}        # blobs   -> interior seed + adaptive lasso
FE = ".nii.gz"


def vox2world(affine, ijk):
    ijk = np.asarray(ijk, dtype=float)
    return (affine[:3, :3] @ ijk + affine[:3, 3]).tolist()


def bbox_of(mask):
    idx = np.argwhere(mask)
    return [idx.min(0).tolist(), idx.max(0).tolist()]


def interior_points(mask, spacing, n):
    """distance-transform peak (most-interior voxel) + n-1 samples from the eroded core."""
    dt = ndi.distance_transform_edt(mask, sampling=spacing)
    peak = np.unravel_index(int(np.argmax(dt)), dt.shape)
    pts = [list(map(int, peak))]
    if n > 1:
        thr = np.percentile(dt[mask], 80)           # eroded core = deep interior
        core = np.argwhere(dt >= max(thr, 1e-6))
        if len(core):
            rng = np.random.default_rng(0)
            sel = core[rng.choice(len(core), size=min(n - 1, len(core)), replace=False)]
            pts += [list(map(int, p)) for p in sel]
    return pts


def _centroid_path(mask, spacing):
    """Robust fallback centerline: per cross-section along the longest bbox axis, take the centroid
    snapped to the nearest in-mask voxel (guaranteed inside) -> ordered path. Never fails; used when
    skeletonize returns <2 voxels (e.g. Lee 3D thinning collapsing an even, symmetric prism)."""
    idx = np.argwhere(mask)
    axis = int(np.argmax(idx.max(0) - idx.min(0)))       # dominant axis
    path = []
    for v in range(int(idx[:, axis].min()), int(idx[:, axis].max()) + 1):
        sl = idx[idx[:, axis] == v]
        if len(sl) == 0:
            continue
        cen = sl.mean(0)
        p = sl[int(np.argmin(((sl - cen) ** 2).sum(1)))]  # nearest in-mask voxel to the centroid
        path.append([int(x) for x in p])
    return path


def centerline(mask, spacing):
    """Skeletonize a tubular mask -> ordered diameter path (voxel coords) + endpoints.
    Falls back to the centroid-along-longest-axis path if skeletonize yields <2 voxels or the
    graph pass fails, so we always return a usable ordered centerline + 2 endpoints."""
    from skimage.morphology import skeletonize
    import networkx as nx
    try:
        skel = skeletonize(mask)                          # skimage handles 3D (Lee thinning)
        coords = [tuple(int(v) for v in c) for c in np.argwhere(skel)]
        if len(coords) < 2:
            raise ValueError("skeleton too small")
        cset = set(coords)
        G = nx.Graph(); G.add_nodes_from(coords)
        offs = [(dx, dy, dz) for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)
                if not (dx == 0 and dy == 0 and dz == 0)]
        sp = np.asarray(spacing, dtype=float)
        for c in coords:
            for o in offs:
                nb = (c[0] + o[0], c[1] + o[1], c[2] + o[2])
                if nb in cset:
                    G.add_edge(c, nb, weight=float(np.linalg.norm(np.asarray(o) * sp)))
        comp = max(nx.connected_components(G), key=len)
        H = G.subgraph(comp)
        src = next(iter(comp))                            # tree-diameter: 2x farthest-node
        far1 = max(nx.single_source_dijkstra_path_length(H, src).items(), key=lambda kv: kv[1])[0]
        lengths, paths = nx.single_source_dijkstra(H, far1)
        far2 = max(lengths.items(), key=lambda kv: kv[1])[0]
        path = [list(p) for p in paths[far2]]
        return path, [list(far1), list(far2)]
    except Exception:
        path = _centroid_path(mask, spacing)
        ends = [path[0], path[-1]] if len(path) >= 2 else path
        return path, ends


def _resample_contour_mm(contour, spacing_xy, k_mm):
    """Uniform arc-length subsample of a closed 2D contour (voxel coords) every ~k_mm."""
    sp = np.asarray(spacing_xy, float)
    d = np.linalg.norm(np.diff(np.vstack([contour, contour[:1]]), axis=0) * sp, axis=1)
    cum = np.concatenate([[0], np.cumsum(d)])
    total = cum[-1]
    if total <= 0:
        return [0]
    targets = np.arange(0, total, max(k_mm, 1e-3))
    return [int(np.searchsorted(cum, t) % len(contour)) for t in targets]


def _curvature_high(contour, curv_rad):
    """Indices where the turning angle exceeds curv_rad (the defect notch is high-curvature)."""
    out = []
    n = len(contour)
    for i in range(n):
        a = contour[i] - contour[(i - 1) % n]
        b = contour[(i + 1) % n] - contour[i]
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na < 1e-6 or nb < 1e-6:
            continue
        cosang = np.clip(np.dot(a, b) / (na * nb), -1, 1)
        if np.arccos(cosang) >= curv_rad:
            out.append(i)
    return out


def adaptive_lasso(struct_vol, adjacent_vol, spacing, k_mm, band_mm, curv_rad, cap=60):
    """Per axial slice: positive lasso points on the structure contour, densified where the
    contour nears an ADJACENT chamber (septal band) or bends sharply. Returns {z: [[i,j],...]}."""
    from skimage.measure import find_contours
    sxy = (float(spacing[0]), float(spacing[1]))
    lasso = {}
    zs = np.unique(np.argwhere(struct_vol)[:, 2]) if struct_vol.any() else []
    for z in zs:
        sl = struct_vol[:, :, int(z)]
        if sl.sum() < 4:
            continue
        contours = find_contours(sl.astype(float), 0.5)
        if not contours:
            continue
        cont = max(contours, key=len)                 # (n,2) in (row=i, col=j)
        # distance from each pixel to the nearest adjacent-chamber voxel, in mm, on this slice
        adj = adjacent_vol[:, :, int(z)]
        dist_adj = ndi.distance_transform_edt(~adj.astype(bool), sampling=sxy) if adj.any() \
            else np.full(sl.shape, np.inf)
        keep = set(_resample_contour_mm(cont, sxy, k_mm))          # base uniform
        keep |= set(_curvature_high(cont, curv_rad))               # curvature densify
        for i, (r, c) in enumerate(cont):                          # proximity densify (septal band)
            if dist_adj[int(round(r)), int(round(c))] <= band_mm:
                keep.add(i)
        pts = [[int(round(cont[i][0])), int(round(cont[i][1]))] for i in sorted(keep)]
        if len(pts) > cap:                                         # bound interaction budget
            step = len(pts) / cap
            pts = [pts[int(k * step)] for k in range(cap)]
        if pts:
            lasso[int(z)] = pts
    return lasso


def negative_points(adjacent_vol, spacing, n=3):
    """A few interior points inside the union of adjacent structures (disambiguation negatives)."""
    if not adjacent_vol.any():
        return []
    return interior_points(adjacent_vol, spacing, n)


def process_case(lab_path, structures, args):
    img = nib.load(str(lab_path))
    arr = np.rint(np.asanyarray(img.dataobj)).astype(np.int16)
    affine = img.affine
    spacing = tuple(float(z) for z in img.header.get_zooms()[:3])
    curv_rad = np.deg2rad(args.lasso_curv_deg)
    present = {int(v) for v in np.unique(arr)} - {0}

    out = {"case": lab_path.name[:-len(FE)], "shape": list(arr.shape),
           "spacing": list(spacing), "affine": affine.tolist(), "structures": {}}

    for name, sid in structures.items():
        if sid not in present:
            continue
        mask = arr == sid
        rec = {"id": sid, "bbox_voxel": bbox_of(mask)}
        fg = interior_points(mask, spacing, args.n_fg)
        rec["interior_point_voxel"] = fg[0]
        rec["fg_points_voxel"] = fg
        rec["interior_point_world"] = vox2world(affine, fg[0])

        # adjacent structures = all other foreground (for negatives + septal proximity)
        adjacent = (arr != 0) & (arr != sid)
        rec["neg_points_voxel"] = negative_points(adjacent, spacing, 3)

        if name in VESSELS:
            cl, ends = centerline(mask, spacing)
            rec["centerline_voxel"] = cl
            rec["endpoints_voxel"] = ends
            rec["endpoints_world"] = [vox2world(affine, e) for e in ends]
            # per-endpoint radius (mm) = distance-transform value at the endpoint -> SeqSeg --seed X Y Z R
            dtv = ndi.distance_transform_edt(mask, sampling=spacing)
            seeds = []
            for e in ends:
                r = float(dtv[int(e[0]), int(e[1]), int(e[2])])
                wx, wy, wz = vox2world(affine, e)                    # nibabel affine = RAS+
                # SeqSeg/SimpleITK read images in LPS -> negate X,Y so seeds land in the CT's frame
                seeds.append([-wx, -wy, wz, max(r, 1.0)])            # [x,y,z,r_mm] LPS, radius floored 1mm
            rec["seeds_world_r"] = seeds
        elif name in CHAMBERS:
            rec["lasso_slices"] = adaptive_lasso(mask, adjacent, spacing,
                                                 args.lasso_k_mm, args.lasso_band_mm, curv_rad)
        out["structures"][name] = rec
    return out, arr, affine


def write_qc(out, arr, affine, qc_path):
    """Burn prompts into a label volume for visual QC in Slicer:
       10=fg points, 11=neg points, 12=centerline, 13=lasso."""
    qc = np.zeros_like(arr, dtype=np.int16)
    def put(ijk, val, rad=1):
        i, j, k = ijk
        qc[max(0, i - rad):i + rad + 1, max(0, j - rad):j + rad + 1, max(0, k - rad):k + rad + 1] = val
    for rec in out["structures"].values():
        for p in rec.get("fg_points_voxel", []): put(p, 10, 2)
        for p in rec.get("neg_points_voxel", []): put(p, 11, 2)
        for p in rec.get("centerline_voxel", []): put(p, 12, 0)
        for z, pts in rec.get("lasso_slices", {}).items():
            for (i, j) in pts: qc[i, j, int(z)] = 13
    nib.save(nib.Nifti1Image(qc, affine), str(qc_path))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels-dir", required=True, help="dir of <case>.nii.gz LCC seed labels")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--cases", default=None, help="comma-separated case ids (default: all in labels-dir)")
    ap.add_argument("--structures", default=None, help="comma-separated names (default: all in DEFAULT_STRUCTURES present)")
    ap.add_argument("--label-json", default=None, help="optional JSON {name: id} overriding DEFAULT_STRUCTURES")
    ap.add_argument("--n-fg", type=int, default=3)
    ap.add_argument("--lasso-k-mm", type=float, default=6.0)
    ap.add_argument("--lasso-band-mm", type=float, default=4.0)
    ap.add_argument("--lasso-curv-deg", type=float, default=35.0)
    ap.add_argument("--write-qc", action="store_true")
    args = ap.parse_args()

    labels = Path(args.labels_dir)
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    structures = json.loads(Path(args.label_json).read_text()) if args.label_json else dict(DEFAULT_STRUCTURES)
    if args.structures:
        want = {s.strip() for s in args.structures.split(",")}
        structures = {k: v for k, v in structures.items() if k in want}

    files = sorted(labels.glob("*" + FE))
    if args.cases:
        want = {c.strip() for c in args.cases.split(",")}
        files = [f for f in files if f.name[:-len(FE)] in want]
    if not files:
        sys.exit("no label files matched")

    for f in files:
        try:
            rec, arr, affine = process_case(f, structures, args)
        except Exception as e:
            print("  [ERROR] " + f.name + ": " + repr(e)[:120]); continue
        (out / (rec["case"] + "_prompts.json")).write_text(json.dumps(rec, indent=1))
        n_cl = sum(len(s.get("centerline_voxel", [])) for s in rec["structures"].values())
        n_la = sum(len(p) for s in rec["structures"].values() for p in s.get("lasso_slices", {}).values())
        print("  " + rec["case"] + ": structs=" + str(list(rec["structures"])) +
              "  centerline_pts=" + str(n_cl) + "  lasso_pts=" + str(n_la))
        if args.write_qc:
            write_qc(rec, arr, affine, out / (rec["case"] + "_prompts_qc" + FE))
    print("[done] prompts -> " + str(out))


if __name__ == "__main__":
    main()
