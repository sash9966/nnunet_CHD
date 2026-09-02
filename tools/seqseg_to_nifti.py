#!/usr/bin/env python3
"""
seqseg_to_nifti.py  (workstream D)
Reduce SeqSeg's output to ONE binary NIfTI per case/vessel, on the reference CT grid — so it can be
stitched with the nnInteractive mask. Drops the VTK/SimVascular clutter from the picture entirely.

Two paths, tried in order:
  1. SeqSeg already wrote a NIfTI segmentation for that vessel -> use it (verify grid vs the CT).
  2. Only surfaces (.vtp/.vtk) -> VOXELIZE the surface onto the CT grid with VTK
     (vtkPolyDataToImageStencil). Surfaces are in LPS (VTK/SimpleITK); the CT grid is taken from the
     CT itself (origin/spacing/direction), so the result lands in the same frame as the CT and the
     nnInteractive mask.

Usage:
  python tools/seqseg_to_nifti.py --seqseg-dir out/seqseg_fanwei_bif1 --ct-dir <dir of <case>_0000.nii.gz> \
      --out-dir out/seqseg_fanwei_bif1_nifti [--surface-glob '*.vtp'] [--vessels Aorta,Pulmonary]
"""
import argparse, glob, os, sys
import numpy as np
import nibabel as nib

FE = ".nii.gz"
# preference order when several surfaces exist: assembled/final beats per-step debris
PREFER = ("final", "assembl", "surface", "vessel", "seg")


def pick_surface(vdir, pattern):
    cands = sorted(glob.glob(os.path.join(vdir, "**", pattern), recursive=True))
    cands = [c for c in cands if "_seqseg_single_staging" not in c]
    if not cands:
        return None
    def score(p):
        b = os.path.basename(p).lower()
        pref = min([i for i, k in enumerate(PREFER) if k in b], default=len(PREFER))
        return (pref, -os.path.getsize(p))          # preferred name, then biggest
    return sorted(cands, key=score)[0]


def voxelize(surf_path, ct_path):
    """Rasterize a closed surface onto the CT grid -> bool array in the CT's (i,j,k) index space.

    We transform the SURFACE POINTS into index space and rasterize there with unit spacing / zero
    origin / identity direction, instead of building an oriented output image. Reason:
    vtkPolyDataToImageStencil IGNORES a direction matrix, so on any volume whose direction is not
    identity (e.g. a RAS-stored NIfTI, which SimpleITK reports as direction diag(-1,-1,1)) an
    oriented-output approach silently rasterizes nothing. Index space is direction-agnostic.
    """
    import vtk
    from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk
    import SimpleITK as sitk

    r = vtk.vtkXMLPolyDataReader() if surf_path.endswith(".vtp") else vtk.vtkPolyDataReader()
    r.SetFileName(surf_path); r.Update()
    poly = r.GetOutput()
    if poly.GetNumberOfPoints() == 0:
        return None

    ct = sitk.ReadImage(ct_path)
    size = np.array(ct.GetSize(), dtype=int)                  # (i,j,k)
    spacing = np.array(ct.GetSpacing(), dtype=float)
    origin = np.array(ct.GetOrigin(), dtype=float)
    D = np.array(ct.GetDirection(), dtype=float).reshape(3, 3)

    # physical (LPS) -> continuous index:  p = origin + D @ (spacing * idx)
    pts = vtk_to_numpy(poly.GetPoints().GetData()).astype(float)
    idx = (np.linalg.inv(D) @ (pts - origin).T).T / spacing
    npts = vtk.vtkPoints(); npts.SetData(numpy_to_vtk(np.ascontiguousarray(idx), deep=1))
    poly.SetPoints(npts)

    ext = (0, int(size[0]) - 1, 0, int(size[1]) - 1, 0, int(size[2]) - 1)
    img = vtk.vtkImageData()
    img.SetDimensions(*[int(v) for v in size]); img.SetSpacing(1, 1, 1); img.SetOrigin(0, 0, 0)
    img.AllocateScalars(vtk.VTK_UNSIGNED_CHAR, 1)
    img.GetPointData().GetScalars().Fill(1)

    sten = vtk.vtkPolyDataToImageStencil()
    sten.SetInputData(poly)
    sten.SetOutputOrigin(0, 0, 0); sten.SetOutputSpacing(1, 1, 1)
    sten.SetOutputWholeExtent(*ext); sten.Update()

    cut = vtk.vtkImageStencil()
    cut.SetInputData(img); cut.SetStencilConnection(sten.GetOutputPort())
    cut.ReverseStencilOff(); cut.SetBackgroundValue(0); cut.Update()

    arr = vtk_to_numpy(cut.GetOutput().GetPointData().GetScalars())
    arr = arr.reshape(int(size[2]), int(size[1]), int(size[0])).transpose(2, 1, 0)   # -> (i,j,k)
    return arr > 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seqseg-dir", required=True, help="<case>/<vessel>/ tree")
    ap.add_argument("--ct-dir", required=True, help="reference CTs (<case>_0000.nii.gz)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--vessels", default="Aorta,Pulmonary")
    ap.add_argument("--surface-glob", default="*.vt*")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    vessels = [v.strip() for v in args.vessels.split(",")]

    cases = sorted(d for d in os.listdir(args.seqseg_dir) if os.path.isdir(os.path.join(args.seqseg_dir, d)))
    n_nii = n_vox = n_none = 0
    for c in cases:
        ct = os.path.join(args.ct_dir, c + "_0000" + FE)
        if not os.path.isfile(ct):
            print("  [skip %s] no CT at %s" % (c, ct)); continue
        ref = nib.load(ct); shape = ref.shape
        for v in vessels:
            vdir = os.path.join(args.seqseg_dir, c, v)
            if not os.path.isdir(vdir):
                continue
            out = os.path.join(args.out_dir, "%s_%s%s" % (c, v, FE))
            # path 1: an existing NIfTI on the right grid
            got = None; how = ""
            for f in sorted(glob.glob(os.path.join(vdir, "**", "*" + FE), recursive=True)):
                if "_seqseg_single_staging" in f:
                    continue
                m = np.asanyarray(nib.load(f).dataobj)
                if m.shape == shape:
                    got = m > 0; how = "nifti:" + os.path.basename(f); n_nii += 1; break
            # path 2: voxelize a surface
            if got is None:
                s = pick_surface(vdir, args.surface_glob)
                if s is not None:
                    try:
                        got = voxelize(s, ct)
                        how = "voxelized:" + os.path.basename(s); n_vox += 1
                    except Exception as e:
                        print("  [%s/%s] voxelize failed: %r" % (c, v, e)[:160])
            if got is None:
                print("  [%s/%s] nothing usable found" % (c, v)); n_none += 1; continue
            nib.save(nib.Nifti1Image(got.astype(np.uint8), ref.affine), out)
            print("  %-16s %-10s %8d vox  (%s)" % (c, v, int(got.sum()), how))
    print("[done] from-nifti=%d voxelized=%d missing=%d -> %s" % (n_nii, n_vox, n_none, args.out_dir))


if __name__ == "__main__":
    main()
