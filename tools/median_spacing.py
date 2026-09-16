#!/usr/bin/env python3
"""
median_spacing.py
Report the median voxel spacing of a raw nnU-Net image folder in **nnU-Net's own axis convention**,
so the result can be fed straight to `nnUNetv2_plan_experiment -overwrite_target_spacing`.

Axis convention (this is the part that is easy to get wrong):
  SimpleITK's GetSpacing() returns (x, y, z). nnU-Net's SimpleITKIO stores
  `list(GetSpacing())[::-1]` -> (z, y, x) = (slice, row, col), and the dataset fingerprint's
  'spacings' use that order. `-overwrite_target_spacing` is consumed by
  determine_fullres_target_spacing(), which the planner transposes AFTERWARDS
  (fullres_spacing[transpose_forward]) -- so the value passed on the CLI must be PRE-transpose
  (z, y, x). That is what this script prints.

Also prints the per-case spread and the transpose axis that nnU-Net would derive
(argmax of the target spacing), because overriding the spacing can silently change
transpose_forward and therefore the data layout.

Header-only reads (ReadImageInformation), so it is fast and never loads pixel data.

Usage:
  python tools/median_spacing.py --images-dir $nnUNet_raw/Dataset080_.../imagesTr [--out-json x.json]
"""
import argparse, glob, json, os, sys

import numpy as np
import SimpleITK as sitk


def header_spacing_zyx(path):
    r = sitk.ImageFileReader()
    r.SetFileName(path)
    r.ReadImageInformation()
    return [abs(v) for v in list(r.GetSpacing())[::-1]]      # (x,y,z) -> (z,y,x)


def main():
    a = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a.add_argument("--images-dir", required=True)
    a.add_argument("--out-json", default=None)
    a.add_argument("--round", type=int, default=4, help="decimals to round the reported median to")
    args = a.parse_args()

    files = sorted(glob.glob(os.path.join(args.images_dir, "*.nii.gz")))
    if not files:
        sys.exit("FATAL: no .nii.gz in %s" % args.images_dir)

    rows = []
    for f in files:
        try:
            rows.append((os.path.basename(f), header_spacing_zyx(f)))
        except Exception as e:
            print("  [warn] could not read header of %s: %s" % (os.path.basename(f), e))
    if not rows:
        sys.exit("FATAL: no readable headers")

    sp = np.array([r[1] for r in rows], dtype=float)
    med = np.median(sp, axis=0)
    print("\nper-case spacing  (z, y, x)  [nnU-Net order]")
    for name, s in rows:
        print("  %-34s %8.4f %8.4f %8.4f" % (name, s[0], s[1], s[2]))
    print("\n  n            = %d" % len(rows))
    print("  median (zyx) = %8.4f %8.4f %8.4f" % tuple(med))
    print("  min    (zyx) = %8.4f %8.4f %8.4f" % tuple(sp.min(axis=0)))
    print("  max    (zyx) = %8.4f %8.4f %8.4f" % tuple(sp.max(axis=0)))
    print("  mean   (zyx) = %8.4f %8.4f %8.4f" % tuple(sp.mean(axis=0)))

    hetero = float((sp.max(axis=0) / np.maximum(sp.min(axis=0), 1e-9)).max())
    if hetero > 1.5:
        print("\n  [!] spacing is heterogeneous across cases (max/min ratio %.2f on at least one axis)." % hetero)
        print("      A single median is then a weak summary of the target domain -- worth reporting the")
        print("      spread alongside any result obtained by matching the median.")

    med_r = [round(float(v), args.round) for v in med]
    ax = int(np.argmax(med))
    tf = [ax] + [i for i in range(3) if i != ax]
    print("\n  transpose_forward nnU-Net would derive from THIS spacing: %s  (argmax axis = %d)" % (tf, ax))
    print("\n  CLI value (pre-transpose, z y x):")
    print("      -overwrite_target_spacing %s %s %s" % tuple(str(v) for v in med_r))

    if args.out_json:
        json.dump({"images_dir": args.images_dir, "n": len(rows),
                   "median_zyx": med_r,
                   "min_zyx": [float(v) for v in sp.min(axis=0)],
                   "max_zyx": [float(v) for v in sp.max(axis=0)],
                   "transpose_forward_implied": tf,
                   "per_case": {n: [float(x) for x in s] for n, s in rows}},
                  open(args.out_json, "w"), indent=2)
        print("\n  json -> %s" % args.out_json)


if __name__ == "__main__":
    main()
