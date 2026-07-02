#!/usr/bin/env python3
"""
audit_myocardial_closure.py  (READ-ONLY)
=======================================

Phase-0 make-or-break audit for the Dataset052 "close the chambers" plan.

The method rests on the myocardium (label 5) being *closable* — i.e. that, once
we insert separator planes (septal_defect + outflow_cap), `Myo ∪ separators`
bounds every blood-pool compartment. This tool measures, per case, whether that
premise holds on the ImageCHD ground truth. It NEVER writes to any dataset.

Per case it reports:
  * has_myo, myo voxels, Myo connected-component count (26-conn);
  * Myo Euler number (topology proxy; no persistent-homology lib on disk) and
    the volume Myo *encloses* (fill-holes − Myo) as a closability signal;
  * for each ordered blood-pool pair, direct 26-adjacency voxels = where the two
    lumens touch with NO myocardium between them (a separator will be needed);
  * for the should-be-separated pairs, the fraction of that contact band that is
    ringed by Myo (is the orifice bounded by wall on its sides?);
  * outflow derivability: is AO(6)/PA(7) 26-adjacent to a ventricular pool
    (LV1/RV2)? if neither artery has a ventricular orifice -> exclusion candidate;
  * a coarse exclusion estimate (no derivable outflow; TGA-IVS needs the
    diagnosis join, done in the summary).

Usage:
  python tools/audit_myocardial_closure.py --labels <dir_of_*.nii.gz> --out <csv> [--limit N]
Label ids are read from --dataset-json if given, else default ImageCHD scheme.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import nibabel as nib
from scipy import ndimage as ndi

try:
    import cc3d
    _HAVE_CC3D = True
except Exception:
    _HAVE_CC3D = False
try:
    from skimage.measure import euler_number
    _HAVE_EULER = True
except Exception:
    _HAVE_EULER = False

DEFAULT_IDS = {"LV": 1, "RV": 2, "LA": 3, "RA": 4, "Myo": 5, "AO": 6, "PA": 7}
POOLS = ["LV", "RV", "LA", "RA", "AO", "PA"]           # lumen compartments (Myo excluded)
# pairs we care about + their category
PAIRS = [
    ("LV", "RV", "VSD (ventricular septum)"),
    ("LA", "RA", "ASD (atrial septum)"),
    ("LV", "AO", "outflow"), ("RV", "AO", "outflow"),
    ("LV", "PA", "outflow"), ("RV", "PA", "outflow"),
    ("AO", "PA", "arterial (future)"),
    ("LV", "LA", "inlet valve (out of scope)"),
    ("RV", "RA", "inlet valve (out of scope)"),
]


def _ncomp(mask: np.ndarray) -> int:
    if not mask.any():
        return 0
    if _HAVE_CC3D:
        return int(cc3d.connected_components(mask.astype(np.uint8), connectivity=26).max())
    _, n = ndi.label(mask, structure=ndi.generate_binary_structure(3, 1))
    return int(n)


def _bbox(mask: np.ndarray, pad: int = 2):
    c = np.array(np.nonzero(mask))
    if c.size == 0:
        return None
    mn = np.maximum(c.min(1) - pad, 0)
    mx = np.minimum(c.max(1) + 1 + pad, mask.shape)
    return tuple(slice(int(a), int(b)) for a, b in zip(mn, mx))


def _direct_adj_voxels(seg: np.ndarray, ida: int, idb: int) -> int:
    """1-voxel 26-adjacency between two labels, on the union bbox for speed."""
    a = seg == ida
    b = seg == idb
    if not a.any() or not b.any():
        return 0
    sl = _bbox(a | b, 2)
    ac, bc = a[sl], b[sl]
    st = ndi.generate_binary_structure(3, 3)  # 26-conn
    return int((ndi.binary_dilation(ac, st) & bc).sum())


def _myo_ring_fraction(seg: np.ndarray, ida: int, idb: int, myo_id: int) -> float:
    """Of the a/b contact band, fraction whose local neighborhood contains Myo
    (is the orifice bounded by wall on its sides?)."""
    a = seg == ida
    b = seg == idb
    if not a.any() or not b.any():
        return float("nan")
    sl = _bbox(a | b, 3)
    ac, bc, mc = a[sl], b[sl], (seg[sl] == myo_id)
    st = ndi.generate_binary_structure(3, 3)
    band = ndi.binary_dilation(ac, st) & bc
    if not band.any():
        return float("nan")
    myo_near = ndi.binary_dilation(mc, st, iterations=2)
    return float((band & myo_near).sum()) / float(band.sum())


def _euler(mask: np.ndarray):
    if not _HAVE_EULER or not mask.any():
        return None
    sl = _bbox(mask, 2)
    try:
        return int(euler_number(mask[sl], connectivity=3))
    except Exception:
        return None


def audit_case(path: Path, ids: dict) -> dict:
    seg = np.asanyarray(nib.load(str(path)).dataobj).astype(np.int16)
    myo_id = ids["Myo"]
    myo = seg == myo_id
    row = {"case": path.name.replace(".nii.gz", "")}
    row["has_myo"] = bool(myo.any())
    row["myo_vox"] = int(myo.sum())
    row["myo_components"] = _ncomp(myo)
    row["myo_euler"] = _euler(myo)
    # closability signal: how much Myo encloses (fill holes − Myo)
    if myo.any():
        sl = _bbox(myo, 3)
        filled = ndi.binary_fill_holes(myo[sl])
        row["myo_encloses_vox"] = int((filled & ~myo[sl]).sum())
    else:
        row["myo_encloses_vox"] = 0
    # pool components
    for p in POOLS:
        row[f"{p}_comp"] = _ncomp(seg == ids[p])
    # adjacency + ring for the pairs of interest
    for a, b, _cat in PAIRS:
        row[f"adj_{a}_{b}"] = _direct_adj_voxels(seg, ids[a], ids[b])
    for a, b, cat in PAIRS:
        if cat in ("VSD (ventricular septum)", "outflow"):
            row[f"ring_{a}_{b}"] = round(_myo_ring_fraction(seg, ids[a], ids[b], myo_id), 3)
    # outflow derivability
    ao_out = (row["adj_LV_AO"] > 0) or (row["adj_RV_AO"] > 0)
    pa_out = (row["adj_LV_PA"] > 0) or (row["adj_PA_LV"] if "adj_PA_LV" in row else row["adj_RV_PA"]) > 0
    row["ao_has_ventricular_orifice"] = bool(ao_out)
    row["pa_has_ventricular_orifice"] = bool((row["adj_LV_PA"] > 0) or (row["adj_RV_PA"] > 0))
    row["vsd_gap_present"] = bool(row["adj_LV_RV"] > 0)
    # exclusion candidate: no derivable outflow orifice for at least one present artery
    ao_present = (seg == ids["AO"]).any()
    pa_present = (seg == ids["PA"]).any()
    row["exclude_no_outflow"] = bool((ao_present and not ao_out) or (pa_present and not row["pa_has_ventricular_orifice"]))
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--labels", required=True, help="dir of *.nii.gz label maps")
    ap.add_argument("--out", required=True, help="output CSV path")
    ap.add_argument("--dataset-json", default=None)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    ids = dict(DEFAULT_IDS)
    if args.dataset_json:
        labels = json.load(open(args.dataset_json)).get("labels", {})
        norm = {str(k).upper(): int(v) for k, v in labels.items()
                if isinstance(v, int) or str(v).isdigit()}
        for k in ids:
            if k.upper() in norm:
                ids[k] = norm[k.upper()]

    files = sorted(Path(args.labels).glob("*.nii.gz"))
    if args.limit:
        files = files[:args.limit]
    print(f"[audit] {len(files)} cases from {args.labels}  (cc3d={_HAVE_CC3D}, euler={_HAVE_EULER})")

    rows = []
    for f in files:
        r = audit_case(f, ids)
        rows.append(r)
        print(f"  {r['case']:16s} myo={r['has_myo']} comps={r['myo_components']} "
              f"euler={r['myo_euler']} LV-RV={r['adj_LV_RV']} LV-LA={r['adj_LV_LA']} "
              f"AO_out={r['ao_has_ventricular_orifice']} PA_out={r['pa_has_ventricular_orifice']} "
              f"exclude_no_outflow={r['exclude_no_outflow']}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({k for r in rows for k in r})
    keys = ["case"] + [k for k in keys if k != "case"]
    with open(args.out, "w", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)

    n = len(rows)
    miss_myo = sum(1 for r in rows if not r["has_myo"])
    excl = sum(1 for r in rows if r["exclude_no_outflow"])
    vsd = sum(1 for r in rows if r["vsd_gap_present"])
    print("\n=== SUMMARY ===")
    print(f"  cases                     : {n}")
    print(f"  missing Myo               : {miss_myo}")
    print(f"  LV-RV gap (VSD) present    : {vsd}")
    print(f"  exclusion candidates (no outflow orifice): {excl}")
    myo_multi = sum(1 for r in rows if r["myo_components"] > 1)
    print(f"  Myo >1 component          : {myo_multi}")
    print(f"  wrote {args.out}")


if __name__ == "__main__":
    main()
