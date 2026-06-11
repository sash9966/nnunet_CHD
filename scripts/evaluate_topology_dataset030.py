#!/usr/bin/env python
"""
Topology-aware evaluation for CHD whole-heart segmentation.

Standard Dice does not reflect the failures this project cares about — vessel
discontinuities, branch fragmentation, and semantic flip-flopping.  This script
augments Dice with connectivity / centerline / consistency metrics so that
methods which genuinely fix topology become visible in the ranking.

Metrics (per case × class, then aggregated per method)
------------------------------------------------------
  1. Dice                        — standard overlap.
  2. subclass_mean               — mean Dice over foreground classes.
  3. clDice (AO, PA)             — centerline Dice (topology overlap).
  4. cc_count                    — predicted connected components per class.
  5. largest_cc_fraction         — |largest CC| / |all predicted fg| per class.
  6. false_disconnected_volume   — predicted fg outside the largest CC (voxels).
  7. centerline_recall (AO, PA)  — fraction of GT skeleton covered by prediction.
  8. junction_confusion          — AO↔PA and RA↔LA mislabel fractions.
  9. label_alternation           — # distinct predicted fg labels inside each GT
                                    vessel structure (flip-flop / leakage proxy).
 10. hard-case report            — per-case table, highlighting ct_1063.

Outputs (into --out)
--------------------
  summary.csv          method × metric means (+ ranking by subclass_mean).
  per_case.json        every metric for every (method, case, class).
  topology_table.csv   the topology-specific metrics, method × class.
  hard_cases.csv       worst cases per method (and ct_1063 explicitly).

Usage
-----
  python scripts/evaluate_topology_dataset030.py \
      --gt /path/gt_segmentations \
      --pred DA5_baseline=/path/predA RegionScaffold=/path/predB \
      --dataset_json /path/dataset.json --out results/topology_eval/

GT and prediction folders are matched by filename.  Requires SimpleITK, scipy,
scikit-image, numpy.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict
from os.path import isdir, isfile, join

import numpy as np

try:
    import SimpleITK as sitk
except ImportError as e:  # pragma: no cover
    raise SystemExit("SimpleITK is required: pip install SimpleITK") from e

try:
    from scipy import ndimage
except ImportError as e:  # pragma: no cover
    raise SystemExit("scipy is required: pip install scipy") from e

try:
    from skimage.morphology import skeletonize
except ImportError as e:  # pragma: no cover
    raise SystemExit("scikit-image is required: pip install scikit-image") from e


_AO_ALIASES = {"ao", "aorta"}
_PA_ALIASES = {"pa", "pulmonary", "pulmonary artery", "pulmonaryartery", "pulmonary trunk"}
_LA_ALIASES = {"la", "left atrium", "leftatrium"}
_RA_ALIASES = {"ra", "right atrium", "rightatrium"}


# ---------------------------------------------------------------------------
# dataset.json parsing
# ---------------------------------------------------------------------------

def parse_labels(dataset_json: dict):
    """Return (fg_id_to_name, name_to_id) excluding background."""
    labels = dataset_json.get("labels", {})
    fg = {int(idx): name for name, idx in labels.items() if int(idx) != 0}
    name_to_id = {}
    for name, idx in labels.items():
        name_to_id[str(name).lower().strip()] = int(idx)
    return fg, name_to_id


def resolve_named_id(name_to_id: dict, aliases: set):
    for alias in aliases:
        if alias in name_to_id:
            return name_to_id[alias]
    return None


# ---------------------------------------------------------------------------
# Metric primitives
# ---------------------------------------------------------------------------

def dice(gt_bin: np.ndarray, pred_bin: np.ndarray) -> float:
    g, p = gt_bin.sum(), pred_bin.sum()
    if g == 0 and p == 0:
        return float("nan")          # class absent in both — undefined
    inter = np.logical_and(gt_bin, pred_bin).sum()
    return 2.0 * inter / (g + p + 1e-8)


def connected_components(mask: np.ndarray):
    """Return (num_components, largest_fraction, false_disconnected_voxels)."""
    total = int(mask.sum())
    if total == 0:
        return 0, float("nan"), 0
    structure = ndimage.generate_binary_structure(mask.ndim, 1)  # 6-connectivity
    labeled, n = ndimage.label(mask, structure=structure)
    if n == 0:
        return 0, float("nan"), 0
    sizes = ndimage.sum(np.ones_like(labeled), labeled, index=range(1, n + 1))
    largest = int(sizes.max())
    return int(n), largest / total, int(total - largest)


def cl_dice(gt_bin: np.ndarray, pred_bin: np.ndarray):
    """Return (clDice, centerline_recall) for one binary class."""
    if gt_bin.sum() == 0 or pred_bin.sum() == 0:
        return float("nan"), float("nan")
    skel_gt = skeletonize(gt_bin)
    skel_pred = skeletonize(pred_bin)
    if skel_gt.sum() == 0 or skel_pred.sum() == 0:
        return float("nan"), float("nan")
    tprec = np.logical_and(skel_pred, gt_bin).sum() / (skel_pred.sum() + 1e-8)
    tsens = np.logical_and(skel_gt, pred_bin).sum() / (skel_gt.sum() + 1e-8)
    cld = 2.0 * tprec * tsens / (tprec + tsens + 1e-8)
    return float(cld), float(tsens)   # tsens == centerline recall


def confusion_fraction(gt: np.ndarray, pred: np.ndarray, a: int, b: int) -> float:
    """Fraction of GT==a voxels predicted as b (directional mislabel rate)."""
    gt_a = gt == a
    if gt_a.sum() == 0:
        return float("nan")
    return float((pred[gt_a] == b).sum() / gt_a.sum())


def label_alternation(gt: np.ndarray, pred: np.ndarray, struct_id: int, fg_ids) -> float:
    """# of distinct predicted foreground labels inside a GT structure.

    1 = perfectly coherent (only one fg label predicted inside the structure);
    higher = flip-flopping / label leakage.
    """
    region = gt == struct_id
    if region.sum() == 0:
        return float("nan")
    present = set(np.unique(pred[region]).tolist()) & set(fg_ids)
    return float(len(present))


# ---------------------------------------------------------------------------
# Per-case evaluation
# ---------------------------------------------------------------------------

def evaluate_case(gt: np.ndarray, pred: np.ndarray, fg, ao, pa, la, ra) -> dict:
    fg_ids = list(fg.keys())
    res = {"per_class": {}}
    dices = []
    for cid, cname in fg.items():
        gt_b = gt == cid
        pr_b = pred == cid
        d = dice(gt_b, pr_b)
        ncc, lcf, fdv = connected_components(pr_b)
        cls = {
            "dice": d,
            "cc_count": ncc,
            "largest_cc_fraction": lcf,
            "false_disconnected_volume": fdv,
        }
        if cid in (ao, pa):
            cld, recall = cl_dice(gt_b, pr_b)
            cls["clDice"] = cld
            cls["centerline_recall"] = recall
            cls["label_alternation"] = label_alternation(gt, pred, cid, fg_ids)
        res["per_class"][cname] = cls
        if not np.isnan(d):
            dices.append(d)

    res["subclass_mean"] = float(np.mean(dices)) if dices else float("nan")
    res["worst_subclass"] = float(np.min(dices)) if dices else float("nan")

    junc = {}
    if ao is not None and pa is not None:
        junc["AO_as_PA"] = confusion_fraction(gt, pred, ao, pa)
        junc["PA_as_AO"] = confusion_fraction(gt, pred, pa, ao)
    if la is not None and ra is not None:
        junc["RA_as_LA"] = confusion_fraction(gt, pred, ra, la)
        junc["LA_as_RA"] = confusion_fraction(gt, pred, la, ra)
    res["junction_confusion"] = junc
    return res


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------

def load_seg(path: str) -> np.ndarray:
    return sitk.GetArrayFromImage(sitk.ReadImage(path)).astype(np.int16)


def match_cases(gt_dir: str, pred_dir: str):
    gt_files = {f for f in os.listdir(gt_dir) if f.endswith((".nii.gz", ".nii", ".mha", ".nrrd"))}
    pred_files = {f for f in os.listdir(pred_dir)}
    return sorted(gt_files & pred_files)


def _nanmean(vals):
    arr = np.array([v for v in vals if v is not None and not (isinstance(v, float) and np.isnan(v))],
                   dtype=float)
    return float(arr.mean()) if arr.size else float("nan")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gt", required=True, help="Folder of ground-truth segmentations.")
    ap.add_argument("--pred", required=True, nargs="+",
                    help="One or more name=dir prediction folders.")
    ap.add_argument("--dataset_json", required=True, help="Path to dataset.json.")
    ap.add_argument("--out", required=True, help="Output directory.")
    ap.add_argument("--hard_case", default="ct_1063",
                    help="Case id to highlight in the hard-case report.")
    args = ap.parse_args()

    with open(args.dataset_json) as f:
        dataset_json = json.load(f)
    fg, name_to_id = parse_labels(dataset_json)
    ao = resolve_named_id(name_to_id, _AO_ALIASES)
    pa = resolve_named_id(name_to_id, _PA_ALIASES)
    la = resolve_named_id(name_to_id, _LA_ALIASES)
    ra = resolve_named_id(name_to_id, _RA_ALIASES)
    print(f"[eval] foreground labels: {fg}")
    print(f"[eval] AO={ao} PA={pa} LA={la} RA={ra}")

    os.makedirs(args.out, exist_ok=True)

    methods = {}
    for spec in args.pred:
        if "=" not in spec:
            raise SystemExit(f"--pred entries must be name=dir, got '{spec}'")
        name, d = spec.split("=", 1)
        if not isdir(d):
            raise SystemExit(f"Prediction dir not found for {name}: {d}")
        methods[name] = d

    per_case = defaultdict(dict)        # method -> case -> result
    for name, pdir in methods.items():
        cases = match_cases(args.gt, pdir)
        print(f"[eval] {name}: {len(cases)} matched cases")
        for fname in cases:
            case_id = fname
            for ext in (".nii.gz", ".nii", ".mha", ".nrrd"):
                if case_id.endswith(ext):
                    case_id = case_id[: -len(ext)]
                    break
            gt = load_seg(join(args.gt, fname))
            pred = load_seg(join(pdir, fname))
            if gt.shape != pred.shape:
                print(f"  [warn] shape mismatch {case_id}: gt{gt.shape} pred{pred.shape}, skipping")
                continue
            per_case[name][case_id] = evaluate_case(gt, pred, fg, ao, pa, la, ra)

    # ---- per-case JSON ----
    with open(join(args.out, "per_case.json"), "w") as f:
        json.dump(per_case, f, indent=2)

    # ---- method-level summary ----
    summary_rows = []
    for name in methods:
        cases = per_case[name]
        scm = _nanmean([c["subclass_mean"] for c in cases.values()])
        worst = _nanmean([c["worst_subclass"] for c in cases.values()])
        row = {"method": name, "n_cases": len(cases),
               "subclass_mean": scm, "mean_worst_subclass": worst}
        # topology aggregates on AO/PA
        for vname, vid in (("AO", ao), ("PA", pa)):
            if vid is None:
                continue
            cn = fg.get(vid)
            row[f"clDice_{vname}"] = _nanmean(
                [c["per_class"][cn].get("clDice") for c in cases.values() if cn in c["per_class"]])
            row[f"centerline_recall_{vname}"] = _nanmean(
                [c["per_class"][cn].get("centerline_recall") for c in cases.values() if cn in c["per_class"]])
            row[f"cc_count_{vname}"] = _nanmean(
                [c["per_class"][cn].get("cc_count") for c in cases.values() if cn in c["per_class"]])
            row[f"largest_cc_frac_{vname}"] = _nanmean(
                [c["per_class"][cn].get("largest_cc_fraction") for c in cases.values() if cn in c["per_class"]])
            row[f"label_alternation_{vname}"] = _nanmean(
                [c["per_class"][cn].get("label_alternation") for c in cases.values() if cn in c["per_class"]])
        summary_rows.append(row)

    summary_rows.sort(key=lambda r: (-r["subclass_mean"]
                                     if not np.isnan(r["subclass_mean"]) else 0))
    for rank, r in enumerate(summary_rows, 1):
        r["rank"] = rank

    fieldnames = ["rank", "method", "n_cases", "subclass_mean", "mean_worst_subclass"]
    extra_fields = [k for k in summary_rows[0] if k not in fieldnames] if summary_rows else []
    fieldnames += sorted(extra_fields)
    with open(join(args.out, "summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in summary_rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})

    # ---- per-class topology table (method × class) ----
    with open(join(args.out, "topology_table.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "class", "dice", "clDice", "centerline_recall",
                    "cc_count", "largest_cc_fraction", "false_disconnected_volume",
                    "label_alternation"])
        for name in methods:
            cases = per_case[name]
            for cid, cn in fg.items():
                vals = [c["per_class"][cn] for c in cases.values() if cn in c["per_class"]]
                if not vals:
                    continue
                w.writerow([
                    name, cn,
                    f"{_nanmean([v.get('dice') for v in vals]):.4f}",
                    f"{_nanmean([v.get('clDice') for v in vals]):.4f}",
                    f"{_nanmean([v.get('centerline_recall') for v in vals]):.4f}",
                    f"{_nanmean([v.get('cc_count') for v in vals]):.2f}",
                    f"{_nanmean([v.get('largest_cc_fraction') for v in vals]):.4f}",
                    f"{_nanmean([v.get('false_disconnected_volume') for v in vals]):.1f}",
                    f"{_nanmean([v.get('label_alternation') for v in vals]):.2f}",
                ])

    # ---- hard cases (incl. the highlighted case) ----
    with open(join(args.out, "hard_cases.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "case", "subclass_mean", "worst_subclass", "is_highlight"])
        for name in methods:
            cases = per_case[name]
            ranked = sorted(cases.items(),
                            key=lambda kv: kv[1]["subclass_mean"]
                            if not np.isnan(kv[1]["subclass_mean"]) else 1.0)
            shown = ranked[:5]
            if args.hard_case in cases and all(args.hard_case != c for c, _ in shown):
                shown.append((args.hard_case, cases[args.hard_case]))
            for cid, r in shown:
                w.writerow([name, cid, f"{r['subclass_mean']:.4f}",
                            f"{r['worst_subclass']:.4f}",
                            "yes" if cid == args.hard_case else ""])

    # ---- console ranking ----
    print("\n=== Ranking by subclass_mean ===")
    for r in summary_rows:
        extra = ""
        if ao is not None:
            extra = f"  clDice_AO={r.get('clDice_AO', float('nan')):.3f}"
        if pa is not None:
            extra += f"  clDice_PA={r.get('clDice_PA', float('nan')):.3f}"
        print(f"  #{r['rank']} {r['method']:<28} "
              f"subclass_mean={r['subclass_mean']:.4f}{extra}")
    print(f"\n[eval] wrote summary.csv, topology_table.csv, hard_cases.csv, "
          f"per_case.json to {args.out}")


if __name__ == "__main__":
    main()
