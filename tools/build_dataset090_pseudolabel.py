#!/usr/bin/env python3
"""
build_dataset090_pseudolabel.py
================================

Assemble Dataset090 for the FIRST pseudo-label training run: ImageCHD (Dataset071,
clean/myo-intact, LPS) + usable Fanwei (Dataset012) + usable clinical pseudo-labels.
Pseudo-label images come from their native datasets; their LABELS come from the
LCC'd native back-projections (predictions/ds071__grid2native_lcc/<case>.nii.gz).

Buckets (hardcoded from the reviewed lists):
  * usable       -> imagesTr/labelsTr, full-weight noisy pseudo-labels
  * quick_check  -> EXCLUDED from run 1 (stock nnU-Net has no per-case low weight);
                    recorded in the split config for a later low-weight pass
  * unusable     -> imagesTs only (image, NO label) — future test/unlabeled
  * dataset080   -> imagesTs only (image, NO label) — reserved expert test set
  * imagechd     -> Dataset071 cases (myo-intact), imagesTr/labelsTr, do the 5-fold VAL

Label schema (Dataset071): 1 LV-BP, 2 RV-BP, 3 LA, 4 RA, 5 Myo, 6 Aorta, 7 Pulmonary.
(myocardium = 5.)

Sanity checks (fail-loud, write nothing bad):
  every listed case exists; usable image/label match size+spacing+origin+direction;
  everything is LPS; label ids subset of the schema; ImageCHD cases keep myo(5);
  training set is disjoint from every held-out bucket (no leak / no train-val overlap).

Emits: imagesTr/, labelsTr/, imagesTs/, dataset.json, split_config.{json,csv},
split_meta.json, and a bucket summary.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
import SimpleITK as sitk

# ---------------------------------------------------------------- bucket lists
FANWEI_USABLE = [
    "CT_007_56810","CT_010_49","CT_030_8","CT_036_49","CT_041_089","CT_059_459",
    "CT_150_068_no","CT_159_69","CT_170_8","CT_177_69","CT_190_49","CT_192_059",
    "CT_201_09","CT_234_69","CT_260_5810","CT_273","CT_298_49","CT_309_069_no",
    "CT_325_459_no","CT_337_69","CT_342_156","CT_395_459_no","CT_429_69","CT_439_6",
    "CT_462_49","CT_477_49","CT_489_49","CT_491_136","CT_502_5910","CT_527_09",
    "CT_535_8","CT_555_069","CT_609_459","CT_697_57","CT_710_579","CT_728_459",
    "CT_790_0","CT_804_479","CT_842_06_1","CT_850_49_no","CT_851_49","CT_898_49",
    "CT_964_49","CT_964_5689","CT_993_05_2",
]
CLIN_USABLE = ["AVSD003","BAF001","BAF002","BAF008","BAF010"]
FANWEI_QUICK = [
    "CT_052_7910","CT_528_0579","CT_584_09_no","CT_704_49","CT_731_6","CT_747_68",
    "CT_754_49","CT_853_56_no","CT_860_8","CT_881_8","CT_914_49",
]
CLIN_QUICK = ["BAF004","BAF007"]
FANWEI_UNUSABLE = ["CT_110_69","CT_335_058","CT_790_069","CT_793_0569_no"]
CLIN_UNUSABLE = ["BAF005"]

SCHEMA = {"background":0,"LV-BP":1,"RV-BP":2,"LA":3,"RA":4,"Myo":5,"Aorta":6,"Pulmonary":7}
MYO_ID = 5
ALLOWED = set(SCHEMA.values())
FE = ".nii.gz"


def orient(img): return sitk.DICOMOrientImageFilter_GetOrientationFromDirectionCosines(img.GetDirection())
def label_ids(path): return {int(x) for x in np.unique(sitk.GetArrayViewFromImage(sitk.ReadImage(str(path))))}
def has_myo(path): return int(MYO_ID) in label_ids(path)
def geom(i): return (i.GetSize(), i.GetSpacing(), i.GetOrigin(), i.GetDirection())
def geom_match(a, b, atol=1e-3):
    ga, gb = geom(a), geom(b)
    return ga[0]==gb[0] and np.allclose(ga[1],gb[1],atol=atol) and np.allclose(ga[2],gb[2],atol=atol) and np.allclose(ga[3],gb[3],atol=atol)
def symlink(src: Path, dst: Path):
    if dst.exists() or dst.is_symlink(): dst.unlink()
    os.symlink(os.path.abspath(str(src)), dst)   # abspath, NOT resolve: keep the nnunet_CHD path
def log(*a): print(*a, flush=True)   # flush so a segfault's last line = the file that crashed


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nnunet-raw", default=os.environ.get("nnUNet_raw"))
    ap.add_argument("--imagechd-dataset", default="Dataset071_ImageCHDClinicalOrientation")
    ap.add_argument("--fanwei-dataset", default="Dataset012_Fanweidata")
    ap.add_argument("--clinical-root", default=None,
                    help="ClinicalImagesPHICleared root (default: <raw>/../ClinicalImagesPHICleared)")
    ap.add_argument("--dataset080", default="Dataset080_ClincalCaseSanjibDetailed")
    ap.add_argument("--lcc-subdir", default="predictions/ds071__grid2native_lcc",
                    help="where the LCC'd native pseudo-labels live under each source")
    ap.add_argument("--target-id", type=int, default=90)
    ap.add_argument("--target-name", default="ImageCHDPseudoCombined")
    ap.add_argument("--limit", type=int, default=None, help="cap ImageCHD + each bucket to N cases (debug)")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    if not args.nnunet_raw: sys.exit("ERROR: set $nnUNet_raw or pass --nnunet-raw")
    # do NOT .resolve(): $nnUNet_raw may be a SYMLINK into another tree (e.g.
    # nnunet_CHD/nnUNet_raw -> nnUNet/nnUNet_raw). Resolving would make raw.parent
    # point at the wrong tree and break the sibling ClinicalImagesPHICleared path.
    raw = Path(args.nnunet_raw)
    chd = raw / args.imagechd_dataset
    fanwei = raw / args.fanwei_dataset
    clin_root = Path(args.clinical_root) if args.clinical_root else (raw.parent / "ClinicalImagesPHICleared")
    ds080 = raw / args.dataset080

    # source locations: (image_dir, lcc_label_dir) per source
    FANWEI_IMG = fanwei / "imagesTr"
    FANWEI_LCC = fanwei / args.lcc_subdir
    CLIN_IMG = clin_root / "imagesTs"
    CLIN_LCC = clin_root / args.lcc_subdir

    failures = []
    def need_dir(p):
        if not Path(p).is_dir(): failures.append(f"missing dir: {p}")

    for p in (chd/"imagesTr", chd/"labelsTr", FANWEI_IMG, FANWEI_LCC, CLIN_IMG, CLIN_LCC):
        need_dir(p)
    if failures:
        print("[d090] PRE-FLIGHT FAILURES:"); [print("  -",f) for f in failures]
        sys.exit("Fix the above (did you run CHD_backproject_ds071.sh to make the LCC labels?)")

    # SimpleITK smoke test: if THIS crashes, the problem is the environment/libraries,
    # NOT any data file (isolates env-vs-file before we touch real images).
    log("[d090] SimpleITK smoke test (env check)...")
    import tempfile
    _t = sitk.GetImageFromArray(np.zeros((4, 4, 4), np.uint8)); _t.SetSpacing((1.0, 1.0, 1.0))
    _ = orient(_t); _ = {int(x) for x in np.unique(sitk.GetArrayViewFromImage(_t))}
    _tf = tempfile.mktemp(suffix=".nii.gz"); sitk.WriteImage(_t, _tf); _ = sitk.ReadImage(_tf); os.remove(_tf)
    log(f"[d090] SimpleITK OK ({sitk.Version.VersionString()}). Building...")

    target_folder = f"Dataset{args.target_id:03d}_{args.target_name}"
    dst = raw / target_folder
    if dst.exists() and not args.overwrite: sys.exit(f"ERROR: {dst} exists (use --overwrite)")
    for sub in ("imagesTr","labelsTr","imagesTs"): (dst/sub).mkdir(parents=True, exist_ok=True)

    rows = []   # split_config records
    def rec(src, cid, bucket, use, weight, notes=""):
        rows.append({"dataset_source":src,"case_id":cid,"bucket":bucket,
                     "intended_use":use,"label_weight":weight,"notes":notes})

    imagechd_ids, pseudo_train_ids = [], []

    # ---------------- ImageCHD base (Dataset071) -> imagesTr/labelsTr ----------------
    chd_imgs = {}
    for f in sorted((chd/"imagesTr").glob(f"*{FE}")):
        base = f.name[:-len(FE)]
        if base.endswith("_0000"): base = base[:-5]
        chd_imgs.setdefault(base, []).append(f)
    ids_chd = sorted(chd_imgs)
    if args.limit: ids_chd = ids_chd[: args.limit]
    log(f"[d090] reading {len(ids_chd)} ImageCHD label(s)...")
    for i, cid in enumerate(ids_chd, 1):
        chans = chd_imgs[cid]
        lab = chd/"labelsTr"/f"{cid}{FE}"
        log(f"[imagechd {i}/{len(ids_chd)}] {cid}  label={lab}")
        if not lab.is_file(): failures.append(f"[imagechd] {cid}: missing label"); continue
        try:
            myo = has_myo(lab)
        except Exception as e:
            failures.append(f"[imagechd] {cid}: label read error {e!r}"); continue
        if not myo: failures.append(f"[imagechd] {cid}: NO myocardium(5) — 071 should be myo-intact"); continue
        for ch in chans: symlink(ch, dst/"imagesTr"/ch.name)
        symlink(lab, dst/"labelsTr"/f"{cid}{FE}")
        imagechd_ids.append(cid); rec(args.imagechd_dataset, cid, "imagechd", "train+val (clean, myo-intact)", 1.0, "Dataset071 base")

    # ---------------- usable pseudo-labels -> imagesTr/labelsTr ----------------
    def add_pseudo(cid, img_dir, lcc_dir, src_name):
        img = img_dir/f"{cid}_0000{FE}"; lab = lcc_dir/f"{cid}{FE}"
        log(f"[usable] {cid}  img={img}")
        if not img.is_file(): failures.append(f"[usable] {cid}: missing image {img}"); return
        if not lab.is_file(): failures.append(f"[usable] {cid}: missing LCC label {lab}"); return
        try:
            im = sitk.ReadImage(str(img)); la = sitk.ReadImage(str(lab))
            oi, ol = orient(im), orient(la)
            ids = {int(x) for x in np.unique(sitk.GetArrayViewFromImage(la))}
            gm = geom_match(im, la)
        except Exception as e:
            failures.append(f"[usable] {cid}: read/geom error {e!r}"); return
        if oi!="LPS" or ol!="LPS": failures.append(f"[usable] {cid}: not LPS (img {oi} lab {ol})"); return
        if not ids <= ALLOWED: failures.append(f"[usable] {cid}: label ids {sorted(ids)} outside schema"); return
        if not gm: failures.append(f"[usable] {cid}: image/label geometry mismatch"); return
        symlink(img, dst/"imagesTr"/f"{cid}_0000{FE}")
        symlink(lab, dst/"labelsTr"/f"{cid}{FE}")
        pseudo_train_ids.append(cid); rec(src_name, cid, "usable", "pseudo_label_train", 1.0, "LCC native pseudo-label")
    fu = FANWEI_USABLE[: args.limit] if args.limit else FANWEI_USABLE
    cu = CLIN_USABLE[: args.limit] if args.limit else CLIN_USABLE
    for cid in fu: add_pseudo(cid, FANWEI_IMG, FANWEI_LCC, args.fanwei_dataset)
    for cid in cu: add_pseudo(cid, CLIN_IMG, CLIN_LCC, "ClinicalImagesPHICleared")

    # ---------------- held-out -> imagesTs (image only): unusable + quick_check + Dataset080 ----------------
    # NOTE: quick_check are EXCLUDED from training (no labels used), but their images go to
    # imagesTs so they get predicted with the trained model (the "remaining Fanwei" eval).
    ts_seen = set()
    log("[d090] staging held-out imagesTs (unusable + quick_check + Dataset080)...")
    def add_test(cid, img_dir, src_name, bucket, use, weight, notes):
        img = img_dir/f"{cid}_0000{FE}"
        log(f"[test/{bucket}] {cid}")
        if not img.is_file(): failures.append(f"[{bucket}] {cid}: missing image {img}"); return
        if cid in ts_seen: rec(src_name, cid, bucket, use+" (dup skipped)", weight, notes+"; dup basename"); return
        symlink(img, dst/"imagesTs"/f"{cid}_0000{FE}"); ts_seen.add(cid)
        rec(src_name, cid, bucket, use, weight, notes)
    for cid in FANWEI_UNUSABLE: add_test(cid, FANWEI_IMG, args.fanwei_dataset, "unusable", "held_out_test / unlabeled", 0.0, "unusable for myo-sensitive train")
    for cid in CLIN_UNUSABLE:   add_test(cid, CLIN_IMG, "ClinicalImagesPHICleared", "unusable", "held_out_test / unlabeled", 0.0, "unusable")
    for cid in FANWEI_QUICK:    add_test(cid, FANWEI_IMG, args.fanwei_dataset, "quick_check", "held_out_predict (opt low-weight later)", 0.3, "excluded from train run1")
    for cid in CLIN_QUICK:      add_test(cid, CLIN_IMG, "ClinicalImagesPHICleared", "quick_check", "held_out_predict (opt low-weight later)", 0.3, "excluded from train run1")
    if ds080.is_dir():
        for f in sorted((ds080/"imagesTr").glob(f"*_0000{FE}")):
            cid = f.name[:-len(f"_0000{FE}")]
            add_test(cid, ds080/"imagesTr", args.dataset080, "dataset080", "held_out_test (expert GT)", 0.0, "reserved expert test set")
    else:
        failures.append(f"note: {ds080} not found — Dataset080 test cases not added")

    # ---------------- LEAK CHECK ----------------
    train_ids = set(imagechd_ids) | set(pseudo_train_ids)
    held = set(FANWEI_QUICK)|set(CLIN_QUICK)|set(FANWEI_UNUSABLE)|set(CLIN_UNUSABLE)|ts_seen
    leak = train_ids & held
    if leak: failures.append(f"LEAK: held-out case(s) in training set: {sorted(leak)}")
    if len(train_ids) != len(imagechd_ids)+len(pseudo_train_ids):
        failures.append("duplicate case id across imagechd/usable training sets")

    if failures:
        print(f"[d090] {len(failures)} FAILURE(S):"); [print("  -",f) for f in failures]
        sys.exit(1)

    # ---------------- dataset.json + configs ----------------
    n_train = len(train_ids)
    (dst/"dataset.json").write_text(json.dumps({
        "channel_names": {"0":"CT"}, "labels": SCHEMA, "numTraining": n_train, "file_ending": FE,
        "name": target_folder,
        "description": (f"Pseudo-label run 1: {args.imagechd_dataset} (myo-intact) + usable Fanwei/clinical "
                        f"LCC pseudo-labels. Held-out (unusable + Dataset080) in imagesTs. Full-weight noisy labels."),
    }, indent=2))
    (dst/"split_meta.json").write_text(json.dumps(
        {"imagechd": sorted(imagechd_ids), "pseudo_train": sorted(pseudo_train_ids),
         "imagechd_source": args.imagechd_dataset}, indent=1))
    (dst/"split_config.json").write_text(json.dumps(rows, indent=1))
    with (dst/"split_config.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["dataset_source","case_id","bucket","intended_use","label_weight","notes"])
        w.writeheader(); w.writerows(rows)

    # ---------------- summary ----------------
    from collections import Counter
    by_bucket = Counter(r["bucket"] for r in rows)
    print(f"\n[d090] built {dst}")
    print("  bucket counts:")
    for b in ("imagechd","usable","quick_check","unusable","dataset080"):
        print(f"    {b:12s} {by_bucket.get(b,0)}")
    print(f"  imagesTr/labelsTr (TRAIN): {n_train}  = imagechd {len(imagechd_ids)} + usable {len(pseudo_train_ids)}")
    print(f"  imagesTs (HELD-OUT, images only): {len(ts_seen)}  (unusable + quick_check + Dataset080)")
    print(f"  quick_check (excluded from TRAIN run1; predicted as held-out): {len(FANWEI_QUICK)+len(CLIN_QUICK)}")
    print(f"  split config -> split_config.json / split_config.csv | folds -> split_meta.json")
    print(f"  NEXT: plan_and_preprocess -d {args.target_id}; then write splits (ImageCHD 5-fold val + pseudo in train)")


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except BaseException:
        import traceback
        traceback.print_exc()
        sys.stdout.flush(); sys.stderr.flush()
        raise
