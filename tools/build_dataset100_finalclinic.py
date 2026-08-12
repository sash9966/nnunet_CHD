#!/usr/bin/env python3
"""
build_dataset100_finalclinic.py
===============================

Clinic-facing ALL-DATA training dataset:  Dataset100 = Dataset091 (verbatim) + ALL Dataset080
clinical detailed cases added to TRAINING.

*** This is a deployment / qualitative-review model, NOT an evaluation dataset. ***
Dataset080 is intentionally in training here, so Dataset080 Dice from THIS model is NOT an
unbiased performance estimate.

Does not touch Dataset090 or Dataset091 (symlink-only, no writes into the sources).

Outputs in the new dataset folder:
  imagesTr/, labelsTr/ (symlinks), dataset.json, manifest.csv, README.md, build_report.json

Validation (fails loudly): every image has a matching label; image/label shapes match; all label
ids in {0..7}; all Dataset080 cases included; all Dataset091 cases preserved; none of the excluded
bad ds090 pseudo-label cases sneak in (unless they legitimately arrive via Dataset080/Dataset091).
"""
import argparse, csv, json, os, shutil, sys
from pathlib import Path

import numpy as np
import nibabel as nib   # memory-safe reader (no SimpleITK GetArrayView use-after-free)

FE = ".nii.gz"
LABELS = {"background": 0, "LV-BP": 1, "RV-BP": 2, "LA": 3, "RA": 4, "Myo": 5, "Aorta": 6, "Pulmonary": 7}
# user-facing id mapping (same ids): 6=Ao, 7=PA
ALLOWED_IDS = set(range(8))  # {0..7}
# bad ds090 pseudo-label cases that must NOT enter training (unless via Dataset080/091 by design)
EXCLUDE_BAD = {"CT_704_49", "CT_853_56_no", "CT_881_8", "BAF005",
               "CT_110_69", "CT_335_058", "CT_790_069", "CT_793_0569_no"}


def symlink(src, dst):
    dst = Path(dst)
    if dst.exists() or dst.is_symlink(): dst.unlink()
    os.symlink(os.path.abspath(str(src)), dst)


def spatial_shape(path):
    return tuple(int(x) for x in nib.load(str(path)).shape[:3])


def label_id_set(path):
    data = np.asanyarray(nib.load(str(path)).dataobj)
    return {int(x) for x in np.unique(np.rint(data).astype(np.int64))}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nnunet-raw", default=os.environ.get("nnUNet_raw"))
    ap.add_argument("--src-dataset", default="Dataset091_ImageCHDPseudoCombinedV2")
    ap.add_argument("--d080-images", default=None,
                    help="default: <raw>/Dataset080_ClinicalCaseSanjibDetailed/imagesTr")
    ap.add_argument("--d080-labels", default=None,
                    help="default: <raw>/Dataset080_ClinicalCaseSanjibDetailed/labelsTr")
    ap.add_argument("--d080-name", default="Dataset080_ClinicalCaseSanjibDetailed")
    ap.add_argument("--target-id", type=int, default=100)
    ap.add_argument("--target-name", default="FinalClinic")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    if not args.nnunet_raw: sys.exit("ERROR: set $nnUNet_raw or pass --nnunet-raw")
    raw = Path(args.nnunet_raw)
    src = raw / args.src_dataset
    if not (src/"imagesTr").is_dir(): sys.exit(f"ERROR: {src}/imagesTr missing (build Dataset091 first)")
    d080_img = Path(args.d080_images) if args.d080_images else (raw/args.d080_name/"imagesTr")
    d080_lab = Path(args.d080_labels) if args.d080_labels else (raw/args.d080_name/"labelsTr")
    if not d080_img.is_dir(): sys.exit(f"ERROR: Dataset080 images dir not found: {d080_img}")
    if not d080_lab.is_dir(): sys.exit(f"ERROR: Dataset080 labels dir not found: {d080_lab}")

    target = f"Dataset{args.target_id:03d}_{args.target_name}"
    dst = raw / target
    if dst.exists():
        if not args.overwrite: sys.exit(f"ERROR: {dst} exists (use --overwrite)")
        shutil.rmtree(dst)
    (dst/"imagesTr").mkdir(parents=True); (dst/"labelsTr").mkdir(parents=True)

    # label_type classification from Dataset091's split_meta
    imagechd, pseudo = set(), set()
    smp = src/"split_meta.json"
    if smp.is_file():
        sm = json.loads(smp.read_text())
        imagechd = set(sm.get("imagechd", [])); pseudo = set(sm.get("pseudo_train", []))

    rows = []              # manifest rows
    d091_ids, d080_ids = [], []
    errors = []

    def record(cid, img_src, lab_src, source_ds, ltype):
        rows.append({"case_id": cid, "image_path": os.path.abspath(str(img_src)),
                     "label_path": os.path.abspath(str(lab_src)), "source_dataset": source_ds,
                     "label_type": ltype, "included_for_training": "true"})

    def validate(cid, img_dst, lab_dst, full_label_scan):
        if not Path(lab_dst).exists(): errors.append(f"{cid}: no matching label"); return
        try:
            ish, lsh = spatial_shape(img_dst), spatial_shape(lab_dst)
        except Exception as e:
            errors.append(f"{cid}: shape read error {e!r}"); return
        if ish != lsh: errors.append(f"{cid}: image/label shape mismatch {ish} vs {lsh}")
        if full_label_scan:
            try:
                ids = label_id_set(lab_dst)
            except Exception as e:
                errors.append(f"{cid}: label read error {e!r}"); return
            bad = ids - ALLOWED_IDS
            if bad: errors.append(f"{cid}: label ids outside 0-7: {sorted(bad)}")

    # ---- 1) Dataset091 training (symlink verbatim) ----
    for img in sorted((src/"imagesTr").glob(f"*_0000{FE}")):
        cid = img.name[:-len(f"_0000{FE}")]
        lab = src/"labelsTr"/f"{cid}{FE}"
        symlink(img, dst/"imagesTr"/img.name)
        if lab.is_file(): symlink(lab, dst/"labelsTr"/lab.name)
        ltype = "imagechd_ground_truth" if cid in imagechd else ("pseudo_label" if cid in pseudo else "unknown")
        record(cid, img, lab, args.src_dataset, ltype)
        d091_ids.append(cid)
        validate(cid, dst/"imagesTr"/img.name, dst/"labelsTr"/lab.name, full_label_scan=True)

    # ---- 2) Dataset080 clinical detailed -> training (symlink) ----
    for img in sorted(d080_img.glob(f"*_0000{FE}")):
        cid = img.name[:-len(f"_0000{FE}")]
        lab = d080_lab/f"{cid}{FE}"
        symlink(img, dst/"imagesTr"/img.name)
        if lab.is_file(): symlink(lab, dst/"labelsTr"/lab.name)
        record(cid, img, lab, args.d080_name, "expert_manual")
        d080_ids.append(cid)
        validate(cid, dst/"imagesTr"/img.name, dst/"labelsTr"/lab.name, full_label_scan=True)

    # ---- 3) required checks ----
    train_ids = [r["case_id"] for r in rows]
    train_set = set(train_ids)
    if len(train_ids) != len(train_set):
        dups = sorted({c for c in train_ids if train_ids.count(c) > 1})
        errors.append(f"duplicate case ids in training: {dups}")
    if not set(d091_ids) <= train_set: errors.append("some Dataset091 cases missing from training")
    if d080_ids and not set(d080_ids) <= train_set: errors.append("some Dataset080 cases missing from training")
    # excluded bad cases must not enter unless they legitimately came via 091/080
    leaked = (EXCLUDE_BAD & train_set) - set(d080_ids) - set(d091_ids)
    if leaked: errors.append(f"excluded bad ds090 pseudo-label cases present: {sorted(leaked)}")

    if errors:
        print("[d100] VALIDATION FAILURES:"); [print("  -", e) for e in errors]
        sys.exit("Build failed validation — nothing usable was written beyond symlinks; fix and re-run with --overwrite.")

    # ---- 4) dataset.json ----
    n_train = len(rows)
    dj = {"channel_names": {"0": "CT"}, "labels": LABELS, "numTraining": n_train,
          "file_ending": FE, "name": target,
          "description": ("Clinic-facing ALL-DATA model: Dataset091 + all Dataset080 clinical "
                          "detailed cases in TRAINING. Deployment/qualitative review only — "
                          "Dataset080 is in training, so its Dice from this model is NOT unbiased.")}
    (dst/"dataset.json").write_text(json.dumps(dj, indent=2))

    # ---- 5) manifest.csv ----
    with open(dst/"manifest.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["case_id", "image_path", "label_path",
                                          "source_dataset", "label_type", "included_for_training"])
        w.writeheader(); [w.writerow(r) for r in rows]

    # ---- 6) README ----
    (dst/"README.md").write_text(
        "# " + target + "\n\n"
        "This dataset includes Dataset080 clinical cases in training and is intended for "
        "clinic-facing all-data model training, not unbiased evaluation.\n\n"
        f"- Training cases: {n_train} (Dataset091: {len(d091_ids)} + Dataset080: {len(d080_ids)})\n"
        "- Labels: 0=background 1=LV-BP 2=RV-BP 3=LA 4=RA 5=Myo 6=Ao 7=PA\n"
        "- Do NOT report Dataset080 Dice from this model as unbiased performance.\n")

    # ---- 6b) split_meta for 5-fold training: ImageCHD does the val folds; the pseudo-labels
    #         and Dataset080 clinic cases stay train-only in every fold (same idea as D091). ----
    train_only = sorted(pseudo | set(d080_ids))
    (dst/"split_meta.json").write_text(json.dumps(
        {"imagechd": sorted(imagechd), "train_only": train_only, "d080": sorted(d080_ids)}, indent=1))

    # ---- 7) build report ----
    lt_counts = {t: sum(1 for r in rows if r["label_type"] == t)
                 for t in sorted({r["label_type"] for r in rows})}
    (dst/"build_report.json").write_text(json.dumps(
        {"target": target, "n_train": n_train, "n_d091": len(d091_ids), "n_d080": len(d080_ids),
         "d080_cases": sorted(d080_ids), "excluded_bad_confirmed_absent": sorted(EXCLUDE_BAD),
         "label_types": lt_counts}, indent=2))

    print(f"[d100] built {dst}")
    print(f"  training cases: {n_train}  = Dataset091 {len(d091_ids)} + Dataset080 {len(d080_ids)}")
    print(f"  Dataset080 added: {sorted(d080_ids)}")
    print(f"  label_type counts: {lt_counts}")
    print(f"  excluded bad cases confirmed ABSENT: {sorted(EXCLUDE_BAD)}")
    print(f"  wrote dataset.json, manifest.csv, README.md, build_report.json")
    print(f"  NEXT: plan_and_preprocess -d {args.target_id}; train fold 'all'")


if __name__ == "__main__":
    main()
