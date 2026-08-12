#!/usr/bin/env python3
"""
build_dataset091_from_090.py
============================

Dataset091 = Dataset090, IDENTICAL, plus a few QC-approved ds090 pseudo-label cases
promoted from held-out (imagesTs) into training. Symlink-only (no sitk reads).

- ImageCHD base + original usable pseudo-labels: copied verbatim from Dataset090 (imagesTr/labelsTr).
- Each promoted case: image from Dataset090/imagesTs, label from the ds090 QC'd predictions
  (default predictions/ds090__grid2native_lcc/<case>.nii.gz) -> added to imagesTr/labelsTr.
- imagesTs = Dataset090/imagesTs MINUS the promoted cases (remaining held-out).
- Fold strategy preserved: split_meta keeps the same imagechd list; promoted cases join
  pseudo_train (train-only, never val) -> 090 and 091 share identical ImageCHD val folds.

Safety: refuses to promote any case in --exclude or in Dataset090's dataset080 bucket
(so Dataset080 / BAF004 can never leak into training). Leak-checks train vs imagesTs.
"""
import argparse, json, os, shutil, sys
from pathlib import Path

FE = ".nii.gz"


def symlink(src, dst):
    dst = Path(dst)
    if dst.exists() or dst.is_symlink(): dst.unlink()
    os.symlink(os.path.abspath(str(src)), dst)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nnunet-raw", default=os.environ.get("nnUNet_raw"))
    ap.add_argument("--src-dataset", default="Dataset090_ImageCHDPseudoCombined")
    ap.add_argument("--target-id", type=int, default=91)
    ap.add_argument("--target-name", default="ImageCHDPseudoCombinedV2")
    ap.add_argument("--promoted", required=True, help="comma-separated case ids to promote to training")
    ap.add_argument("--promoted-label-dir", default=None,
                    help="dir with QC-approved labels for promoted cases "
                         "(default: <src>/predictions/ds090__grid2native_lcc)")
    ap.add_argument("--promoted-image-dir", default=None,
                    help="extra dir(s) (comma-sep) to search for promoted images, tried before the "
                         "defaults (<src>/imagesTs, <raw>/Dataset012_Fanweidata/imagesTr)")
    ap.add_argument("--exclude", default="", help="comma-separated case ids that must NOT be promoted")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    if not args.nnunet_raw: sys.exit("ERROR: set $nnUNet_raw or pass --nnunet-raw")
    raw = Path(args.nnunet_raw)
    src = raw / args.src_dataset
    if not (src/"imagesTr").is_dir(): sys.exit(f"ERROR: {src}/imagesTr missing (build+train Dataset090 first)")
    promoted = [c.strip() for c in args.promoted.split(",") if c.strip()]
    exclude = {c.strip() for c in args.exclude.split(",") if c.strip()}
    label_dir = Path(args.promoted_label_dir) if args.promoted_label_dir else (src/"predictions"/"ds090__grid2native_lcc")

    # ---- safety: promoted must not overlap exclude or Dataset080 ----
    bad = set(promoted) & exclude
    if bad: sys.exit(f"ERROR: promoted cases are in --exclude: {sorted(bad)}")
    d080 = set()
    scp = src/"split_config.json"
    if scp.is_file():
        for r in json.loads(scp.read_text()):
            if r.get("bucket") == "dataset080": d080.add(r["case_id"])
    bad2 = set(promoted) & d080
    if bad2: sys.exit(f"ERROR: promoted cases are Dataset080 held-out test cases: {sorted(bad2)}")

    target = f"Dataset{args.target_id:03d}_{args.target_name}"
    dst = raw / target
    if dst.exists():
        if not args.overwrite: sys.exit(f"ERROR: {dst} exists (use --overwrite)")
        shutil.rmtree(dst)
    for s in ("imagesTr", "labelsTr", "imagesTs"): (dst/s).mkdir(parents=True, exist_ok=True)

    # ---- 1) copy ALL of Dataset090 training verbatim ----
    for f in sorted((src/"imagesTr").glob(f"*{FE}")): symlink(f, dst/"imagesTr"/f.name)
    for f in sorted((src/"labelsTr").glob(f"*{FE}")): symlink(f, dst/"labelsTr"/f.name)
    base_train = len(list((dst/"labelsTr").glob(f"*{FE}")))

    # ---- 2) promoted -> training (image from D090/imagesTs OR the Fanwei/clinical source;
    #         label from the ds090 QC dir) ----
    img_search = []
    if args.promoted_image_dir:
        img_search += [Path(p) for p in args.promoted_image_dir.split(",") if p.strip()]
    img_search += [src/"imagesTs", raw/"Dataset012_Fanweidata"/"imagesTr"]

    def find_img(cid):
        for d in img_search:
            p = d/f"{cid}_0000{FE}"
            if p.is_file(): return p
        return None

    fails, promoted_ok = [], []
    for cid in promoted:
        img = find_img(cid)
        lab = label_dir/f"{cid}{FE}"
        if img is None: fails.append(f"{cid}: image not found in {[str(d) for d in img_search]}"); continue
        if not lab.is_file(): fails.append(f"{cid}: QC label not found ({lab})"); continue
        symlink(img, dst/"imagesTr"/f"{cid}_0000{FE}")
        symlink(lab, dst/"labelsTr"/f"{cid}{FE}")
        promoted_ok.append(cid)
    if fails:
        print("[d091] FAILURES:"); [print("  -", x) for x in fails]
        sys.exit("Fix: run Dataset090 Phase-3 held-out inference (ds090__grid2native_lcc) "
                 "or pass --promoted-label-dir pointing at your QC'd labels.")

    # ---- 3) imagesTs = 090 imagesTs MINUS promoted ----
    promoted_set = set(promoted_ok)
    n_ts = 0
    for f in sorted((src/"imagesTs").glob(f"*_0000{FE}")):
        cid = f.name[: -len(f"_0000{FE}")]
        if cid in promoted_set: continue
        symlink(f, dst/"imagesTs"/f.name); n_ts += 1

    # ---- 4) dataset.json (copy 090's scheme; bump numTraining) ----
    dj = json.loads((src/"dataset.json").read_text())
    n_train = len(list((dst/"labelsTr").glob(f"*{FE}")))
    dj["numTraining"] = n_train; dj["name"] = target
    dj["description"] = f"Dataset090 + {len(promoted_ok)} QC-approved ds090 pseudo-label cases promoted to train."
    (dst/"dataset.json").write_text(json.dumps(dj, indent=2))

    # ---- 5) split_meta: same imagechd; pseudo_train += promoted ----
    sm = json.loads((src/"split_meta.json").read_text())
    sm["pseudo_train"] = sorted(set(sm.get("pseudo_train", [])) | promoted_set)
    sm["promoted_from_090"] = sorted(promoted_ok)
    (dst/"split_meta.json").write_text(json.dumps(sm, indent=1))

    # ---- 6) leak checks ----
    train_ids = {f.name[: -len(FE)] for f in (dst/"labelsTr").glob(f"*{FE}")}
    ts_ids = {f.name[: -len(f"_0000{FE}")] for f in (dst/"imagesTs").glob(f"*_0000{FE}")}
    if train_ids & ts_ids: sys.exit(f"LEAK: case in both train and imagesTs: {sorted(train_ids & ts_ids)}")
    if d080 & train_ids: sys.exit(f"LEAK: Dataset080 case in training: {sorted(d080 & train_ids)}")

    print(f"[d091] built {dst}")
    print(f"  train: {n_train}  = 090 base {base_train} + promoted {len(promoted_ok)}")
    print(f"  promoted (QC-approved ds090): {promoted_ok}")
    print(f"  imagesTs (remaining held-out): {n_ts}")
    print(f"  Dataset080 confirmed NOT in training: {sorted(d080)}")
    print(f"  NEXT: plan_and_preprocess -d {args.target_id}; splits (ImageCHD 5-fold val + pseudo train-only)")


if __name__ == "__main__":
    main()
