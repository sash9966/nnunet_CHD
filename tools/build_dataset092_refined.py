#!/usr/bin/env python3
"""
build_dataset092_refined.py
Dataset092 = Dataset091 verbatim, but with the QC-approved Fanwei cases' labels REPLACED by the
promptable-refinement output (nnInteractive + SeqSeg), plus any QC-approved case that was previously
held out promoted into training.

Two effects, both optional/separable:
  * RELABEL  — a case already in D091 training keeps its image, gets the refined label instead of the
               coarse LCC pseudo-label. This is the label-quality effect.
  * ADD      — a case that was held out (rejected by the earlier QC) but is now usable enters training
               with its refined label. This is the extra-data effect. Disable with --no-new-cases to
               keep the case list identical to D091, giving a CLEAN A/B on label quality alone.

Symlink-only (never copies, never reads pixel data — see the sitk use-after-free lesson in
docs/CONVENTIONS.md). Uses os.path.abspath, NOT .resolve(), because nnUNet_raw is itself a symlink.
Hard-fails if any excluded case (Dataset080 test cases) would enter training.

Writes: imagesTr/labelsTr/imagesTs symlinks, dataset.json, split_meta.json (imagechd + pseudo_train,
consumed by the splits phase), manifest.csv recording the LABEL SOURCE per case, and build_report.json.
"""
import argparse, csv, json, os, shutil, sys
from pathlib import Path

FE = ".nii.gz"


def ap_(p):                      # absolute WITHOUT resolving symlinks
    return os.path.abspath(str(p))


def link(src, dst):
    if os.path.islink(dst) or os.path.exists(dst):
        os.remove(dst)
    os.symlink(ap_(src), dst)


def find_image(case, search_dirs):
    for d in search_dirs:
        c = Path(d) / (case + "_0000" + FE)
        if c.is_file():
            return c
    return None


def main():
    a = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a.add_argument("--nnunet-raw", required=True)
    a.add_argument("--src-dataset", default="Dataset091_ImageCHDPseudoCombinedV2")
    a.add_argument("--target-id", type=int, default=92)
    a.add_argument("--target-name", default="ImageCHDRefined")
    a.add_argument("--refined-labels-dir", required=True,
                   help="refined multi-label masks, <case>.nii.gz (e.g. chd_refinement/out/refined_fanwei_merged)")
    a.add_argument("--qc-approved", required=True,
                   help="comma-separated cases whose refined label passed QC. Already-in-train -> RELABEL; held-out -> ADD")
    a.add_argument("--exclude", default="", help="cases that must NEVER enter training (Dataset080 test)")
    a.add_argument("--image-dirs", default="", help="extra comma-separated dirs to search for ADDed cases' images")
    a.add_argument("--no-new-cases", action="store_true", help="RELABEL only; keep D091's exact case list")
    a.add_argument("--overwrite", action="store_true")
    args = a.parse_args()

    raw = Path(args.nnunet_raw)
    src = raw / args.src_dataset
    dst = raw / ("Dataset%03d_%s" % (args.target_id, args.target_name))
    ref = Path(args.refined_labels_dir)
    for p, what in ((src, "source dataset"), (ref, "refined labels dir")):
        if not p.is_dir():
            sys.exit("FATAL: %s not found: %s" % (what, p))

    qc = [c.strip() for c in args.qc_approved.split(",") if c.strip()]
    excl = {c.strip() for c in args.exclude.split(",") if c.strip()}
    bad = sorted(set(qc) & excl)
    if bad:
        sys.exit("FATAL: excluded case(s) present in --qc-approved: %s" % bad)

    img_dirs = [raw / "Dataset012_Fanweidata" / "imagesTr",
                raw / "Dataset071_ImageCHDClinicalOrientation" / "imagesTr",
                src / "imagesTs", src / "imagesTr"]
    img_dirs += [Path(d) for d in args.image_dirs.split(",") if d.strip()]

    if dst.exists():
        if not args.overwrite:
            sys.exit("FATAL: %s exists (use --overwrite)" % dst)
        shutil.rmtree(dst)
    for sub in ("imagesTr", "labelsTr", "imagesTs"):
        (dst / sub).mkdir(parents=True, exist_ok=True)

    src_tr = sorted(p.name[:-len(FE)] for p in (src / "labelsTr").glob("*" + FE))
    if not src_tr:
        sys.exit("FATAL: no labels in %s/labelsTr" % src)
    in_train = set(src_tr)
    relabel = [c for c in qc if c in in_train]
    add = [] if args.no_new_cases else [c for c in qc if c not in in_train]

    # 1) mirror D091 verbatim (image + label symlinks), swapping labels for RELABEL cases
    rows, n_ref, n_lcc = [], 0, 0
    for c in src_tr:
        im = src / "imagesTr" / (c + "_0000" + FE)
        if not im.is_file():
            sys.exit("FATAL: %s missing its image in %s" % (c, src / "imagesTr"))
        link(im, dst / "imagesTr" / (c + "_0000" + FE))
        if c in relabel:
            rl = ref / (c + FE)
            if not rl.is_file():
                sys.exit("FATAL: refined label missing for QC-approved case %s: %s" % (c, rl))
            link(rl, dst / "labelsTr" / (c + FE)); src_lbl = "refined"; n_ref += 1
        else:
            link(src / "labelsTr" / (c + FE), dst / "labelsTr" / (c + FE)); src_lbl = "lcc(D091)"; n_lcc += 1
        rows.append({"case": c, "role": "train", "origin": "D091", "label_source": src_lbl})

    # 2) ADD previously held-out, now-usable cases (image from source datasets, label = refined)
    added = []
    for c in add:
        if c in excl:
            sys.exit("FATAL: refusing to add excluded case %s" % c)
        rl = ref / (c + FE)
        if not rl.is_file():
            print("  [skip add %s] no refined label at %s" % (c, rl)); continue
        im = find_image(c, img_dirs)
        if im is None:
            print("  [skip add %s] no image found in any image dir" % c); continue
        link(im, dst / "imagesTr" / (c + "_0000" + FE))
        link(rl, dst / "labelsTr" / (c + FE))
        rows.append({"case": c, "role": "train", "origin": "ADDED(was held out)", "label_source": "refined"})
        added.append(c); n_ref += 1

    # 3) imagesTs = D091's held-out set MINUS anything we just promoted
    n_ts = 0
    for p in sorted((src / "imagesTs").glob("*" + FE)):
        case = p.name[:-len(FE)]
        base = case[:-5] if case.endswith("_0000") else case
        if base in added:
            continue
        link(p, dst / "imagesTs" / p.name); n_ts += 1
        rows.append({"case": base, "role": "test", "origin": "D091", "label_source": "-"})

    # 4) leak check
    train_cases = {p.name[:-len(FE)] for p in (dst / "labelsTr").glob("*" + FE)}
    leaked = sorted(train_cases & excl)
    if leaked:
        sys.exit("FATAL: excluded case(s) leaked into training: %s" % leaked)

    # 5) split_meta.json — extend D091's, with added cases counted as pseudo (train-only, never val)
    sm_src = src / "split_meta.json"
    if sm_src.is_file():
        sm = json.loads(sm_src.read_text())
    else:
        sm = {"imagechd": sorted(c for c in train_cases if not c.startswith(("CT_", "BAF", "CHIPS"))),
              "pseudo_train": sorted(c for c in train_cases if c.startswith(("CT_", "BAF", "CHIPS")))}
        print("  [warn] no split_meta.json in source — derived it by case-name prefix")
    sm["pseudo_train"] = sorted(set(sm.get("pseudo_train", [])) | set(added))
    sm["imagechd"] = sorted(set(sm.get("imagechd", [])) & train_cases)
    sm["relabelled_refined"] = sorted(relabel)
    sm["added_cases"] = sorted(added)
    (dst / "split_meta.json").write_text(json.dumps(sm, indent=1))
    if set(sm["imagechd"]) & set(sm["pseudo_train"]):
        sys.exit("FATAL: a case is both imagechd and pseudo_train")

    # 6) dataset.json (labels/channels copied from the source so the scheme cannot drift)
    dj = json.loads((src / "dataset.json").read_text())
    dj["numTraining"] = len(train_cases)
    (dst / "dataset.json").write_text(json.dumps(dj, indent=1))

    with open(dst / "manifest.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["case", "role", "origin", "label_source"])
        w.writeheader(); w.writerows(rows)
    report = {"target": dst.name, "source": src.name, "refined_labels_dir": str(ref),
              "numTraining": len(train_cases), "labels_refined": n_ref, "labels_lcc": n_lcc,
              "relabelled": sorted(relabel), "added": sorted(added), "imagesTs": n_ts,
              "no_new_cases": bool(args.no_new_cases)}
    (dst / "build_report.json").write_text(json.dumps(report, indent=2))

    print("[built] %s" % dst)
    print("  train=%d   labels: refined=%d  lcc(D091)=%d   imagesTs=%d" % (len(train_cases), n_ref, n_lcc, n_ts))
    print("  RELABELLED (%d): %s" % (len(relabel), ", ".join(sorted(relabel)) or "-"))
    print("  ADDED      (%d): %s" % (len(added), ", ".join(sorted(added)) or "-"))
    print("  ImageCHD(val pool)=%d  pseudo(train-only)=%d" % (len(sm["imagechd"]), len(sm["pseudo_train"])))


if __name__ == "__main__":
    main()
