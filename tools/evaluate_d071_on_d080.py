#!/usr/bin/env python3
"""Strict, paired D071/D090/D091 evaluation on the frozen eight-case D080 cohort.

Run through scripts/CHD_Dataset071_eval_dataset080.sh. `prepare` verifies and
fingerprints inputs before inference; `evaluate` requires those same files and
scores all three ensembles. No resampling, missing-case exclusion or low-score
filtering is performed. D100 contains these expert cases and is not evaluated.
Dependencies: numpy, nibabel (already used by the repository's Dice evaluator).
"""
import argparse
import csv
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import nibabel as nib
import numpy as np


CASES = ("BAF004", "CHIPS001", "CHIPS002", "CHIPS005", "CHIPS006", "CHIPS007", "CHIPS010", "CHIPS016")
LABELS = {"LV-BP": 1, "RV-BP": 2, "LA": 3, "RA": 4, "Myo": 5, "Ao": 6, "PA": 7}
METRICS = ("WH", *LABELS, "Macro7")
DATASETS = {"D071": "Dataset071_ImageCHDClinicalOrientation",
            "D090": "Dataset090_ImageCHDPseudoCombined",
            "D091": "Dataset091_ImageCHDPseudoCombinedV2"}
MODEL_NAME = "nnUNetTrainerDA5_200epochs__nnUNetResEncUNetMPlans__3d_fullres"


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def fingerprint(path):
    path = Path(path).absolute()
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f"Missing or empty file: {path}")
    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError(f"File changed while being read: {path}")
    return {"path": str(path), "bytes": after.st_size, "sha256": digest.hexdigest()}


def case_files(directory, images=False):
    directory = Path(directory)
    suffix = "_0000.nii.gz" if images else ".nii.gz"
    expected = {case + suffix for case in CASES}
    actual = {p.name for p in directory.glob("*.nii*")}
    if actual != expected:
        raise ValueError(f"Case set mismatch in {directory}: missing={sorted(expected - actual)}, "
                         f"unexpected={sorted(actual - expected)}")
    return {case: directory / (case + suffix) for case in CASES}


def load_volume(path, labels=False):
    img = nib.load(str(path))
    if len(img.shape) != 3 or not np.isfinite(img.affine).all() or abs(np.linalg.det(img.affine[:3, :3])) < 1e-12:
        raise ValueError(f"Invalid 3D geometry: {path}")
    if not labels:
        return img, None
    data = np.asanyarray(img.dataobj)
    if not np.isfinite(data).all() or not np.isin(data, np.arange(8)).all():
        raise ValueError(f"Expected integer labels 0..7: {path}")
    return img, data.astype(np.uint8)


def same_geometry(reference, other, path):
    if reference.shape != other.shape or not np.allclose(reference.affine, other.affine, rtol=0, atol=1e-4):
        raise ValueError(f"Geometry mismatch (shape/affine); will not resample: {path}")


def validate_cohort(images_dir, gt_dir, prediction_dirs):
    images = case_files(images_dir, images=True)
    truth = case_files(gt_dir)
    predictions = {name: case_files(folder) for name, folder in prediction_dirs.items()}
    for case in CASES:
        image, _ = load_volume(images[case])
        gt, _ = load_volume(truth[case], labels=True)
        same_geometry(image, gt, truth[case])
        for files in predictions.values():
            pred, _ = load_volume(files[case], labels=True)
            same_geometry(gt, pred, files[case])
    return [*images.values(), *truth.values(), *(p for files in predictions.values() for p in files.values())]


def metadata_files(folder, model, images=None):
    """Check model identity in saved prediction plans; folds/TTA need job records."""
    paths = [Path(folder) / "plans.json", Path(folder) / "dataset.json"]
    plans, dataset = [json.loads(p.read_text()) for p in paths]
    if plans.get("dataset_name") != DATASETS[model] or plans.get("plans_name") != "nnUNetResEncUNetMPlans":
        raise ValueError(f"Unexpected {model} saved plans in {folder}")
    values = list(dataset.get("labels", {}).values())
    if any(not isinstance(x, int) for x in values) or set(values) != set(range(8)):
        raise ValueError(f"Unexpected {model} label definitions in {folder}")
    if "3d_fullres" not in plans.get("configurations", {}):
        raise ValueError(f"Missing full-resolution configuration in {folder}")
    args_path = Path(folder) / "predict_from_raw_data_args.json"
    if images is not None:
        if Path(folder).name != f"{model}_ens":
            raise ValueError(f"Expected the audited {model}_ens ensemble folder, got {folder}")
        saved_args = json.loads(args_path.read_text())
        if saved_args.get("list_of_lists_or_source_folder") != str(images.absolute()):
            raise ValueError(f"Prediction metadata does not identify the expected native CT input: {folder}")
        if saved_args.get("num_parts") != 1 or saved_args.get("part_id") != 0:
            raise ValueError(f"Expected a single complete prediction job: {folder}")
        if saved_args.get("folder_with_segs_from_prev_stage") is not None:
            raise ValueError(f"Unexpected previous-stage input in {folder}")
        paths.append(args_path)
    elif args_path.is_file():
        paths.append(args_path)
    return paths


def configuration(args):
    return {name: str(getattr(args, name).absolute()) for name in ("images", "gt", "model", "d090", "d091", "out")}


def prepare(args):
    args.out.mkdir(parents=True, exist_ok=True)
    manifest = args.out / "run_manifest.json"
    if manifest.exists() or (args.out / "D071_ens").exists() or (args.out / "analysis").exists():
        raise ValueError("Use a fresh output directory; this run already contains outputs")
    if args.model.name != MODEL_NAME or args.model.parent.name != DATASETS["D071"]:
        raise ValueError(f"Expected D071 200-epoch ResEnc-M model, got {args.model}")
    files = metadata_files(args.model, "D071")
    files += [args.model / f"fold_{fold}" / "checkpoint_final.pth" for fold in range(5)]
    for name, directory in (("D090", args.d090), ("D091", args.d091)):
        files += metadata_files(directory, name, args.images)
    files += validate_cohort(args.images, args.gt, {"D090": args.d090, "D091": args.d091})
    repo = Path(__file__).absolute().parents[1]
    files += [Path(__file__).absolute(), repo / "scripts/CHD_Dataset071_eval_dataset080.sh",
              repo / "tools/plot_d080_three_model_violins.py",
              repo / "scripts/CHD_predict_dataset080_d092.sh"]
    try:
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
        status = subprocess.check_output(["git", "status", "--short"], cwd=repo, text=True)
    except (OSError, subprocess.CalledProcessError):
        revision, status = "unavailable", "unavailable"
    print("[preflight] All eight cases align. Fingerprinting images, GT, existing masks and D071 weights...", flush=True)
    record = {"created_utc": datetime.now(timezone.utc).isoformat(), "configuration": configuration(args),
              "cases": list(CASES), "git_revision": revision, "git_status": status,
              "protocol": {"trainer": "nnUNetTrainerDA5_200epochs", "plans": "nnUNetResEncUNetMPlans",
                           "configuration": "3d_fullres", "folds": [0, 1, 2, 3, 4],
                           "checkpoint": "checkpoint_final.pth", "test_time_mirroring": False,
                           "sliding_window_step": 0.5, "external_resizing": False,
                           "postprocessing": "none", "input": "native Dataset080 CT",
                           "plans_note": "Each trained model retains its own patch and normalization settings."},
              "comparator_provenance": "Existing D090/D091 ensemble folders from the September evaluation. "
                  "Saved JSON verifies dataset/plans, but does not independently record folds/checkpoint/TTA. "
                  "Those settings are supported by CHD_predict_dataset080_d092.sh and the job records.",
              "inputs": [fingerprint(p) for p in files]}
    write_json(manifest, record)
    print(f"[preflight] Passed; manifest: {manifest}", flush=True)


def dice(pred, gt):
    total = int(np.count_nonzero(pred)) + int(np.count_nonzero(gt))
    return 1.0 if total == 0 else 2.0 * int(np.count_nonzero(pred & gt)) / total


def score(pred, gt):
    result = {"WH": dice(pred > 0, gt > 0)}
    result.update({name: dice(pred == value, gt == value) for name, value in LABELS.items()})
    result["Macro7"] = float(np.mean([result[name] for name in LABELS]))
    return result


def write_csv(path, rows, columns):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def evaluate(args):
    record = json.loads((args.out / "run_manifest.json").read_text())
    if record["configuration"] != configuration(args):
        raise ValueError("Evaluation paths differ from the prepared run")
    analysis = args.out / "analysis"
    if analysis.exists():
        raise ValueError(f"Refusing to overwrite existing analysis: {analysis}")
    print("[evaluate] Verifying original inputs and weights are unchanged...", flush=True)
    for original in record["inputs"]:
        if fingerprint(original["path"]) != original:
            raise ValueError(f"Input changed since preflight: {original['path']}")
    folders = {"D071": args.out / "D071_ens", "D090": args.d090, "D091": args.d091}
    metadata = metadata_files(folders["D071"], "D071", args.images)
    validate_cohort(args.images, args.gt, folders)
    new_files = list(case_files(folders["D071"]).values()) + metadata
    new_fingerprints = [fingerprint(p) for p in new_files]
    rows = {name: [] for name in folders}
    empty_truth = []
    for case in CASES:
        _, gt = load_volume(args.gt / f"{case}.nii.gz", labels=True)
        empty_truth.extend({"Patient_ID": case, "Class": name} for name, value in LABELS.items() if not np.any(gt == value))
        for name, folder in folders.items():
            _, pred = load_volume(folder / f"{case}.nii.gz", labels=True)
            rows[name].append({"Patient_ID": case, **score(pred, gt)})
    summary = [{"Model": name, "N": len(CASES),
                **{f"{metric}_{stat}": float(getattr(np, stat)([r[metric] for r in model_rows]))
                   for metric in METRICS for stat in ("mean", "median")}}
               for name, model_rows in rows.items()]
    paired = []
    for baseline, adapted in (("D071", "D090"), ("D071", "D091"), ("D090", "D091")):
        for a, b in zip(rows[baseline], rows[adapted]):
            paired.append({"Comparison": f"{adapted}-{baseline}", "Patient_ID": a["Patient_ID"],
                           **{metric: b[metric] - a[metric] for metric in METRICS}})
    # All validation and scoring succeeds before any report files are published.
    analysis.mkdir()
    for name, model_rows in rows.items():
        write_csv(analysis / f"dice_{name}.csv", model_rows, ("Patient_ID", *METRICS))
    write_csv(analysis / "summary.csv", summary, list(summary[0]))
    write_csv(analysis / "paired_deltas.csv", paired, ("Comparison", "Patient_ID", *METRICS))
    write_json(analysis / "evaluation_provenance.json", {
        "completed_utc": datetime.now(timezone.utc).isoformat(), "prediction_files": new_fingerprints,
        "labels": LABELS, "n_patients": len(CASES), "empty_ground_truth_labels": empty_truth,
        "scoring": "Whole heart = union labels 1..7; Macro7 = unweighted per-patient mean across all seven labels. "
                   "Both masks empty: Dice 1; only one empty: Dice 0. No cases/classes/scores excluded.",
        "geometry": "Matching 3D shape and affine (absolute tolerance 1e-4, relative tolerance zero). No resampling.",
        "interpretation": "Patient is the unit of observation (n=8). D100 is excluded because it trained on D080. "
                          "The D071 baseline quantifies native-input generalization, not the resized pseudo-label route."})
    (analysis / "README.txt").write_text(
        "Matched D071 / D090 / D091 comparison on eight Dataset080 cases\n\n"
        "dice_D*.csv: one row per patient; WH = foreground union; Macro7 = mean of seven structures.\n"
        "summary.csv: unrounded across-patient means and medians.\n"
        "paired_deltas.csv: within-patient score differences (adapted minus baseline).\n"
        "All scores are included, including absent-GT classes and low/zero Dice. Inspect absent labels in\n"
        "evaluation_provenance.json, especially CHIPS002 RV. Eight patients are not forty independent folds.\n"
        "D071 uses native CT without external resizing/LCC; each model keeps its own saved plans.\n"
        "Existing D090/D091 masks are rescored without modification. Their historical protocol is described\n"
        "in ../run_manifest.json. No clinical usability or time saving is inferred from Dice alone.\n"
        "The parent COMPLETE marker appears only after evaluation and all three figure formats succeed.\n")
    for row in summary:
        print(f"{row['Model']}: n={row['N']}  mean WH={row['WH_mean']:.6f}  mean Macro7={row['Macro7_mean']:.6f}")
    print(f"[evaluate] CSVs and provenance: {analysis}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "evaluate"))
    for option in ("images", "gt", "model", "d090", "d091", "out"):
        parser.add_argument(f"--{option}", type=Path, required=True)
    args = parser.parse_args()
    try:
        (prepare if args.phase == "prepare" else evaluate)(args)
    except (ValueError, OSError, KeyError) as exc:
        parser.exit(1, f"FATAL: {exc}\n")


if __name__ == "__main__":
    main()
