"""Small synthetic NIfTI tests; no trained models or patient data required.

python -m unittest discover -s tests -p 'test_evaluate_d071_on_d080.py'
"""
import argparse
import csv
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import nibabel as nib
import numpy as np


MODULE = Path(__file__).absolute().parents[1] / "tools/evaluate_d071_on_d080.py"
SPEC = importlib.util.spec_from_file_location("evaluation", MODULE)
evaluation = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(evaluation)


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.gt = np.arange(8, dtype=np.uint8).reshape(2, 2, 2)
        self.args = argparse.Namespace(
            images=self.root / "imagesTr", gt=self.root / "labelsTr",
            model=self.root / evaluation.DATASETS["D071"] / evaluation.MODEL_NAME,
            d090=self.root / "D090_ens", d091=self.root / "D091_ens", out=self.root / "run")
        for folder in (self.args.images, self.args.gt, self.args.d090, self.args.d091):
            folder.mkdir(parents=True)
        for case in evaluation.CASES:
            self.save(self.args.images / f"{case}_0000.nii.gz", self.gt.astype(np.float32))
            for folder in (self.args.gt, self.args.d090, self.args.d091):
                self.save(folder / f"{case}.nii.gz", self.gt)
        for model, folder in (("D071", self.args.model), ("D090", self.args.d090), ("D091", self.args.d091)):
            self.metadata(folder, model)
        for fold in range(5):
            folder = self.args.model / f"fold_{fold}"
            folder.mkdir()
            (folder / "checkpoint_final.pth").write_bytes(b"synthetic-checkpoint")

    def save(self, path, data, affine=None):
        nib.save(nib.Nifti1Image(data, np.eye(4) if affine is None else affine), path)

    def metadata(self, folder, model):
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "plans.json").write_text(json.dumps({
            "dataset_name": evaluation.DATASETS[model], "plans_name": "nnUNetResEncUNetMPlans",
            "configurations": {"3d_fullres": {}}}))
        (folder / "dataset.json").write_text(json.dumps({"labels": {"background": 0, **evaluation.LABELS}}))
        (folder / "predict_from_raw_data_args.json").write_text(json.dumps({
            "list_of_lists_or_source_folder": str(self.args.images.absolute()), "num_parts": 1,
            "part_id": 0, "folder_with_segs_from_prev_stage": None}))

    def add_baseline(self):
        folder = self.args.out / "D071_ens"
        self.metadata(folder, "D071")
        for case in evaluation.CASES:
            self.save(folder / f"{case}.nii.gz", np.zeros_like(self.gt))

    def test_empty_mask_policy_and_partial_overlap(self):
        empty = np.zeros(4, dtype=bool)
        self.assertEqual(evaluation.dice(empty, empty), 1)
        self.assertEqual(evaluation.dice(~empty, empty), 0)
        self.assertAlmostEqual(evaluation.dice(np.array([1, 1, 0, 0], bool), np.array([1, 0, 1, 0], bool)), 0.5)

    def test_complete_run_scores_all_eight_without_filtering(self):
        evaluation.prepare(self.args)
        self.add_baseline()
        evaluation.evaluate(self.args)
        with (self.args.out / "analysis/dice_D071.csv").open() as stream:
            baseline = list(csv.DictReader(stream))
        self.assertEqual(len(baseline), 8)
        self.assertTrue(all(float(row[metric]) == 0 for row in baseline for metric in evaluation.METRICS))
        with (self.args.out / "analysis/summary.csv").open() as stream:
            summary = list(csv.DictReader(stream))
        self.assertEqual([float(row["WH_mean"]) for row in summary], [0, 1, 1])
        self.assertEqual([float(row["Macro7_mean"]) for row in summary], [0, 1, 1])

    def test_missing_case_is_fatal(self):
        (self.args.d090 / "CHIPS002.nii.gz").unlink()
        with self.assertRaisesRegex(ValueError, "Case set mismatch"):
            evaluation.prepare(self.args)

    def test_affine_mismatch_is_fatal_even_when_shapes_match(self):
        affine = np.eye(4)
        affine[0, 3] = 10
        self.save(self.args.d091 / "CHIPS002.nii.gz", self.gt, affine)
        with self.assertRaisesRegex(ValueError, "Geometry mismatch"):
            evaluation.prepare(self.args)

    def test_fractional_and_unknown_labels_are_fatal(self):
        path = self.args.d090 / "CHIPS002.nii.gz"
        for bad in (0.5, 8, float("nan")):
            data = self.gt.astype(np.float32)
            data[0, 0, 0] = bad
            self.save(path, data)
            with self.assertRaisesRegex(ValueError, "integer labels"):
                evaluation.prepare(self.args)

    def test_changed_ground_truth_between_phases_is_fatal(self):
        evaluation.prepare(self.args)
        self.add_baseline()
        self.save(self.args.gt / "CHIPS002.nii.gz", np.zeros_like(self.gt))
        with self.assertRaisesRegex(ValueError, "Input changed"):
            evaluation.evaluate(self.args)
        self.assertFalse((self.args.out / "analysis").exists())

    def test_incomplete_baseline_is_fatal(self):
        evaluation.prepare(self.args)
        self.add_baseline()
        (self.args.out / "D071_ens/CHIPS002.nii.gz").unlink()
        with self.assertRaisesRegex(ValueError, "Case set mismatch"):
            evaluation.evaluate(self.args)
        self.assertFalse((self.args.out / "analysis").exists())

    def test_wrong_model_metadata_is_fatal(self):
        plans = self.args.d090 / "plans.json"
        data = json.loads(plans.read_text())
        data["dataset_name"] = "Dataset030_imageCHD_HU"
        plans.write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError, "Unexpected D090 saved plans"):
            evaluation.prepare(self.args)

    def test_non_native_prediction_inputs_are_fatal(self):
        path = self.args.d090 / "predict_from_raw_data_args.json"
        data = json.loads(path.read_text())
        data["list_of_lists_or_source_folder"] = "/wrong/resized_inputs"
        path.write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError, "expected native CT input"):
            evaluation.prepare(self.args)


if __name__ == "__main__":
    unittest.main()
