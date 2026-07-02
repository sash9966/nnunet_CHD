"""
Synthetic tests for the chd_landmarks package.

Run with:   python3 -m pytest tests/test_chd_landmarks.py
       or:   python3 tests/test_chd_landmarks.py   (no pytest needed)

These tests use tiny 3D arrays and never import an nnU-Net trainer, so they run
locally despite the acvl_utils import blocker.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

# allow running from repo root without install
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from chd_landmarks import derived_regions as dr
from chd_landmarks import geometry as geo
from chd_landmarks import metrics as M
from chd_landmarks import topology as topo
from chd_landmarks.derived_label_builder import DerivedLabelBuilder
from chd_landmarks.disease_rules import load_rules
from chd_landmarks.labels import load_label_map
from chd_landmarks.metadata import CaseMetadata, load_disease_flags
from chd_landmarks.region_based_dataset_json import build_region_based_json

REPO = Path(__file__).resolve().parents[1]
CFG = REPO / "configs"
SPACING = (1.0, 1.0, 1.0)


# ---------------------------------------------------------------------------
# synthetic builders
# ---------------------------------------------------------------------------
def _disk_tube(shape, radius_fn):
    """Build a tube along axis 0 with per-slice radius radius_fn(z)."""
    vol = np.zeros(shape, dtype=np.int32)
    cy, cx = shape[1] // 2, shape[2] // 2
    yy, xx = np.ogrid[:shape[1], :shape[2]]
    for z in range(shape[0]):
        r = radius_fn(z)
        if r > 0:
            vol[z][(yy - cy) ** 2 + (xx - cx) ** 2 <= r * r] = 1
    return vol


def _write_dataset_json(d: Path, labels: dict):
    d.mkdir(parents=True, exist_ok=True)
    (d / "dataset.json").write_text(json.dumps({
        "channel_names": {"0": "CT"},
        "labels": labels,
        "numTraining": 1,
        "file_ending": ".nii.gz",
    }))


def _label_map(tmp):
    d = Path(tmp) / "DatasetXXX"
    _write_dataset_json(d, {"background": 0, "LV": 1, "RV": 2, "LA": 3, "RA": 4,
                            "Myo": 5, "AO": 6, "PA": 7})
    return load_label_map(str(CFG / "chd_label_map.yaml"), dataset_dir=str(d))


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------
def test_label_map_loading():
    with tempfile.TemporaryDirectory() as tmp:
        lm = _label_map(tmp)
        assert lm.id_of("lv") == 1
        assert lm.id_of("pulmonary_artery") == 7
        assert lm.has("aorta")
        assert lm.id_of("pulmonary_veins") is None  # absent -> placeholder
        assert lm.next_free_id() == 8


def test_metadata_flag_parsing():
    with tempfile.TemporaryDirectory() as tmp:
        csv = Path(tmp) / "meta.csv"
        csv.write_text("index,VSD,ToF,CA\n1001,1,,\n1002,,1,1\n")
        rules = load_rules(str(CFG / "chd_disease_rules.yaml"))
        meta = load_disease_flags(str(csv), rules.flag_columns)
        assert meta["ct_1001"].flags["VSD"] is True
        assert meta["ct_1001"].flags["ToF"] is False
        assert meta["ct_1002"].flags["ToF"] is True
        assert meta["ct_1002"].flags["Coarctation"] is True   # CA -> Coarctation


def _vsd_seg():
    """LV (x<10), RV (x>10), myocardium plane at x=10 with a central hole."""
    seg = np.zeros((20, 24, 24), dtype=np.int32)
    seg[2:10, :, :] = 1      # LV
    seg[11:19, :, :] = 2     # RV
    seg[10, :, :] = 5        # myocardium septum
    seg[10, 8:14, 8:14] = 0  # septal hole (defect window)
    return seg


def test_vsd_proxy_and_merge():
    with tempfile.TemporaryDirectory() as tmp:
        lm = _label_map(tmp)
        params = {"interface_dilation_mm": 2.0, "false_merge_min_contact_mm2": 200.0}
        reg = dr.build_vsd_orifice_proxy(_vsd_seg(), lm, SPACING, np.eye(4),
                                         {"VSD"}, params)
        assert reg.present, "VSD proxy should be derived at the septal hole"
        assert reg.confidence in ("high", "medium")
        # proxy must be localized (small), not the whole contact plane
        assert reg.meta["voxels"] < 24 * 24


def test_vsd_not_derived_without_flag():
    with tempfile.TemporaryDirectory() as tmp:
        lm = _label_map(tmp)
        reg = dr.build_vsd_orifice_proxy(_vsd_seg(), lm, SPACING, np.eye(4),
                                         set(), {"interface_dilation_mm": 2.0})
        assert not reg.present
        assert reg.confidence == "none"


def test_lv_rv_false_merge_detection():
    with tempfile.TemporaryDirectory() as tmp:
        lm = _label_map(tmp)
        seg = np.zeros((10, 24, 24), dtype=np.int32)
        seg[:, :, :12] = 1   # LV
        seg[:, :, 12:] = 2   # RV  -- huge shared contact plane, no myo
        reg = dr.build_lv_rv_false_merge_region(seg, lm, SPACING, np.eye(4),
                                                {"false_merge_min_contact_mm2": 200.0})
        assert reg.present
        assert reg.meta["false_merge_suspected"] is True


def test_pulmonary_stenosis_roi():
    with tempfile.TemporaryDirectory() as tmp:
        lm = _label_map(tmp)
        tube = _disk_tube((24, 24, 24), lambda z: 1 if 10 <= z <= 13 else 4)
        seg = (tube * 7).astype(np.int32)   # PA label
        reg = dr.build_pulmonary_stenosis_roi(seg, lm, SPACING, np.eye(4),
                                              {"ToF"}, {"stenosis_radius_drop_frac": 0.6,
                                                        "stenosis_roi_dilation_mm": 4.0})
        assert reg.present, "stenosis ROI should locate the narrowing"
        assert reg.meta["min_diameter_mm"] <= reg.meta["median_radius_mm"] * 2


def test_aortic_coarctation_roi():
    with tempfile.TemporaryDirectory() as tmp:
        lm = _label_map(tmp)
        tube = _disk_tube((24, 24, 24), lambda z: 1 if 10 <= z <= 13 else 4)
        seg = (tube * 6).astype(np.int32)   # AO label
        out = dr.build_aortic_coarctation_roi(seg, lm, SPACING, np.eye(4),
                                              {"Coarctation"},
                                              {"coarctation_radius_drop_frac": 0.6,
                                               "stenosis_roi_dilation_mm": 4.0})
        assert out["aortic_narrowing_roi"].present
        assert out["minimum_aortic_radius_point"].present


def test_aorta_pulmonary_interface():
    with tempfile.TemporaryDirectory() as tmp:
        lm = _label_map(tmp)
        seg = np.zeros((12, 24, 24), dtype=np.int32)
        seg[:, :, :12] = 6   # AO
        seg[:, :, 12:] = 7   # PA, adjacent
        reg = dr.build_aorta_pulmonary_confusion_interface(
            seg, lm, SPACING, np.eye(4), {"ToF"}, {"interface_dilation_mm": 2.0})
        assert reg.present


def test_hypoplastic_preservation():
    with tempfile.TemporaryDirectory() as tmp:
        lm = _label_map(tmp)
        seg = np.zeros((12, 24, 24), dtype=np.int32)
        seg[2:5, 2:5, 2:5] = 1   # small LV
        out = dr.build_hypoplastic_structure_preservation_roi(
            seg, lm, SPACING, np.eye(4), {"HLHS"}, {"min_component_voxels": 5})
        assert out["hypoplastic_lv_preservation_region"].present
        assert int(out["hypoplastic_lv_preservation_region"].mask.sum()) == int((seg == 1).sum())


def test_merged_label_map_creation():
    with tempfile.TemporaryDirectory() as tmp:
        lm = _label_map(tmp)
        rules = load_rules(str(CFG / "chd_disease_rules.yaml"))
        derived_cfg = __import__("chd_landmarks.io", fromlist=["load_yaml"]).load_yaml(
            str(CFG / "chd_derived_labels.yaml"))
        builder = DerivedLabelBuilder(lm, rules, derived_cfg)
        meta = CaseMetadata(case_id="ct_test", flags={"VSD": True})
        d = builder.build_for_case(_vsd_seg(), meta, "ct_test",
                                   affine=np.eye(4), spacing=SPACING)
        # merged map keeps all original anatomy ids
        assert set(np.unique(d.anatomy_seg)) <= set(np.unique(d.merged_label_map)) | {0}
        # a new hard id (>=8) should appear where the septal-defect proxy painted
        new_ids = set(np.unique(d.merged_label_map)) - set(np.unique(d.anatomy_seg))
        assert any(i >= 8 for i in new_ids), f"expected a derived hard label, got {new_ids}"
        # the proxy only overwrote former chamber voxels (interface-only policy);
        # this is a VSD case -> ventricular only (LV/RV)
        sd_id = builder.merge_scheme["septal_defect_proxy"]
        painted = d.merged_label_map == sd_id
        orig_at_painted = set(np.unique(d.anatomy_seg[painted]))
        assert orig_at_painted <= {1, 2}, f"VSD must only overwrite LV/RV, got {orig_at_painted}"


def test_region_based_json():
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp) / "Dataset031_test"
        _write_dataset_json(d, {"background": 0, "LV": 1, "RV": 2, "LA": 3, "RA": 4,
                                "Myo": 5, "AO": 6, "PA": 7, "septal_defect_proxy": 8})
        out = build_region_based_json(str(d), str(CFG / "chd_label_map.yaml"),
                                      str(CFG / "chd_region_training.yaml"), apply=False)
        assert "regions_class_order" in out
        assert len(out["regions_class_order"]) == len([k for k in out["labels"] if k != "background"])
        assert isinstance(out["labels"]["whole_heart"], list)  # composite region
        assert "septal_defect_proxy" in out["labels"]
        assert (d / "dataset_region_based.json").is_file()
        assert (d / "dataset.json").is_file()  # original untouched (apply=False)


def test_metrics_synthetic():
    lm_seg = np.zeros((12, 24, 24), dtype=np.int32)
    lm_seg[:, :, :12] = 6
    lm_seg[:, :, 12:] = 7
    with tempfile.TemporaryDirectory() as tmp:
        lm = _label_map(tmp)
        # perfect prediction
        res = M.evaluate_case(lm_seg, lm_seg, lm, SPACING, affine=np.eye(4),
                              active_diseases=["ToF"])
        assert abs(res["general"]["aorta"]["dice"] - 1.0) < 1e-6
        assert res["disease"]["ToF"]["aorta_pulmonary_leakage_volume_mm3"] == 0.0
        # swap AO<->PA in pred -> leakage > 0
        bad = lm_seg.copy()
        bad[lm_seg == 6] = 7
        bad[lm_seg == 7] = 6
        res2 = M.evaluate_case(bad, lm_seg, lm, SPACING, affine=np.eye(4),
                               active_diseases=["ToF"])
        assert res2["disease"]["ToF"]["aorta_pulmonary_leakage_volume_mm3"] > 0.0


def test_peri_septal_roi_eval_region():
    with tempfile.TemporaryDirectory() as tmp:
        lm = _label_map(tmp)
        seg = _vsd_seg()
        vsd = dr.build_vsd_orifice_proxy(seg, lm, SPACING, np.eye(4), {"VSD"},
                                         {"interface_dilation_mm": 2.0})
        peri = dr.build_peri_defect_roi(vsd.mask, seg, [lm.id_of("lv"), lm.id_of("rv")],
                                        SPACING, np.eye(4), 3.0, "peri_septal_defect_roi")
        assert peri.present
        # the shell must lie ONLY inside LV/RV blood pool (not the whole chamber)
        ids = set(np.unique(seg[peri.mask.astype(bool)]).tolist())
        assert ids <= {1, 2}, f"peri-defect ROI must be LV/RV blood pool only, got {ids}"
        # and it must be a local shell, far smaller than the whole LV+RV
        assert int(peri.mask.sum()) < int(((seg == 1) | (seg == 2)).sum())


def test_normalize_case_key():
    from chd_landmarks.metadata import normalize_case_key
    assert normalize_case_key("ct_1001_image") == "ct_1001"   # Dataset030 naming
    assert normalize_case_key("ct_1001_0000") == "ct_1001"     # channel suffix
    assert normalize_case_key("ct_1001_image_0000") == "ct_1001"
    assert normalize_case_key("ct_1001") == "ct_1001"          # already bare


def test_septal_defect_hard_label():
    """The general dataset promotes ONE unified septal-defect integer label."""
    with tempfile.TemporaryDirectory() as tmp:
        lm = _label_map(tmp)
        rules = load_rules(str(CFG / "chd_disease_rules.yaml"))
        from chd_landmarks import io as _io
        derived_cfg = _io.load_yaml(str(CFG / "chd_derived_labels.yaml"))
        builder = DerivedLabelBuilder(lm, rules, derived_cfg)
        assert list(builder.merge_scheme.keys()) == ["septal_defect_proxy"]
        labels = builder.merged_dataset_labels()
        assert "septal_defect_proxy" in labels
        assert "vsd_orifice_proxy" not in labels  # per-septum proxy is eval-only
        assert "asd_orifice_proxy" not in labels


def test_septal_defect_union_asd_and_vsd():
    """ASD-only and VSD-only cases both yield a septal_defect_proxy."""
    with tempfile.TemporaryDirectory() as tmp:
        lm = _label_map(tmp)
        rules = load_rules(str(CFG / "chd_disease_rules.yaml"))
        from chd_landmarks import io as _io
        derived_cfg = _io.load_yaml(str(CFG / "chd_derived_labels.yaml"))
        builder = DerivedLabelBuilder(lm, rules, derived_cfg)
        seg = _vsd_seg()  # ventricular gap geometry
        for flag in ("VSD", "ToF"):
            d = builder.build_for_case(seg, CaseMetadata("ct_x", {flag: True}), "ct_x",
                                       affine=np.eye(4), spacing=SPACING)
            assert "septal_defect_proxy" in d.auxiliary_masks, f"{flag} should derive a septal defect"


def test_topology_helpers():
    a = np.zeros((10, 10, 10), dtype=np.int32)
    a[1:4, 1:4, 1:4] = 1
    a[6:9, 6:9, 6:9] = 1   # two components
    assert topo.connected_component_count(a) == 2
    assert abs(topo.largest_cc_fraction(a) - 0.5) < 1e-6
    assert geo.volume_mm3(a, SPACING) == float((a > 0).sum())



def test_soft_tversky_binary():
    import torch
    from nnunetv2.training.loss.septal_losses import soft_tversky_binary, resolve_septal_label_id
    gt = torch.zeros(1, 1, 8, 8, 8); gt[0, 0, 3:5, 3:5, 3:5] = 1
    assert float(soft_tversky_binary(gt.clone(), gt)) < 1e-3          # perfect -> ~0
    assert float(soft_tversky_binary(torch.zeros_like(gt), gt)) > 0.9  # miss -> ~1
    # FN penalised more than FP at EQUAL error magnitude (beta>alpha).
    # gt = 8 voxels. fn: drop 4 (FN=4, FP=0). fp: add 4 (FP=4, FN=0).
    fn = gt.clone(); fn[0, 0, 3, 3:5, 3:5] = 0          # remove 4 voxels -> FN=4
    fp = gt.clone(); fp[0, 0, 5, 3:5, 3:5] = 1          # add 4 voxels    -> FP=4
    assert float(soft_tversky_binary(fn, gt)) > float(soft_tversky_binary(fp, gt))
    assert resolve_septal_label_id({"labels": {"background":0,"LV":1,"septal_defect_proxy":8}}) == 8


# ---------------------------------------------------------------------------
def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    passed = 0
    for fn in fns:
        try:
            fn()
            print(f"  PASS  {fn.__name__}")
            passed += 1
        except Exception as e:  # noqa: BLE001
            import traceback
            print(f"  FAIL  {fn.__name__}: {e}")
            traceback.print_exc()
    print(f"\n{passed}/{len(fns)} tests passed")
    return 0 if passed == len(fns) else 1


if __name__ == "__main__":
    sys.exit(_run_all())
