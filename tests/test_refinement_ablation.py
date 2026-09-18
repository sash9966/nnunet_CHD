"""Regression tests for the observed failure mechanisms; no model downloads/GPU."""
import ast
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import nibabel as nib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools'))
from refinement_common import ARMS, chamber_prompts, compose, directed_seeds, load_label, save_label, write_json
from seqseg_bounded import bound_next, instrument, tag_step
from build_refinement_ablation import build, fixed_plans, validate_splits, validate_training
from run_refinement_ablation import assemble, complete, evaluate, init, is_complete, read_run


class RefinementRegressionTests(unittest.TestCase):
    def seed(self):
        a = np.zeros((12, 12, 12), np.uint8)
        a[1:4, 1:4, 1:4] = 1; a[6:9, 1:4, 1:4] = 2
        a[1:4, 6:9, 1:4] = 5; a[6:9, 6:9, 1:4] = 6
        return a

    def test_baseline_exact(self):
        seed = self.seed(); out, _, _ = compose(seed, {}, (1, 1, 1))
        np.testing.assert_array_equal(out, seed)

    def test_fallback_restores_all_seed_voxels(self):
        seed = self.seed(); chamber = (seed == 1) | (seed == 5)
        out, qc, _ = compose(seed, {1: chamber, 5: np.zeros_like(seed)}, (1, 1, 1))
        np.testing.assert_array_equal(out == 5, seed == 5)
        self.assertIn('empty', qc['structures']['5']['fallback'])

    def test_non_target_labels_cannot_be_eaten(self):
        seed = self.seed(); proposal = (seed > 0)
        out, _, _ = compose(seed, {1: proposal}, (1, 1, 1))
        for sid in (2, 5, 6):
            np.testing.assert_array_equal(out == sid, seed == sid)

    def test_overlap_order_independent_abstention(self):
        seed = self.seed(); a = seed == 1; b = seed == 2
        a[4:7, 2, 2] = True; b[3:7, 2, 2] = True
        x, _, conflicts = compose(seed, {1: a, 2: b}, (1, 1, 1), cleanup=False)
        y, _, _ = compose(seed, {2: b, 1: a}, (1, 1, 1), cleanup=False)
        np.testing.assert_array_equal(x, y)
        self.assertTrue(conflicts[4, 2, 2]); self.assertEqual(x[4, 2, 2], 0)

    def test_shrinkage_rejected(self):
        seed = self.seed(); tiny = np.zeros_like(seed); tiny[2, 2, 2] = 1
        out, qc, _ = compose(seed, {1: tiny}, (1, 1, 1))
        np.testing.assert_array_equal(out, seed)
        self.assertIn('seed_loss', qc['structures']['1']['fallback'])

    def test_vessel_union_preserves_seed_and_adds(self):
        seed = self.seed(); vessel = seed == 6; vessel[8:11, 7, 2] = True
        out, _, _ = compose(seed, {6: vessel}, (1, 1, 1))
        self.assertTrue(np.all(out[seed == 6] == 6)); self.assertEqual(out[10, 7, 2], 6)
        np.testing.assert_array_equal(out[seed == 5], seed[seed == 5])

    def test_empty_vessel_union_is_unchanged_not_successful_growth(self):
        seed = self.seed()
        out, qc, _ = compose(seed, {6: seed == 6}, (1, 1, 1))
        np.testing.assert_array_equal(out, seed)
        self.assertEqual(qc['structures']['6']['outcome'], 'unchanged_seed')
        self.assertEqual(qc['structures']['6']['added_voxels'], 0)

    def test_excessive_vessel_growth_restores_only_that_vessel(self):
        seed = self.seed()
        vessel = (seed == 6) | (seed == 0)
        out, qc, _ = compose(seed, {6: vessel}, (1, 1, 1))
        np.testing.assert_array_equal(out, seed)
        self.assertIn('growth', qc['structures']['6']['fallback'])
        self.assertEqual(qc['structures']['6']['outcome'], 'fallback_to_seed')

    def test_seed_connected_cleanup_rejects_island(self):
        seed = self.seed(); a = seed == 1; a[10, 10, 10] = True
        out, _, _ = compose(seed, {1: a}, (1, 1, 1))
        self.assertEqual(out[10, 10, 10], 0)

    def test_negative_opponents_and_lasso_geometry(self):
        seed = self.seed(); lassos, negatives = chamber_prompts(seed, 1, (1, 1, 1), 1.)
        self.assertEqual({p['opponent'] for p in negatives}, {2, 5, 6})
        for p in negatives:
            self.assertEqual(int(seed[tuple(p['point'])]), p['opponent'])
        for crop, bbox in lassos:
            self.assertEqual(crop.shape, tuple(b-a for a, b in bbox))
            self.assertTrue(np.all(seed[tuple(slice(a, b) for a, b in bbox)][crop > 0] == 1))

    def test_inset_seeds_have_opposite_local_directions(self):
        a = np.zeros((30, 11, 11), bool); a[2:28, 3:8, 3:8] = True
        affine = np.diag([.5, 1., 1., 1.])
        rows, details = directed_seeds(a, affine, (.5, 1., 1.), 3.)
        vectors = [np.array(r[1])-np.array(r[0]) for r in rows]
        self.assertLess(float(np.dot(vectors[0], vectors[1])), 0)
        for r, d in zip(rows, details):
            self.assertTrue(a[tuple(d['current_voxel'])]); self.assertGreater(r[2], 1.)
            expected = nib.affines.apply_affine(affine, d['current_voxel'])*[-1, -1, 1]
            np.testing.assert_allclose(r[1], expected)

    def test_seed_walks_inward_past_thin_spur_without_radius_floor(self):
        a = np.zeros((50, 15, 15), bool)
        a[2:48, 7, 7] = True; a[15:35, 3:12, 3:12] = True
        path = [[x, 7, 7] for x in range(2, 48)]
        with patch('label_to_prompts.centerline', return_value=(path, [path[0], path[-1]])):
            rows, details = directed_seeds(a, np.eye(4), (1., 1., 1.))
        for row, detail in zip(rows, details):
            self.assertEqual(detail['initial_radius_mm'], 1.)
            self.assertTrue(detail['core_target_met'])
            self.assertGreater(detail['radius_mm'], 1.)
            self.assertGreater(detail['inset_mm'], 5.)
            self.assertTrue(a[tuple(detail['current_voxel'])])

    def test_affine_mismatch_fails_even_if_shape_matches(self):
        with tempfile.TemporaryDirectory() as d:
            a = self.seed(); ref = nib.Nifti1Image(a, np.eye(4)); affine = np.eye(4); affine[0, 3] = 10
            p = Path(d)/'mask.nii.gz'; nib.save(nib.Nifti1Image(a, affine), p)
            with self.assertRaises(ValueError):
                load_label(p, ref)

    def test_completion_checks_bytes(self):
        with tempfile.TemporaryDirectory() as d:
            folder = Path(d); p = folder/'output'; p.write_bytes(b'good')
            complete(folder, [p]); self.assertTrue(is_complete(folder)); p.write_bytes(b'bad')
            with self.assertRaises(ValueError):
                is_complete(folder)

    def test_plans_keep_spacing_architecture_normalization(self):
        p = {'dataset_name': 'old', 'plans_name': 'old', 'foreground_intensity_properties_per_channel': {'0': {'mean': 100}},
             'configurations': {'3d_fullres': {'spacing': [1, 1, 1], 'patch_size': [64, 192, 192], 'data_identifier': 'old', 'architecture': {'layers': 6}}}}
        original = copy.deepcopy(p); a = fixed_plans(p, 'Dataset194_Test')
        self.assertEqual(p, original)
        self.assertEqual(a['foreground_intensity_properties_per_channel'], p['foreground_intensity_properties_per_channel'])
        for key in ('spacing', 'patch_size', 'architecture'):
            self.assertEqual(a['configurations']['3d_fullres'][key], p['configurations']['3d_fullres'][key])

    def test_split_leakage_rejected(self):
        splits = [{'train': ['pseudo'], 'val': ['expert']}]*5
        validate_splits(splits, {'pseudo', 'expert'}, {'pseudo'})
        with self.assertRaises(ValueError):
            validate_splits(splits, {'pseudo', 'expert'}, {'expert'})
        with self.assertRaises(ValueError):
            validate_splits([{'train': ['CHIPS016'], 'val': ['expert']}]*5, {'CHIPS016', 'expert'}, {'CHIPS016'})

    def test_saved_candidates_to_four_training_datasets_and_scores(self):
        """Exercise real NIfTI/manifest/CSV paths, excluding only model inference."""
        with tempfile.TemporaryDirectory() as d:
            base = Path(d); root = base/'run'; raw = base/'raw'; pre = base/'pre'
            source_name = 'Dataset090_Test'; source = raw/source_name
            (source/'imagesTr').mkdir(parents=True); (source/'labelsTr').mkdir()
            seed = self.seed(); ref = nib.Nifti1Image(seed, np.eye(4))
            for case in ('pseudo', 'expert'):
                save_label(source/'imagesTr'/(case+'_0000.nii.gz'), seed, ref)
                save_label(source/'labelsTr'/(case+'.nii.gz'), seed, ref)
            selection = source/'split_config.csv'
            selection.write_text('case_id,intended_use\npseudo,pseudo_label_train\nexpert,validation\n')
            init(SimpleNamespace(run=str(root), accepted_csv=str(selection), case_report=None,
                 all_cases=False, cases=None, expected_cases=1, images_dir=str(source/'imagesTr'),
                 seeds_dir=str(source/'labelsTr'), min_retention=.7, max_growth=4., vessel_max_growth=20.,
                 no_cleanup=False, prompt_margin_mm=1., seed_inset_mm=5., vessel_mode='union'))
            cfg = read_run(root)
            folder = root/'nni'/'pseudo'; files = []
            for sid in range(1, 5):
                p = folder/('%d.nii.gz' % sid); save_label(p, seed == sid, ref); files.append(p)
            complete(folder, files)
            for sid in (6, 7):
                folder = root/'seqseg'/'pseudo'/str(sid); candidate = seed == sid
                if sid == 6:
                    candidate[8:11, 7, 2] = True
                save_label(folder/'candidate.nii.gz', candidate, ref)
                write_json(folder/'status.json', {'status': 'completed' if sid == 6 else 'absent_seed'})
                complete(folder, [folder/'candidate.nii.gz', folder/'status.json'])
            assemble(SimpleNamespace(arms=ARMS), root, cfg)
            for arm in ARMS:
                self.assertTrue(is_complete(root/'arms'/arm))
                _, label = load_label(root/'arms'/arm/'pseudo.nii.gz', ref)
                np.testing.assert_array_equal(label == 5, seed == 5)
                self.assertEqual(label[10, 7, 2], 6 if arm in ('seqseg', 'combined') else 0)
            evaluate(SimpleNamespace(arms=ARMS, gt_dir=str(source/'labelsTr')), root, cfg)
            self.assertEqual(len((root/'scores.csv').read_text().splitlines()), 33)
            write_json(source/'dataset.json', {'file_ending': '.nii.gz', 'channel_names': {'0': 'CT'},
                       'labels': {str(i): i for i in range(8)}, 'numTraining': 2})
            write_json(pre/source_name/'splits_final.json', [{'train': ['pseudo'], 'val': ['expert']}]*5)
            plans = pre/source_name/'reference.json'
            write_json(plans, {'dataset_name': source_name, 'plans_name': 'original',
                       'configurations': {'3d_fullres': {'spacing': [1., 1., 1.], 'data_identifier': 'original'}}})
            build(SimpleNamespace(raw=str(raw), preprocessed=str(pre), source_dataset=source_name,
                  reference_plans=str(plans), results=str(base/'results'), dataset_ids=[194, 195, 196, 197]), root)
            for arm in ARMS:
                study, entry = validate_training(root, arm)
                self.assertEqual(set(entry['labels']), {'pseudo', 'expert'})
                self.assertEqual(study['splits'][0]['val'], ['expert'])
            save_label(source/'labelsTr'/'pseudo.nii.gz', np.zeros_like(seed), ref)
            with self.assertRaisesRegex(ValueError, 'Input changed'):
                read_run(root)


class DepthTests(unittest.TestCase):
    def test_both_daughters_inherit_depth_and_third_split_stops(self):
        points = np.array([[1., 0, 0], [0, 1., 0]]); radii = np.ones(2); angles = np.zeros(2)
        events = []; step = {'point': np.zeros(3), 'chances': 0}
        for expected in (1, 2):
            p, r, a, depth = bound_next(points, radii, angles, step, 2, 3, events)
            self.assertEqual(depth, expected)
            main = tag_step({'point': p[0], 'chances': 0}, depth)
            side = tag_step({'point': p[1], 'chances': 0}, depth)
            self.assertEqual(main['_refinement_depth'], side['_refinement_depth'])
            step = main
        p, r, a, depth = bound_next(points, radii, angles, step, 2, 3, events)
        self.assertEqual(p.size, 0); self.assertEqual(step['chances'], 3)
        self.assertTrue(events[-1]['stopped'])

    def test_single_continuation_at_depth_two_is_allowed(self):
        step = {'point': np.zeros(3), '_refinement_depth': 2}
        p, _, _, depth = bound_next(np.ones((1, 3)), np.ones(1), np.ones(1), step, 2, 3, [])
        self.assertEqual(len(p), 1); self.assertEqual(depth, 2)

    def test_adapter_fails_closed_on_unknown_tracer(self):
        with self.assertRaises(RuntimeError):
            instrument('def trace_centerline():\n    return None\n')

    def test_adapter_tags_main_and_side_in_executable_fixture(self):
        source = '''def trace_centerline(step_seg, number_chances):
    point_tree, radius_tree, angle_change = get_next_points()
    if point_tree.size == 0:
        return []
    main = create_step_dict(point_tree[0])
    side = create_step_dict(point_tree[1])
    return [main, side]
'''
        ns = {'get_next_points': lambda: (np.ones((2, 3)), np.ones(2), np.ones(2)),
              'create_step_dict': lambda p: {'point': p}, '_refinement_bound_next': bound_next,
              '_refinement_tag_step': tag_step, '_refinement_max_depth': 2, '_refinement_events': []}
        exec(compile(instrument(source), '<fixture>', 'exec'), ns)
        daughters = ns['trace_centerline']({'point': np.zeros(3), '_refinement_depth': 1}, 3)
        self.assertEqual([d['_refinement_depth'] for d in daughters], [2, 2])
        self.assertEqual(ns['trace_centerline'](daughters[0], 3), [])


if __name__ == '__main__':
    unittest.main()
