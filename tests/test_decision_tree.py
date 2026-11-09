from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd

from plaque_pipeline.config import Config
from plaque_pipeline.decision_tree import (animal_holdout, animal_cv_splits, check_marker_features,
                                          prepare_table, run_decision_tree)


class DecisionTreeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.cfg = replace(Config(), output_root=self.root/'current', previous_output_root=self.root/'previous',
                           historical_excluded_images=('excluded_image',), excluded_images=())

    def write_cohort(self, cohort):
        folder = (self.cfg.previous_output_root if cohort == 'historical' else self.cfg.output_root)/'morphology_results/combined'
        folder.mkdir(parents=True)
        features, central = [], {'MX04': [], 'OC': []}
        animals = [(f'{label}_{i}', label) for label in self.cfg.train_labels for i in range(6)] if cohort == 'historical' else [('PF1', 'APP-FIRE+transplant'), ('PF2', 'APP-FIRE')]
        for animal, label in animals:
            for plaque in range(2):
                identity = dict(image_name=f'{animal}_{plaque}', sample_id=animal, Genotype=label)
                if cohort != 'historical':
                    identity['animal_id'] = animal
                f = identity.copy()
                for marker, suffix, multiplier in [('MX04', 'm04', 1), ('OC', 'oc', 2)]:
                    value = multiplier * (100 if label == 'APP-WT' else 300) + plaque
                    props = dict(area_voxels=value/.0324, volume_um3=value, extent=.5, surface_area_um2=value*2,
                                 complexity_volume_surface=.5, sphericity=.8, solidity=.7, elongation=2,
                                 major_axis_length_um=10, minor_axis_length_um=5)
                    central[marker].append(dict(identity, **props))
                    for target, source in dict(area_voxels='area_voxels', volume_um3='volume_um3', extent='extent',
                                               surface_area='surface_area_um2', complexity='complexity_volume_surface',
                                               sphericity='sphericity', solidity='solidity', elongation='elongation',
                                               major_axis_length='major_axis_length_um', minor_axis_length='minor_axis_length_um').items():
                        f[f'{target}_{suffix}'] = props[source]
                f.update(volume_ratio_oc_m04=f['volume_um3_oc']/f['volume_um3_m04'],
                         surface_area_ratio_oc_m04=f['surface_area_oc']/f['surface_area_m04'],
                         oc_halo_volume_um3=f['volume_um3_oc']-f['volume_um3_m04'])
                features.append(f)
        for marker in central:
            pd.DataFrame(central[marker]).to_csv(folder/f'overall_{marker}_central.csv', index=False)
        path = folder/'decision_tree_features.csv'
        pd.DataFrame(features).to_csv(path, index=False)
        return path

    def test_no_animal_overlap_and_every_cv_plaque_predicted_once(self):
        path = self.write_cohort('historical')
        frame, _ = prepare_table(path, 'historical', self.cfg)
        coverage = np.zeros(len(frame))
        for train, test in animal_cv_splits(frame, self.cfg):
            self.assertFalse(set(frame.iloc[train].animal_id) & set(frame.iloc[test].animal_id))
            self.assertEqual(set(frame.iloc[test].Genotype), set(self.cfg.train_labels))
            coverage[test] += 1
        np.testing.assert_array_equal(coverage, 1)
        train, test = animal_holdout(frame, self.cfg)
        self.assertFalse(set(frame.iloc[train].animal_id) & set(frame.iloc[test].animal_id))

    def test_marker_swap_and_calibration_mismatch_are_rejected(self):
        path = self.write_cohort('historical')
        frame, _ = prepare_table(path, 'historical', self.cfg)
        check_marker_features(frame, path.parent, self.cfg)
        swapped = frame.copy()
        swapped['volume_um3_oc'] = swapped['volume_um3_m04']
        with self.assertRaises(ValueError):
            check_marker_features(swapped, path.parent, self.cfg)
        with self.assertRaises(ValueError):
            check_marker_features(frame, path.parent, replace(self.cfg, voxel_xy_um=.099))

    def test_exclusions_missing_features_and_conflicting_animals(self):
        path = self.write_cohort('historical')
        frame = pd.read_csv(path)
        frame.loc[0, 'image_name'] = 'excluded_image'
        frame.loc[1, 'solidity_oc'] = np.nan
        frame.to_csv(path, index=False)
        kept, dropped = prepare_table(path, 'historical', self.cfg)
        self.assertEqual(len(dropped), 2)
        self.assertEqual(len(kept), 22)
        self.assertTrue(dropped.exclusion_reason.str.contains('missing_or_nonfinite').any())
        frame.loc[3, 'Genotype'] = 'APP-FIRE'
        frame.to_csv(path, index=False)
        with self.assertRaises(ValueError):
            prepare_table(path, 'historical', self.cfg)

    def test_full_workflow_keeps_transplants_out_of_training(self):
        self.write_cohort('historical')
        self.write_cohort('current')
        out = self.root/'run'
        with patch('plaque_pipeline.decision_tree.save_tree'), patch('plaque_pipeline.decision_tree.save_summary_plot'):
            run_decision_tree(self.cfg, out)
        splits = pd.read_csv(out/'animal_split_membership.csv')
        self.assertTrue(splits.groupby('animal_id').holdout_split.nunique().eq(1).all())
        self.assertTrue(splits.groupby('animal_id').cv_test_fold.nunique().eq(1).all())
        historical = pd.read_csv(out/'historical_model_inputs.csv')
        self.assertTrue(historical.cohort.eq('historical').all())
        predictions = pd.read_csv(out/'transplanted_plaque_predictions.csv')
        self.assertEqual(set(predictions.original_animal_id), {'PF1'})
        self.assertNotIn('correct', predictions)
        self.assertTrue(predictions.prediction_scope.eq('phenotype_prediction').all())
        external = pd.read_csv(out/'new_cohort_controls_predictions.csv')
        self.assertEqual(set(external.original_animal_id), {'PF2'})
        saved = joblib.load(out/'final_tree.joblib')
        self.assertEqual(saved['model'].tree_.n_node_samples[0], len(historical))
        current = pd.read_csv(out/'new_cohort_model_inputs.csv').query("Genotype == 'APP-FIRE+transplant'")
        np.testing.assert_array_equal(saved['model'].predict(current[saved['features']]), predictions.predicted_label)
        metrics = pd.read_csv(out/'evaluation_metrics.csv')
        self.assertIn('training_resubstitution_not_independent', set(metrics.evaluation))
        self.assertEqual(len(pd.read_csv(out/'grouped_cv_predictions.csv')), len(historical))


if __name__ == '__main__':
    unittest.main()
