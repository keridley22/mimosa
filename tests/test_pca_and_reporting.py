from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from plaque_pipeline.config import Config
from plaque_pipeline.decision_tree import save_holdout_classification_report
from plaque_pipeline.pca import aggregate_animals, fit_profiles, select_features


class PcaAndReportingTests(unittest.TestCase):
    def test_report_uses_original_holdout_labels(self):
        predictions = pd.DataFrame(dict(Genotype=['APP-WT']*9+['APP-FIRE']*10,
                                        predicted_label=['APP-WT']*9+['APP-FIRE']*9+['APP-WT'],
                                        animal_id=['a']*9+['b']*5+['c']*5))
        original = predictions.copy(deep=True)
        with tempfile.TemporaryDirectory() as tmp:
            report = save_holdout_classification_report(predictions, ['APP-WT', 'APP-FIRE'], Path(tmp)).set_index('class_or_average')
            self.assertAlmostEqual(report.loc['APP-WT', 'precision'], .9)
            self.assertAlmostEqual(report.loc['APP-WT', 'recall'], 1.)
            self.assertAlmostEqual(report.loc['APP-FIRE', 'precision'], 1.)
            self.assertAlmostEqual(report.loc['APP-FIRE', 'recall'], .9)
            self.assertAlmostEqual(report.loc['macro avg', 'f1_score'], 18/19)
            self.assertEqual(report.loc['weighted avg', 'support'], 19)
            self.assertIn('94.74%', (Path(tmp)/'holdout_classification_report.md').read_text())
        pd.testing.assert_frame_equal(predictions, original)

    def profiles(self):
        return pd.DataFrame(dict(animal_id=['h1','h1','h2','h3','h4','t1','t1'],
                                 original_animal_id=['h1','h1','h2','h3','h4','t1','t1'],
                                 cohort=['historical']*5+['transplanted']*2,
                                 Genotype=['APP-WT']*3+['APP-FIRE']*2+['APP-FIRE+transplant']*2,
                                 f1=[1.,3.,4.,6.,8.,30.,40.], f2=[4.,2.,5.,8.,3.,-5.,-10.]))

    def test_transplants_do_not_change_historical_scaling_or_pca(self):
        data = self.profiles()
        first = fit_profiles(data, ['f1','f2'], 'historical')
        changed = data.copy()
        changed.loc[changed.cohort.eq('transplanted'), ['f1','f2']] *= 1000
        second = fit_profiles(changed, ['f1','f2'], 'historical')
        np.testing.assert_allclose(first['scaler'].mean_, data.iloc[:5][['f1','f2']].mean())
        np.testing.assert_array_equal(first['scaler'].mean_, second['scaler'].mean_)
        np.testing.assert_array_equal(first['model'].components_, second['model'].components_)
        np.testing.assert_array_equal(first['scores'].iloc[:5][['PC1','PC2']], second['scores'].iloc[:5][['PC1','PC2']])
        self.assertEqual(first['scores'].used_to_fit_pca.tolist(), [True]*5+[False]*2)

    def test_animal_means_precede_scaling_and_have_equal_fit_weight(self):
        data = self.profiles()
        animals = aggregate_animals(data, ['f1','f2'])
        self.assertEqual(len(animals), 5)
        self.assertEqual(animals.set_index('animal_id').loc['h1','f1'], 2.)
        self.assertEqual(animals.set_index('animal_id').loc['h1','n_plaques'], 2)
        fit = fit_profiles(animals, ['f1','f2'], 'historical')
        self.assertAlmostEqual(fit['scaler'].mean_[0], (2+4+6+8)/4)
        self.assertEqual(int(fit['scores'].used_to_fit_pca.sum()), 4)

    def test_duplicate_volume_and_constant_features_are_handled(self):
        cfg = replace(Config(), model_features=('volume_um3_m04','area_voxels_m04','f2'))
        data = self.profiles().rename(columns={'f1':'volume_um3_m04'})
        data['area_voxels_m04'] = data.volume_um3_m04/.0324
        chosen, selection = select_features(data, cfg)
        self.assertEqual(chosen, ['volume_um3_m04','f2'])
        self.assertEqual(selection.included_in_pca.tolist(), [True,False,True])
        data['constant'] = 1.
        fit = fit_profiles(data, chosen+['constant'], 'historical')
        self.assertEqual(fit['constant_features'], ['constant'])
        self.assertEqual(fit['active_features'], chosen)


if __name__ == '__main__':
    unittest.main()
