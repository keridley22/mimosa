from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
from scipy.stats import ttest_ind
import tifffile

from plaque_pipeline.config import Config
from plaque_pipeline.halo_stats import (
    DATASET, OUTCOMES, calculate_halo, divide_or_nan, load_cohort, audit_cohort, run_halo_stats,
)
from plaque_pipeline.stats import analyse_dataset


class HaloStatsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.cfg = replace(Config(), output_root=self.root / "current", previous_output_root=self.root / "old",
                           historical_excluded_images=("excluded",))

    def test_halo_is_set_difference_and_handles_empty_oc(self):
        mx04 = np.array([1, 1, 0, 0]).reshape(1, 2, 2)
        oc = np.array([0, 1, 1, 1]).reshape(1, 2, 2)
        mask, values = calculate_halo(mx04, oc, .0324)
        self.assertEqual(values["oc_halo_voxels"], 2)
        self.assertAlmostEqual(values["oc_halo_volume_um3"], .0648)
        self.assertAlmostEqual(values["oc_halo_fraction_of_oc"], 2/3)
        self.assertNotEqual(mask.sum(), oc.sum() - mx04.sum())
        _, empty = calculate_halo(mx04, np.zeros_like(oc), .0324)
        self.assertEqual(empty["oc_halo_volume_um3"], 0)
        self.assertTrue(np.isnan(empty["oc_halo_fraction_of_oc"]))
        for numerator, denominator in ((1, 0), (np.nan, 2), (2, np.nan), (1, -1)):
            self.assertTrue(np.isnan(divide_or_nan(numerator, denominator)))

    def write_cohort(self, root, records, historical=False):
        morphology = root / "morphology_results"
        (morphology / "combined").mkdir(parents=True)
        features, central = [], {"MX04": [], "OC": []}
        for name, animal, genotype, n_mx, n_oc in records:
            masks = {"MX04": (np.arange(12) < n_mx).reshape(2, 2, 3),
                     "OC": ((np.arange(12) >= 2) & (np.arange(12) < n_oc + 2)).reshape(2, 2, 3)}
            identity = dict(image_name=name, sample_id=animal, Genotype=genotype)
            if not historical:
                identity["animal_id"] = animal
            row = dict(identity)
            for marker, suffix in (("MX04", "m04"), ("OC", "oc")):
                folder = morphology / "central_masks" / marker
                folder.mkdir(parents=True, exist_ok=True)
                tifffile.imwrite(folder / f"{name}_{marker}_central.ome.tiff", masks[marker].astype("uint8"),
                                 photometric="minisblack", metadata={"axes": "ZYX"})
                count = int(masks[marker].sum())
                surface = float(count * 2 + 1)
                row.update({f"area_voxels_{suffix}": count, f"volume_um3_{suffix}": count * .0324,
                            f"surface_area_{suffix}": surface})
                central[marker].append(dict(identity, area_voxels=count, volume_um3=count * .0324,
                                            surface_area_um2=surface))
            halo, values = calculate_halo(masks["MX04"], masks["OC"], .0324)
            folder = morphology / "halo_masks"
            folder.mkdir(exist_ok=True)
            tifffile.imwrite(folder / f"{name}_OC_halo.ome.tiff", halo.astype("uint8"),
                             photometric="minisblack", metadata={"axes": "ZYX"})
            row.update(values)
            row["volume_ratio_oc_m04"] = row["volume_um3_oc"] / row["volume_um3_m04"]
            row["surface_area_ratio_oc_m04"] = row["surface_area_oc"] / row["surface_area_m04"]
            features.append(row)
        for marker, rows in central.items():
            pd.DataFrame(rows).to_csv(morphology / "combined" / f"overall_{marker}_central.csv", index=False)
        pd.DataFrame(features).to_csv(morphology / "combined/decision_tree_features.csv", index=False)

    def write_both(self):
        self.write_cohort(self.cfg.previous_output_root, [
            ("same", "A", "APPPS1", 2, 8), ("a_second", "A", "APPPS1", 8, 4),
            ("b", "B", "APPPS1", 3, 9), ("c", "C", "APPPS1xFIRE", 5, 6),
            ("d", "D", "APPPS1xFIRE", 6, 5), ("excluded", "E", "APPPS1", 1, 10),
        ], historical=True)
        self.write_cohort(self.cfg.output_root, [
            ("same", "A", "APP-FIRE+transplant", 4, 7),
            ("t2", "PF2", "APP-FIRE+transplant", 7, 8),
            ("new_control", "new", "APP-FIRE", 1, 10),
        ])

    def test_cohorts_ratios_replicates_and_exports(self):
        self.write_both()
        out = run_halo_stats(self.cfg, self.root / "results")
        plaques = pd.read_csv(out / "plaque_calculations.csv")
        means = pd.read_csv(out / f"{DATASET}_all_animal_means.csv")
        tests = pd.read_csv(out / f"{DATASET}_ttest_results.csv")
        self.assertEqual((len(plaques), len(means), len(tests)), (7, 6, 12))
        self.assertEqual(set(plaques.calculation_audit), {"passed"})
        self.assertTrue({"historical:A", "transplanted:A"}.issubset(set(means.animal_id)))
        self.assertFalse(plaques.image_name.duplicated().any())
        self.assertAlmostEqual(means.set_index("animal_id").loc["historical:A", "volume_ratio_oc_m04"], 2.25)
        # Mean plaque ratio (4 + .5)/2 differs from ratio of summed volumes 12/10.
        row = tests.query("feature == 'volume_ratio_oc_m04' and group1 == 'APP-WT' and group2 == 'APP-FIRE+transplant'").iloc[0]
        self.assertEqual((row.n1_animals, row.n2_animals), (2, 2))
        self.assertAlmostEqual(row.p_value, ttest_ind([2.25, 3], [1.75, 8/7], equal_var=False).pvalue)
        self.assertTrue(tests.FDR_BH_p_value.ge(tests.p_value - 1e-12).all())
        omitted = pd.read_csv(out / "omitted_source_records.csv")
        self.assertEqual(set(omitted.source_image_name), {"excluded", "new_control"})
        for feature in OUTCOMES:
            self.assertTrue((out / "animal_means" / f"{feature}.csv").is_file())
            self.assertTrue((out / "individual_plaque_values" / f"{feature}.csv").is_file())

    def test_stale_halo_is_rejected_before_export(self):
        self.write_both()
        df, _, _ = load_cohort(self.cfg.output_root, "transplanted", self.cfg)
        path = self.cfg.output_root / "morphology_results/halo_masks/same_OC_halo.ome.tiff"
        tifffile.imwrite(path, np.zeros((2, 2, 3), dtype="uint8"),
                         photometric="minisblack", metadata={"axes": "ZYX"})
        with self.assertRaisesRegex(ValueError, "Saved halo mask differs"):
            audit_cohort(df, self.cfg.output_root, self.cfg)

    def test_missing_outcome_does_not_drop_other_metrics(self):
        path = self.root / "missing.csv"
        rows = []
        for animal, group, value in (("a", "APP-WT", 1), ("b", "APP-WT", 3),
                                     ("c", "APP-FIRE+transplant", 4), ("d", "APP-FIRE+transplant", 8)):
            rows.append(dict(image_name=animal, animal_id=animal, Genotype=group,
                             volume_ratio_oc_m04=value,
                             surface_area_ratio_oc_m04=np.nan if animal == "a" else value))
        pd.DataFrame(rows).to_csv(path, index=False)
        result = analyse_dataset(path, "missing", self.cfg,
                                 feature_names=("volume_ratio_oc_m04", "surface_area_ratio_oc_m04"))
        comparison = result.query("group1 == 'APP-WT' and group2 == 'APP-FIRE+transplant'").set_index("feature")
        self.assertEqual(comparison.loc["volume_ratio_oc_m04", "status"], "ok")
        self.assertEqual(comparison.loc["surface_area_ratio_oc_m04", "status"], "insufficient_animals")


if __name__ == "__main__":
    unittest.main()
