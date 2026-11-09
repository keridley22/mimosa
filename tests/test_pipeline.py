from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import json
import xml.etree.ElementTree as ET

import h5py
import numpy as np
import pandas as pd
import tifffile
from scipy.stats import ttest_ind

from plaque_pipeline.config import Config
from plaque_pipeline.calibration import validate_pixels
from plaque_pipeline.metadata import load_metadata, resolve_metadata
from plaque_pipeline.probabilities import probability_to_zyx, _clean_h5_base, h5_to_all_masks
from plaque_pipeline.stats import analyse_dataset, bh_adjust, run_default_ttests, historical_comparison_input


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.cfg = replace(Config(), output_root=self.root / "out", genotype_csv=self.root / "metadata.csv",
                           raw_input_dir=self.root / "raw", include_historical_controls=False)

    def test_historical_controls_preserve_markers_and_keep_cohorts_distinct(self):
        cfg = replace(self.cfg, previous_output_root=self.root/"previous", include_historical_controls=True,
                      historical_excluded_images=("excluded",))
        historical = cfg.historical_control_table("MX04")
        self.assertEqual(historical.name, "overall_MX04_central.csv")
        self.assertEqual(cfg.historical_control_table("OC").name, "overall_OC_central.csv")
        historical.parent.mkdir(parents=True)
        pd.DataFrame([dict(image_name="same_image", sample_id="A_2", Genotype="APPPS1", volume_um3=10, area_voxels=10/.0324),
                      dict(image_name="fire_image", sample_id="B_2", Genotype="APPPS1xFIRE", volume_um3=12, area_voxels=12/.0324),
                      dict(image_name="excluded", sample_id="B_2", Genotype="APPPS1xFIRE", volume_um3=100, area_voxels=100/.0324)]).to_csv(historical, index=False)
        current = self.root/"current.csv"
        pd.DataFrame([dict(image_name="same_image", sample_id="A_2", animal_id="PF742", Genotype="APP-FIRE+transplant", volume_um3=11, area_voxels=11/.0324),
                      dict(image_name="new_control", sample_id="B_2", animal_id="PF349", Genotype="APP-FIRE", volume_um3=99, area_voxels=99/.0324)]).to_csv(current, index=False)
        d = pd.read_csv(historical_comparison_input(current, "MX04", cfg))
        self.assertEqual(d.volume_um3.tolist(), [10, 12, 11])
        self.assertEqual(d.animal_id.tolist(), ["historical:A_2", "historical:B_2", "transplanted:PF742"])
        self.assertFalse(d.image_name.duplicated().any())
        self.assertEqual(d.source_marker_label.tolist(), ["MX04", "MX04", "MX04"])
        result = analyse_dataset(historical_comparison_input(current, "MX04", cfg), "MX04", cfg)
        self.assertTrue((result.status == "insufficient_animals").all())

    def test_animals_are_replicates_and_all_csv_styles_are_written(self):
        rows = []
        # Animal a contributes many plaques but still counts only once.
        for animal, group, values in [("a", "APP-WT", [1] * 20), ("b", "APP-WT", [3]),
                                       ("c", "APP-FIRE+transplant", [4, 8]),
                                       ("d", "APP-FIRE+transplant", [9])]:
            for i, value in enumerate(values):
                rows.append(dict(image_name=f"{animal}_{i}", animal_id=animal, Genotype=group,
                                 volume_um3=value, solidity=np.nan, qc_raw_fraction_on=.5,
                                 centroid_z_um=3))
        path = self.root / "central.csv"
        pd.DataFrame(rows).to_csv(path, index=False)
        result = analyse_dataset(path, "MX04", self.cfg)
        test = result.query("feature == 'volume_um3' and group1 == 'APP-WT' and group2 == 'APP-FIRE+transplant'").iloc[0]
        self.assertEqual(test.n1_animals, 2)
        self.assertEqual(test.n2_animals, 2)
        self.assertAlmostEqual(test.p_value, ttest_ind([1, 3], [6, 9], equal_var=False).pvalue)
        out = self.cfg.ttest_dir / "MX04"
        self.assertEqual(list(pd.read_csv(out / "individual_plaque_values/volume_um3.csv")), ["a", "b", "c", "d"])
        means = pd.read_csv(out / "animal_means/volume_um3.csv")
        self.assertEqual(list(means), list(self.cfg.analysis_genotypes))
        self.assertEqual(means["APP-FIRE+transplant"].dropna().tolist(), [6, 9])
        self.assertFalse((out / "animal_means/centroid_z_um.csv").exists())
        self.assertTrue(result.loc[result.status != "ok", "FDR_BH_p_value"].isna().all())

    def test_fdr_and_all_invalid_tests(self):
        np.testing.assert_allclose(bh_adjust([.01, .04, .03, np.nan]), [.03, .04, .04, np.nan], equal_nan=True)
        path = self.root / "one.csv"
        pd.DataFrame([dict(image_name="a", animal_id="PF1", Genotype="APP-FIRE", volume_um3=2),
                      dict(image_name="b", animal_id="PF2", Genotype="APP-FIRE+transplant", volume_um3=3)]).to_csv(path, index=False)
        result = analyse_dataset(path, "OC", self.cfg)
        self.assertTrue((result.status == "insufficient_animals").all())
        self.assertTrue(result.p_value.isna().all())
        self.assertFalse(result.significant_FDR.any())

    def test_metadata_resolves_animals_and_rejects_conflicts(self):
        pd.DataFrame([dict(filename="A_2", animal_id="PF742", Genotype="APP-FIRE+transplant"),
                      dict(filename="A_3", animal_id="PF742", Genotype="APP-FIRE+transplant")]).to_csv(self.cfg.genotype_csv, index=False)
        lookup = load_metadata(self.cfg)
        self.assertEqual(resolve_metadata("M04405_OC594_slideA2_20251119__s000_A_2_M04_OC", lookup)["animal_id"], "PF742")
        self.assertEqual(resolve_metadata("A_3_image", lookup)["animal_id"], "PF742")
        with self.assertRaises(ValueError):
            resolve_metadata("M04405_OC594_slideA2_20251119__s000_B_2_M04_OC", lookup)
        with self.assertRaises(ValueError):
            resolve_metadata("B_4_image", lookup)

    def test_channels_spacing_and_probability_axes(self):
        pixels = ET.fromstring('<Pixels xmlns="urn:ome" PhysicalSizeX="0.180375" PhysicalSizeY="0.180375" PhysicalSizeZ="1.000715"><Channel Name="Leica/ALEXA 594"/><Channel Name="Leica/DAPI"/></Pixels>')
        validate_pixels(pixels, self.cfg)
        self.assertEqual((self.cfg.mx04_channel_index, self.cfg.oc_channel_index), (0, 1))
        with self.assertRaises(ValueError):
            validate_pixels(pixels, replace(self.cfg, mx04_channel_index=1, oc_channel_index=1))
        wrong_order = ET.fromstring('<Pixels xmlns="urn:ome" PhysicalSizeX="0.180375" PhysicalSizeY="0.180375" PhysicalSizeZ="1.000715"><Channel Name="Leica/DAPI"/><Channel Name="Leica/ALEXA 594"/></Pixels>')
        with self.assertRaises(ValueError):
            validate_pixels(wrong_order, self.cfg)
        pixels.set("PhysicalSizeX", "0.099")
        with self.assertRaises(ValueError):
            validate_pixels(pixels, self.cfg)
        arr = np.zeros((2, 3, 5, 7), dtype=np.float32)
        arr[0] = .8
        arr[1] = .2
        np.testing.assert_allclose(probability_to_zyx(arr, 0, "czyx"), .8)
        with self.assertRaises(ValueError):
            probability_to_zyx(np.zeros((2, 3, 5, 7, 2)), 0, "tzyxc")
        self.assertEqual(_clean_h5_base(Path("sample_MX04.ome_Probabilities.h5"), "MX04"), "sample")

    def test_preprocessing_keeps_biological_channel_identity(self):
        from plaque_pipeline.preprocess import preprocess_ometiff_folder
        self.cfg.raw_input_dir.mkdir()
        pd.DataFrame([dict(filename="A_2", animal_id="PF742", Genotype="APP-FIRE+transplant")]).to_csv(self.cfg.genotype_csv, index=False)
        raw = np.zeros((3, 2, 12, 12), dtype=np.uint16)
        raw[:, 0, 2:5, 7:9] = 100  # MX04: distinct position and brightness.
        raw[:, 1, 8:10, 1:4] = 700  # OC.
        name = "A_2_M04_OC_L1_P+_191125_1"
        tifffile.imwrite(self.cfg.raw_input_dir / f'{name}.ome.tiff', raw, ome=True,
                         metadata=dict(axes='ZCYX', PhysicalSizeX=.18, PhysicalSizeY=.18,
                                       PhysicalSizeZ=1., Channel=dict(Name=list(self.cfg.expected_ome_channel_names))))
        with patch('plaque_pipeline.preprocess.preprocess_stack_zyx', side_effect=lambda stack, cfg: stack):
            preprocess_ometiff_folder(self.cfg)
        for index, channel in enumerate(('MX04', 'OC')):
            result = tifffile.imread(self.cfg.preprocessed_dir / channel / f'{name}_{channel}.ome.tiff')
            np.testing.assert_array_equal(result, raw[:, index])

    def test_automatic_batch_model_routing_and_resume_provenance(self):
        from plaque_pipeline.segmentation import build_command, can_resume, sha256
        from plaque_pipeline.setup import validate_output_root
        for channel in ('MX04', 'OC'):
            command = build_command(self.cfg, channel, self.root/'input.tiff', self.root/'output.h5')
            self.assertIn(f'--project={self.cfg.previous_output_root}/ilastik_training_subset/{channel}/{channel}_pixel_classifier.ilp', command)
            self.assertIn('--readonly', command)
            self.assertIn('--output_format=compressed hdf5', command)
        with self.assertRaises(ValueError):
            validate_output_root(replace(self.cfg, output_root=self.cfg.previous_output_root.parent/'transplanted_plaque_pipeline_outputs'))
        output = self.root/'output.h5'
        output.write_bytes(b'test-probabilities')
        with self.assertRaises(ValueError):
            can_resume(output, dict(marker='OC', project_sha256='correct-model'))
        record = dict(marker='MX04', project_sha256='wrong-model', output_sha256=sha256(output))
        output.with_suffix('.provenance.json').write_text(json.dumps(record))
        with self.assertRaises(ValueError):
            can_resume(output, dict(marker='OC', project_sha256='correct-model'))
        record.update(marker='OC', project_sha256='correct-model')
        output.with_suffix('.provenance.json').write_text(json.dumps(record))
        self.assertTrue(can_resume(output, dict(marker='OC', project_sha256='correct-model')))
        output.write_bytes(b'changed')
        with self.assertRaises(ValueError):
            can_resume(output, dict(marker='OC', project_sha256='correct-model'))

    def test_end_to_end_masks_morphology_and_stats(self):
        from plaque_pipeline.morphology import process_morphology
        cfg = replace(self.cfg, crop_frac=0, central_min_size_voxels=5,
                      per_object_min_area_voxels=5, morphology_closing_radius=1)
        pd.DataFrame([dict(filename="A_2", animal_id="PF742", Genotype="APP-FIRE+transplant")]).to_csv(cfg.genotype_csv, index=False)
        name = "A_2_M04_OC_L1_P+_191125_1"
        z, y, x = np.ogrid[:9, :32, :32]
        inside = ((z-4)/3)**2 + ((y-16)/7)**2 + ((x-16)/7)**2 < 1
        for channel in ("MX04", "OC"):
            rawdir, probdir = cfg.preprocessed_dir/channel, cfg.probability_dir/channel
            rawdir.mkdir(parents=True)
            probdir.mkdir(parents=True)
            tifffile.imwrite(rawdir / f"{name}_{channel}.ome.tiff", inside.astype(np.uint16), ome=True,
                             metadata={"axes": "ZYX"})
            arr = np.stack([np.where(inside, .9, .1), np.where(inside, .1, .9)], axis=-1).astype(np.float32)
            with h5py.File(probdir/f"{name}_{channel}_Probabilities.h5", "w") as h:
                h.create_dataset("exported_data", data=arr)
        h5_to_all_masks(cfg)
        process_morphology(cfg)
        run_default_ttests(cfg)
        for channel in ("MX04", "OC"):
            frame = pd.read_csv(cfg.morphology_combined_dir/f"overall_{channel}_central.csv")
            self.assertEqual(frame.animal_id.tolist(), ["PF742"])
            self.assertGreater(frame.volume_um3.iloc[0], 0)
            results = pd.read_csv(cfg.ttest_dir/channel/f"{channel}_ttest_results.csv")
            self.assertTrue(results.p_value.isna().all())
        (cfg.probability_dir/"OC"/f"{name}_OC_Probabilities.h5").unlink()
        with self.assertRaises(ValueError):
            h5_to_all_masks(cfg)


if __name__ == "__main__":
    unittest.main()
