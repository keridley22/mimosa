from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import tifffile

from plaque_pipeline.young_config import YOUNG_CFG
from plaque_pipeline.metadata import load_metadata
from plaque_pipeline.calibration import validate_pixels, validate_ometiff
from plaque_pipeline.preprocess import match_analysis_spacing, copy_training_subset
from plaque_pipeline.io_utils import save_ome_zyx
from plaque_pipeline.stats import historical_comparison_input, analyse_dataset
from plaque_pipeline.halo_stats import load_cohort, run_halo_stats, DATASET
import test_halo_stats as halo_fixtures


class YoungTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.cfg = replace(YOUNG_CFG, output_root=self.root/'out', previous_output_root=self.root/'old',
                           genotype_csv=self.root/'metadata.csv', raw_input_dir=self.root/'raw')

    def test_metadata_mapping_does_not_relabel_historical_controls(self):
        pd.DataFrame([dict(filename='A_1', animal_id='PF1', Genotype='APP-WT'),
                      dict(filename='B_1', animal_id='PF2', Genotype='APP-FIRE')]).to_csv(self.cfg.genotype_csv,index=False)
        lookup = load_metadata(self.cfg)
        self.assertEqual(lookup['A_1']['Genotype'], 'APP-WT young')
        self.assertEqual(lookup['B_1']['Genotype'], 'APP-FIRE young')
        self.assertEqual(self.cfg.genotype_map['APP-WT'], 'APP-WT')
        self.assertEqual(self.cfg.genotype_map['APPPS1'], 'APP-WT')

    def test_spacing_resampling_preserves_geometry_and_channel_separation(self):
        fine = dict(X=.13875024437927663,Y=.13875024437927663,Z=1.000715)
        arr = np.zeros((2,2,1024,1024),dtype='uint8')
        arr[:,0,256:768,256:768]=100
        arr[:,1]=200
        result, changed = match_analysis_spacing(arr, fine, self.cfg)
        self.assertTrue(changed)
        self.assertEqual(result.shape,(2,2,789,789))
        np.testing.assert_allclose(result[:,1],200)
        physical_before = (arr[0,0]>50).sum()*fine['X']**2
        physical_after = (result[0,0]>50).sum()*self.cfg.voxel_xy_um**2
        self.assertLess(abs(physical_after/physical_before-1),.01)
        self.assertGreater(physical_before/(512*512*.18**2),.59)
        self.assertLess(physical_before/(512*512*.18**2),.60)
        original, changed = match_analysis_spacing(arr,dict(X=.180375,Y=.180375,Z=1.000715),self.cfg)
        self.assertFalse(changed)
        self.assertIs(original,arr)
        path=self.root/'test.ome.tiff'
        save_ome_zyx(path,result[:,0],self.cfg)
        with tifffile.TiffFile(path) as t:
            self.assertEqual(t.series[0].axes,'ZYX')
            self.assertIn('PhysicalSizeX="0.18"', t.ome_metadata)

    def test_only_known_acquisition_spacings_pass(self):
        pixels=ET.fromstring('<Pixels xmlns="urn:ome" PhysicalSizeX="0.13875024437927663" PhysicalSizeY="0.13875024437927663" PhysicalSizeZ="1.000715"><Channel Name="Leica/ALEXA 594"/><Channel Name="Leica/DAPI"/></Pixels>')
        self.assertAlmostEqual(validate_pixels(pixels,self.cfg)['X'],.13875024437927663)
        with self.assertRaises(ValueError):
            validate_pixels(pixels,replace(self.cfg,resample_input_to_nominal=False))
        pixels.set('PhysicalSizeX','.10')
        with self.assertRaises(ValueError):
            validate_pixels(pixels,self.cfg)

    def test_four_groups_remain_distinct_in_marker_and_halo_statistics(self):
        helper=halo_fixtures.HaloStatsTests()
        helper.cfg=self.cfg
        helper.write_cohort(self.cfg.previous_output_root,[
            ('same','A','APPPS1',2,8),('h2','B','APPPS1',3,8),
            ('h3','C','APPPS1xFIRE',4,6),('h4','D','APPPS1xFIRE',5,6)],historical=True)
        helper.write_cohort(self.cfg.output_root,[
            ('same','A','APP-WT young',2,7),('y2','PF2','APP-WT young',3,6),
            ('y3','PF3','APP-FIRE young',4,7),('y4','PF4','APP-FIRE young',5,8)])
        current=self.cfg.morphology_combined_dir/'overall_MX04_central.csv'
        path=historical_comparison_input(current,'MX04',self.cfg)
        frame=pd.read_csv(path)
        self.assertEqual(set(frame.Genotype),set(self.cfg.analysis_genotypes))
        self.assertEqual(set(frame.cohort),{'historical','young'})
        self.assertEqual(frame.animal_id.nunique(),8)
        result=analyse_dataset(path,'MX04',self.cfg)
        self.assertTrue(result.n1_animals.eq(2).all())
        self.assertTrue(result.n2_animals.eq(2).all())
        out=run_halo_stats(self.cfg, self.root/'halo')
        tests=pd.read_csv(out/f'{DATASET}_ttest_results.csv')
        self.assertEqual(len(tests),24)
        self.assertTrue(tests.status.eq('ok').all())
        self.assertEqual(len(pd.read_csv(out/f'{DATASET}_all_animal_means.csv')),8)

    def test_training_subset_links_identical_pixels_without_copying(self):
        for marker in ('MX04','OC'):
            folder=self.cfg.preprocessed_dir/marker
            folder.mkdir(parents=True)
            path=folder/'image.ome.tiff'
            path.write_bytes(b'pixel-data')
        copy_training_subset(self.cfg)
        for marker in ('MX04','OC'):
            a=self.cfg.preprocessed_dir/marker/'image.ome.tiff'
            b=self.cfg.training_subset_dir/marker/'image.ome.tiff'
            self.assertEqual(a.stat().st_ino,b.stat().st_ino)
            self.assertEqual(a.read_bytes(),b.read_bytes())


if __name__=='__main__':
    unittest.main()
