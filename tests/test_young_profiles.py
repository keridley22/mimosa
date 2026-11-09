from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import joblib
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

from plaque_pipeline.young_profiles import project_saved_tree
from plaque_pipeline.decision_tree import file_hash


class YoungProfilesTests(unittest.TestCase):
    def test_frozen_model_is_projected_without_refitting_or_relabelling_animals(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'tree.joblib'
            features=['size','shape']
            x=pd.DataFrame({'size':[1,2,10,11],'shape':[2,3,5,6]})
            tree=DecisionTreeClassifier(random_state=0).fit(x,['APP-WT','APP-WT','APP-FIRE','APP-FIRE'])
            joblib.dump(dict(model=tree,features=features,training_cohort='historical_only'),path)
            before=file_hash(path)
            young=pd.DataFrame(dict(image_name=['a','b'],animal_id=['young:PF1','young:PF2'],
                                     original_animal_id=['PF1','PF2'],cohort=['young','young'],
                                     Genotype=['APP-WT young','APP-FIRE young'],size=[1,11],shape=[2,6]))
            with patch.object(DecisionTreeClassifier,'fit',side_effect=AssertionError('Unexpected refit')):
                result=project_saved_tree(path,young,features,'saved_reference')
            self.assertEqual(result.predicted_label.tolist(),['APP-WT','APP-FIRE'])
            self.assertEqual(result.Genotype.tolist(),['APP-WT young','APP-FIRE young'])
            self.assertEqual(file_hash(path),before)
            with self.assertRaisesRegex(ValueError,'features or training cohort'):
                project_saved_tree(path,young,['shape','size'],'saved_reference')

    def test_nonhistorical_training_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'tree.joblib'
            joblib.dump(dict(features=['size'],training_cohort='young'),path)
            with self.assertRaisesRegex(ValueError,'training cohort'):
                project_saved_tree(path,pd.DataFrame(),['size'],'wrong_reference')
