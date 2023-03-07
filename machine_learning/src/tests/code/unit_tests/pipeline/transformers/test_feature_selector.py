from ml_core.src.main.code.pipeline.transformers.feature_selector import FeatureSelector
import pandas as pd
import numpy as np


class TestFeatureSelector:
    def setup(self):
        self.feature_selector = FeatureSelector(all_features = ['nanny', 'buddy', 'lava', 'whiskey'])

    def test_we_select_features(self):
        nanny_frame = pd.DataFrame({
            'nanny': [5, 1, 1, 2, 3, 5],
            'buddy': [1,2,3,4,5,6],
            'lava': [1,5,4,3,2,1],
            'whiskey': [np.nan] * 6
        })
        labels = np.array([1,2,3,4,5,6])
        df, _ = self.feature_selector.transform(nanny_frame, labels)
        assert len(list(df.columns)) == 3
