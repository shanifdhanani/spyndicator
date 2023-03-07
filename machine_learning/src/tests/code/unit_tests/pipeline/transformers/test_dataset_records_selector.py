from ml_core.src.main.code.pipeline.transformers.dataset_records_selector import DatasetRecordsSelector
import pandas as pd
import numpy as np


class TestFeatureSelector:

    def __init__(self):
        self.transformer = None

    def setup(self):
        self.transformer = DatasetRecordsSelector()

    def test_we_remove_records_when_labels_are_invalid(self):
        nanny_frame = pd.DataFrame({
            'nanny': [5, 1, 1, 2, 3, 5],
            'buddy': [1, 2, 3, 4, 5, 6],
            'lava': [1, 5, 4, 3, 2, 1],
            'whiskey': [np.nan] * 6
        })

        labels = np.array([1, 2, np.nan, '#NAME', 5, '.MISSING'])

        df, _ = self.transformer.transform(nanny_frame, labels)
        assert(len(df)) == 3
