import datetime

from ml_core.src.main.code.pipeline.transformers.datetime_features_encoder import DateTimeFeaturesEncoder
import pandas as pd
import numpy as np


class TestDatetimeFeaturesEncoder:
    def setup(self):
        self.datetime_features_encoder = DateTimeFeaturesEncoder(datetime_features = ['datetime_stuff'])

    def test_one_hot_encoder_one_hot_encodes_properly(self):
        dataframe = pd.DataFrame({
            'stuff': ['stuff1', 'stuff2'],
            'datetime_stuff': [datetime.datetime.now(), np.nan]
        })

        assert len(dataframe.columns) == 2
        transformed_dataframe = self.datetime_features_encoder.transform(dataframe)
        assert len(dataframe.columns) == 2
        assert len(transformed_dataframe.columns) == 1
