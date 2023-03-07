from ml_core.src.main.code.pipeline.transformers.standardization_scaler import StandardizationScaler
import pandas as pd
import numpy as np

class TestStandardizationScaler:
    def setup(self):
        self.scaler = StandardizationScaler()

    def test_standardization_scaler_scales_values(self):
        unscaled_dataframe = pd.DataFrame({'unscaled': [2,3,4,5,6,7,8,9]})
        assert np.mean(unscaled_dataframe['unscaled']) != 0
        assert np.std(unscaled_dataframe['unscaled']) != 1
        scaled_dataframe = self.scaler.transform(unscaled_dataframe)
        assert np.mean(scaled_dataframe['unscaled']) == 0
        assert np.std(scaled_dataframe['unscaled']) == 1