import numpy as np
import pandas as pd

from ml_core.src.main.code.pipeline.transformers.zero_imputer import ZeroImputer


class TestZeroImputer:
    def setup(self):
        self.imputer = ZeroImputer()

    def test_null_values_imputed_to_0(self):
        nanny_frame = pd.DataFrame({'nanny': [np.nan, 1, 2, 3, np.nan]})
        assert nanny_frame.isnull().values.any()
        results = self.imputer.transform(nanny_frame)
        assert not results.isnull().values.any()
        assert results["nanny"][0] == 0
        assert results["nanny"][4] == 0