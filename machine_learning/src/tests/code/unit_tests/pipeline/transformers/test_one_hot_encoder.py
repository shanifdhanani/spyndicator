from ml_core.src.main.code.pipeline.transformers.one_hot_encoder import OneHotEncoder
import pandas as pd


class TestOneHotEncoder:
    def setup(self):
        self.one_hot_encoder = OneHotEncoder()

    def test_one_hot_encoder_one_hot_encodes_properly(self):
        one_cold = pd.DataFrame({
            'stuff': ['stuff1', 'stuff2'],
            'scalar_stuff': [2.1, 99]
        })

        assert len(one_cold.columns) == 2
        results = self.one_hot_encoder.transform(one_cold)
        assert len(results.columns) == 3
