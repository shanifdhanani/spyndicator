from ml_core.src.main.code.pipeline.transformers.frequency_encoder import FrequencyEncoder
import pandas as pd


class TestFrequencyEncoder:
    def setup(self):
        self.frequency_encoder = FrequencyEncoder()

    def test_one_hot_encoder_one_hot_encodes_properly(self):
        one_cold = pd.DataFrame({
            'stuff': ['stuff1', 'stuff2', 'stuff3', 'stuff1'],
            'scalar_stuff': [2.1, 99, 33, 342.1]
        })

        assert len(one_cold.columns) == 2
        results = self.frequency_encoder.transform(one_cold)
        assert len(results.columns) == 2
        transformed_results = list(results['stuff'])
        assert 3 == transformed_results[0]
        assert (1 == transformed_results[1] or 2 == transformed_results[1])
        assert (1 == transformed_results[2] or 2 == transformed_results[2])
        assert 3 == transformed_results[3]
