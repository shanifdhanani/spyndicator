import os

import pandas as pd

from price_forecaster.src.main.code.file_based_historical_data_fetcher import FileBasedHistoricalDataFetcher


class TestFileBasedHistoricalDataFetcher():
    def setup(self):
        file_path = os.path.dirname(__file__) + "/../resources/spx_minutely_data.txt"
        self.data_fetcher = FileBasedHistoricalDataFetcher(file_path)

    def test_get_dataset_returns_properly_formatted_dataset(self):
        dataframe = self.data_fetcher.get_historical_data()
        assert isinstance(dataframe, pd.DataFrame)
        assert dataframe.index.name == "Datetime"
        assert dataframe.iloc[0]["Open"] == 4071.79
        assert str(dataframe.index[0]) == '2023-01-27 15:56:00'