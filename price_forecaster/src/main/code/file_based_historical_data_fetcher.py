import os
import pandas as pd

from price_forecaster.src.main.code.abstract_historical_data_fetcher import AbstractHistoricalDataFetcher


class FileBasedHistoricalDataFetcher(AbstractHistoricalDataFetcher):
    """
    This class retrieves a dataset of OHLC data from a local file and returns it as a Pandas dataframe
    """

    def __init__(self, file_path: str):
        """
        :param file_path (str): The local file path where OHLC data is stored as a CSV
        """

        self.file_path = file_path

    def get_historical_data(self):
        """
        Returns a dataset of OHLC data from the filepath provided

        :return (obj:`pd.DataFrame`): A Pandas dataframe of data
        """

        expanded_file_path = data_file_path = os.path.expanduser(self.file_path)
        stock_quotes = pd.read_csv(expanded_file_path, header = None)
        stock_quotes.columns = ["Datetime", "Open", "High", "Low", "Close"]
        stock_quotes["Datetime"] = pd.to_datetime(stock_quotes["Datetime"])
        stock_quotes = stock_quotes.set_index('Datetime')
        return stock_quotes