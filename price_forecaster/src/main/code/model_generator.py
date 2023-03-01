from price_forecaster.src.main.code.abstract_historical_data_fetcher import AbstractHistoricalDataFetcher


class ModelGenerator:
    """
    This class is responsible for generating a model that predicts the future price of the S&P 500
    """

    def __init__(self, historical__data_fetcher: AbstractHistoricalDataFetcher):
        self.historical_data_fetcher = historical__data_fetcher