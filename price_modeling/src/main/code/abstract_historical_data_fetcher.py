from abc import ABC, abstractmethod


class AbstractHistoricalDataFetcher(ABC):
    """
    This is an abstract class that defines the methods needed to fetch and return a historical dataset as a pandas dataframe
    """

    @abstractmethod
    def get_historical_data(self):
        """
        Subclasses must implement this method to generate a Pandas dataframe of OHLC data and return it to the caller

        :return (obj:`pd.DataFrame`): A Pandas dataframe
        """
        ...