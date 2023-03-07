from sklearn.exceptions import DataConversionWarning
from sklearn.preprocessing import RobustScaler
import warnings
import pandas as pd
import numpy as np

from utils.src.main.code.logging.logger import Logger


class OutlierRobustScaler(object):

    logger = Logger.get_logger(__name__)

    def __init__(self, max_absolute_value = 6):
        warnings.simplefilter("ignore", DataConversionWarning)

        self.scalers = dict()
        self.is_fit = False
        self.max_absolute_value = max_absolute_value

    def transform(self, dataframe):
        """
        Scales features using statistics that are robust to outliers

        :param dataframe: (obj: `pd.DataFrame`): A Pandas DataFrame
        :return (obj:`pd.DataFrame`): A Pandas DataFrame feature values (each column) has been scaled
        """

        scalar_columns = dataframe.select_dtypes(include = ['float64', 'int64']).columns
        for column in scalar_columns:
            scaler = self._get_scaler(dataframe, column)
            if scaler is not None:
                scaled_column = scaler.transform(dataframe[column].values.reshape(-1,1))
                dataframe[column] = scaled_column
                if self.max_absolute_value is not None:
                    dataframe[column] = dataframe[column].apply(lambda value: value if abs(value) < self.max_absolute_value else self.max_absolute_value * np.sign(value))

        self.is_fit = True
        return dataframe

    def _get_scaler(self, dataframe, column):
        """
        Gets the scaler (or creates a new one) for the provided column

        :param dataframe (obj: `pd.DataFrame`): The dataframe full of data
        :param column (str): The name of the column to scale
        :return: A scaler
        """

        if self.is_fit:
            if column in self.scalers:
                return self.scalers[column]
            return None

        scaler = RobustScaler()
        self.scalers[column] = scaler
        scaler.fit(dataframe[column].values.reshape(-1,1))
        return scaler
