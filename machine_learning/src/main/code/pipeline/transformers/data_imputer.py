import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

from machine_learning.src.main.code.pipeline.transformers.exceptions.columns_not_present_exception import ColumnsNotPresentException


class DataImputer(object):
    """
    Takes in data and imputes null values
    """

    def __init__(self):
        self.imputer = SimpleImputer(missing_values = np.nan, strategy = 'median')
        self.is_fit = False

    def transform(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """
        Takes a dataframe and fills missing values

        :param dataframe (obj:`pd.DataFrame`): A Pandas dataframe
        :return (obj:`pd.DataFrane`): A Pandas dataframe where null feature values have been imputed
        """

        if not self.is_fit:
            self._fit_imputer(dataframe)

        missing_columns = self._get_missing_columns(dataframe)
        if missing_columns is not None and len(missing_columns) > 0:
            raise ColumnsNotPresentException(", ".join(missing_columns) + " missing from the dataset")

        extraneous_columns = self._get_extraneous_columns(dataframe)
        for column in extraneous_columns:
            del dataframe[column]

        dataframe_subset = dataframe[self.columns]
        imputed_dataframe = pd.DataFrame(self.imputer.transform(dataframe_subset))
        imputed_dataframe.index = dataframe.index
        for column_index in imputed_dataframe.columns:
            column_name = dataframe_subset.columns[column_index]
            dataframe[column_name] = imputed_dataframe[column_index]

        return dataframe

    def _fit_imputer(self, dataframe: pd.DataFrame) -> None:
        """
        Creates the initial fit of the imputer on the provided (likely training) dataframe

        :param dataframe (obj:`pd.DataFrame`): A Pandas dataframe
        """

        dataframe.dropna(axis = 1, how = 'all', inplace = True)
        scalar_dataframe = dataframe.select_dtypes(include = ['float64', 'int64'])
        self.columns = scalar_dataframe.columns
        self.imputer.fit(scalar_dataframe)
        self.is_fit = True

    def _get_missing_columns(self, dataframe: pd.DataFrame) -> list:
        """
        Gets any columns that are missing from this dataframe

        :param dataframe (obj:`DataFrame`): A pandas dataframe
        :return: A list of missing columns
        """

        missing_columns = list()
        for column in self.columns:
            if column not in dataframe.columns:
                missing_columns.append(column)
        return missing_columns

    def _get_extraneous_columns(self, dataframe: pd.DataFrame) -> list:
        """
        Finds all columns in the provided dataframe that were not present at the time of fitting

        :param dataframe (obj:`pd.DataFrame`): A pandas dataframe
        :return: A list of columns that were not present at the time of fitting
        """

        extraneous_columns = list()
        for column in dataframe.select_dtypes(include = ['float64', 'int64']):
            if column not in self.columns:
                extraneous_columns.append(column)
        return extraneous_columns
