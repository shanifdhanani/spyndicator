from utils.src.main.code.logging.logger import Logger


class ZeroImputer(object):
    """
    Takes in data and imputes null values with 0s
    """

    logger = Logger.get_logger(__name__)

    def transform(self, dataframe: pd.DataFrame):
        """
        Takes a dataframe and fills in missing values with 0s

        :param dataframe (obj:`pd.DataFrame`): A pandas dataframe
        :return (obj:`pd.DataFrame`): A dataframe
        """

        self.logger.info("Imputing missing values with 0s")
        scalar_columns = dataframe.select_dtypes(include = ['float64', 'int64']).columns
        dataframe = dataframe.dropna(axis = 1, how = 'all')
        for column in scalar_columns:
            feature_column = dataframe[column]
            imputed_feature_column = feature_column.fillna(value = 0)
            dataframe[column] = imputed_feature_column

        return dataframe