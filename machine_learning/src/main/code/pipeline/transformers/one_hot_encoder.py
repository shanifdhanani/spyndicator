import pandas as pd


class OneHotEncoder(object):
    """
    This class transforms a provided dataset by turning categorical features into a vector of one-hot-encoded variables
    """

    def __init__(self, categorical_features: list = None):
        self.is_fit = False
        self.original_category_values = dict()
        self.categorical_features = categorical_features

    def transform(self, dataframe):
        """
        Transforms the categorical variables in the provided dataframe into one hot encoded variables

        :param dataframe (obj:`pd.DataFrame`): A Pandas DataFrame
        :return (obj:`pd.DataFrame`): A Pandas DataFrame where feature values have been hot encoded variables
        """

        category_columns = self.categorical_features
        if category_columns is None:
            category_columns = dataframe.select_dtypes(include = ['object', 'bool']).columns

        category_columns = set(category_columns).intersection(set(dataframe.columns))

        for column in category_columns:
            dataframe[column] = dataframe[column].astype(pd.api.types.CategoricalDtype(categories = self._get_categories(dataframe, column)))

        self.is_fit = True
        transformed_dataframe = pd.get_dummies(dataframe, sparse = False)
        return transformed_dataframe

    def _get_categories(self, dataframe, column):
        """
        Returns the set of unique categorical values from the dataframe for the given column if they already exist,
        or records them if they don't and records/returns them

        :param dataframe (obj:`pd.DataFrame`): A Pandas dataframe
        :param column (obj:`str`): The string name of a column
        :return: A set of the column values
        """

        if self.is_fit:
            if column in self.original_category_values:
                return self.original_category_values[column]
            return None
        else:
            self.original_category_values[column] = dataframe[column].unique()
            return self.original_category_values[column]
