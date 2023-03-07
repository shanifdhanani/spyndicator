class PandasColumnRearranger(object):
    """
    This class rearranges a pandas DataFrame columns into the order provided by the caller
    """

    __slots__ = 'ordered_features',

    def __init__(self, ordered_features):
        """
        :param ordered_features (obj:`list<str>`): A list of features names in the order in which they need to appear
        """

        self.ordered_features = ordered_features

    def transform(self, dataset):
        """
        Rearranges the provided dataset in order of `self.ordered_features`

        :param dataset: (obj:`pd.DataFrame`) A pandas DataFrame object
        :return (obj:`pd.DataFrame`) A pandas DataFrame object where the columns have been rearranged in the right order
        """

        # If the updated index has a feature not present in the dataset, it will add that as `np.nan` causing issues
        dataset_columns = dataset.columns
        updated_dataset_index = [feature for feature in self.ordered_features if feature in dataset_columns]
        dataset = dataset.reindex(columns = updated_dataset_index)
        return dataset
