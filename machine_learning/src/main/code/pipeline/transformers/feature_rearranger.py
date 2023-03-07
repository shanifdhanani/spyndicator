class FeatureRearranger(object):
    """
    This class rearranges a provided feature dictionary into the order specified by the caller
    """

    def __init__(self, ordered_features: list):
        """
        :param ordered_features (obj:`list`): A list of features in the order in which they need to appear in the final array
        """

        self.ordered_features = ordered_features

    def transform(self, feature_dictionary: dict) -> list:
        """
        Rearranges the provided feature dictionary based on the provided list of features

        :param feature_dictionary (dict): A dictionary keyed by feature name containing a dataframe for each feature
        :return (obj:`list<pd.DataFrame>`)`: An ordered list of Pandas dataframes
        """

        data_matrix = []

        for feature in self.ordered_features:
            if feature not in feature_dictionary:
                continue
            data_matrix.append(feature_dictionary[feature])

        return data_matrix