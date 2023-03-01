from datetime import datetime

import pandas as pd
from dateutil.relativedelta import relativedelta


class DatasetGenerator():
    """
    This class is responsible for generating a training/evaluation dataset
    """

    MinuteFrequencyToSubsample = 15

    def generate_dataset(self, source_data: pd.DataFrame, create_evaluation_dataset: bool = True):
        """
        This method is responsible for generating a dataset and an optional evaluation dataset, depending on the value of the evaluate param

        :param source_data (obj:`pd.DataFrame`): A Pandas dataframe that contains OHLC data
        :param create_evaluation_dataset (bool): Whether or not to create an evaluation dataset
        :return (tuple): A training dataset and optional evaluation dataset (or null)
        """

        evaluation_data_candidates = None
        evaluation_data = None

        if create_evaluation_dataset:
            one_year_ago = datetime.today() - relativedelta(years = 1)
            training_data_candidates = source_data[source_data.index < one_year_ago]
            evaluation_data_candidates = source_data[source_data.index >= one_year_ago]
        else:
            training_data_candidates = source_data

        training_data_candidates = self._subsample_data_by_time(dataframe = training_data_candidates, minutes = self.MinuteFrequencyToSubsample)
        training_data_candidates = self._construct_dataset_with_labels_from_candidates(candidates = training_data_candidates)
        training_data = self._add_features_to_base_instances_and_return_data(base_instances = training_data_candidates)

        if create_evaluation_dataset:
            evaluation_data_candidates = self._construct_dataset_with_labels_from_candidates(candidates = evaluation_data_candidates)
            evaluation_data = self._add_features_to_base_instances_and_return_data(base_instances = evaluation_data_candidates)

        return training_data, evaluation_data