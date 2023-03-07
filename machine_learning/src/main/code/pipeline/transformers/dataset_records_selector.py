import numpy as np
import pandas as pd

from utils.src.main.code.logging.logger import Logger


class DatasetRecordsSelector:
    """
    1. We drop all records from the dataset for which the labels are invalid
    2. We drop all records where the date columns have less than 5% invalid records
    """

    # This is a list of common bad values that can be found in datasets and outputs from excel/CSVs
    InvalidLabelTokens = set(['.MISSING', '.OTHER', '.UNKNOWN', 'nan', 'NAN', 'NA', 'na', 'ERROR', '#NAME', ' ', '',
                              'NaN', 'N/A', '#ERROR', '#NULL!', '#NUM!', '#REF!', '#VALUE!', '#DIV/0', '#NAME?',
                              '#N/A', '#####', 'None'])

    logger = Logger.get_logger(__name__)

    def __init__(self):
        """
        Initialize the transformer
        """

    def transform(self, dataset: pd.DataFrame, labels: list = None) -> tuple:
        """
        Take in a dataset and return a dataset and labels where all records with bad labels have been removed

        :param dataset (obj:`pd.DataFrame`): A pd.DataFrame object
        :param labels (list): A list of labels ordered respective to the dataset
        :return tuple: A pd.DataFrame object where rows corresponding to invalid labels have been removed and a list of updated labels corresponding to the updated dataset
        """

        # If labels are not present, then we just return the dataset and labels as is
        if labels is None:
            return dataset, labels

        invalid_labels_mask = enumerate(map(self._is_label_invalid, labels))
        invalid_labels_record_indexes = [index for index, mask in invalid_labels_mask if mask is True]

        dataset = dataset.drop(invalid_labels_record_indexes, inplace = False)
        labels = [label for index, label in enumerate(labels) if index not in invalid_labels_record_indexes]

        return dataset, labels

    def _is_label_invalid(self, label) -> bool:
        if label is None:
            return True

        if label in self.InvalidLabelTokens:
            return True

        try:
            if np.isnan(label) or not np.isfinite(label):
                return True
        except:
            return False

        return False


