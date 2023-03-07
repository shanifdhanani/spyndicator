import pandas as pd

from machine_learning.src.main.code.pipeline.transformers.dataset_records_selector import DatasetRecordsSelector
from utils.src.main.code.logging.logger import Logger


class Pipeline(object):
    """
    This class provides a method for producing an incremental, step-by-step method for modifying
    and transforming a provided dataset. This allows for preparing and modifying data prior to using it
    to train a model, and provides a way of performing the same transformations on a new dataset that will be
    used to make predictions
    """

    logger = Logger.get_logger(__name__)

    def __init__(self, feature_name_to_type_mapping: dict = dict(), ordered_features: list = list(), categorical_features: list = list(), label_name: str = None, transformers: list = None):
        """
        :param feature_name_to_type_mapping (dict):
        :param ordered_features (list<str>):
        :param categorical_features (list<str>):
        :param label_name (str)
        :param transformers (obj:`list`): A list of transformer objects to be used in this pipeline, or None for the defaults
        """

        self.fit_features = list()
        self.feature_name_to_type_mapping = feature_name_to_type_mapping
        self.ordered_features = ordered_features
        self.categorical_features = categorical_features
        self.label_name = label_name
        self.ordered_transformers = transformers
        self.is_fit = False # This variable follows the sklearn convention of flagging whether the transformer has been fit to training data or not
        self.transformers = transformers
        if self.transformers is None:
            raise Exception("No transformers provided")

    def transform(self, dataset: pd.DataFrame, labels: list = None) -> tuple:
        """
        This method transforms the provided dataset using the set of defined transformers provided. If this is the first time data is
        passed into this pipeline instance, the pipeline will fit the data. If the pipeline is already fit, it will then transform
        the data in accordance with how it was fit.

        :param dataset (obj:`pd.DataFrame`): A dataframe that represents training/prediction data
        :param labels (obj:`pd.Series`): A list of labels
        :return (tuple): A transformed dataset object and labels
        """

        self.logger.info("Beginning to transform dataset")

        for transformer in self.transformers:
            self.logger.info("Starting transformer: " + str(transformer))
            try:
                if type(transformer) == DatasetRecordsSelector:
                    dataset, labels = transformer.transform(dataset, labels)
                else:
                    dataset = transformer.transform(dataset)
            except Exception as e:
                self.logger.error("Error transforming data in " + str(transformer), exc_info = True)
                continue

        if not self.is_fit:
            self.is_fit = True
        self.logger.info("Transformations complete")

        return dataset, labels

    def get_label_encoder_decoder(self):
        """
        Return the `LabelEncoderAndDecoder` transformer is it is present in the pipeline, else None

        :return: (obj:`LabelEncoderAndDecoder`) the transformer is it is present in the pipeline, else None
        """

        for transformer in self.transformers:
            if type(transformer) == LabelEncoder:
                return transformer

        return None
