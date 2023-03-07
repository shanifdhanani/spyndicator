import pandas as pd

from machine_learning.src.main.code.models.regression.decision_tree_regression_model import DecisionTreeRegressionModel
from machine_learning.src.main.code.models.regression.lasso_model import LassoModel
from machine_learning.src.main.code.models.regression.linear_regression_model import LinearRegressionModel
from machine_learning.src.main.code.models.regression.random_forest_regression_model import RandomForestRegressionModel
from machine_learning.src.main.code.models.regression.svm_model import SVMModel
from machine_learning.src.main.code.models.regression.xgb_model import XgbModel
from price_modeling.src.main.code.abstract_historical_data_fetcher import AbstractHistoricalDataFetcher
from price_modeling.src.main.code.constants.generic_constants import LabelName
from price_modeling.src.main.code.dataset_generator import DatasetGenerator
import pickle

from utils.src.main.code.logging.logger import Logger


class ModelGenerator:
    """
    This class is responsible for generating a model that predicts the future price of the S&P 500
    """

    logger = Logger.get_logger(__name__)

    def __init__(self, historical_data_fetcher: AbstractHistoricalDataFetcher, dataset_generator: DatasetGenerator = None):
        self.historical_data_fetcher = historical_data_fetcher
        self.dataset_generator = dataset_generator
        if self.dataset_generator is None:
            self.dataset_generator = DatasetGenerator()

    def build_model(self, saved_model_filepath: str, evaluate: bool = True) -> object:
        """
        This method is responsible for building the model from the data

        :param saved_model_filepath (str): Where the model should be saved after training
        :param evaluate (bool): True if you would like to add an extra step for evaluation, False if you just want to build the model
        :return (obj): Evaluation statistics if the "evaluate" paramter is set to True, otherwise a boolean to indicate success
        """

        self.logger.info("Starting process to build model")
        source_data = self.historical_data_fetcher.get_historical_data()
        training_data, evaluation_data = self.dataset_generator.generate_dataset(source_data = source_data, create_evaluation_dataset = evaluate)
        trained_model, evaluation_metrics = self._train_model(training_data = training_data, evaluation_data = evaluation_data)
        with open(saved_model_filepath, 'wb') as file:
            pickle.dump(trained_model, file)

        if evaluate:
            self.logger.info("Model generated and evaluation complete. Evaluation metrics are below:")
            self.logger.info(evaluation_metrics.get_in_json_format())
            return evaluation_metrics

        self.logger.info("Model generation complete")
        return True

    def _train_model(self, training_data: pd.DataFrame, evaluation_data: pd.DataFrame) -> tuple:
        """
        This method trains and optionally evaluates a model

        :param training_data (obj:`pd.DataFrame`): The training data
        :param evaluation_data (obj:`pd.Dataframe`): The optional evaluation data
        :return (tuple): The trained model and the evaluation statistics (if applicable)
        """

        model = XgbModel()
        training_labels = training_data[LabelName]
        del training_data[LabelName]
        evaluation_labels = None
        if evaluation_data is not None:
            evaluation_labels = evaluation_data[LabelName]
            del evaluation_data[LabelName]
        model_metrics = model.train_test_and_return_model_metrics(training_dataset = training_data, training_labels = training_labels, test_dataset = evaluation_data, test_labels = evaluation_labels, should_transform = True, should_add_explanation = False)
        return model, model_metrics