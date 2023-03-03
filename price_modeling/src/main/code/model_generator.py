from price_modeling.src.main.code.abstract_historical_data_fetcher import AbstractHistoricalDataFetcher
from price_modeling.src.main.code.dataset_generator import DatasetGenerator
import pickle

from utils.src.main.code.logging.logger import Logger


class ModelGenerator:
    """
    This class is responsible for generating a model that predicts the future price of the S&P 500
    """

    logger = Logger.get_logger(__name__)

    def __init__(self, historical_data_fetcher: AbstractHistoricalDataFetcher, dataset_generator: DatasetGenerator):
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

        source_data = self.historical_data_fetcher.get_historical_data()
        training_data, evaluation_data = self.dataset_generator.generate_dataset(source_data = source_data, create_evaluation_dataset = evaluate)
        trained_model, evaluation_metrics = self._get_trained_model(training_data = training_data, evaluation_data = evaluation_data)
        with open(saved_model_filepath, 'wb') as file:
            pickle.dump(trained_model, file)

        if evaluate:
            return evaluation_metrics

        return True