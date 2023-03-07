import os

from price_modeling.src.main.code.file_based_historical_data_fetcher import FileBasedHistoricalDataFetcher
from price_modeling.src.main.code.model_generator import ModelGenerator

if __name__ == "__main__":
    file_path = os.path.expanduser('~/Documents/projects/spyndicator-data/SPX_1min.txt')
    data_fetcher = FileBasedHistoricalDataFetcher(file_path = file_path)
    model_generator = ModelGenerator(historical_data_fetcher = data_fetcher)
    model_generator.build_model(saved_model_filepath = "/Users/shanif/Desktop/model.pickle", evaluate = True)