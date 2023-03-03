from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta

from price_modeling.src.main.code.constants import NameOfColumnForTradingDaysInFuture, LabelName


class DatasetGenerator():
    """
    This class is responsible for generating a training/evaluation dataset
    """

    MinuteFrequencyToSubsample = 15
    HoursInTradingDay = 6.5
    MinutesInHour = 60
    RecordsPerDay = (MinutesInHour * HoursInTradingDay / MinuteFrequencyToSubsample) + 1
    TimeframesInTheFutureInTradingDays = [
        0.5 / HoursInTradingDay, # 3:30 PM
        1.0 / HoursInTradingDay, # 3:00 PM
        2.0 / HoursInTradingDay, # 2:00 PM
        3.0 / HoursInTradingDay, # 1:00 PM
        4.0 / HoursInTradingDay, # 12:00 PM
        5.0 / HoursInTradingDay, # 11:00 AM
        6.0 / HoursInTradingDay, # 10:00 AM
        1.0, # 9:30 AM
        2.0,
        3.0,
        4.0,
        5.0
    ]

    def generate_dataset(self, source_data: pd.DataFrame, create_evaluation_dataset: bool = True) -> tuple:
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

        # Here we subsample by minute to reduce the dataset size and all of the duplicative data that tends to exist from one minute to the next
        training_data_candidates = training_data_candidates[training_data_candidates.index.minute % self.MinuteFrequencyToSubsample == 0]
        training_data_candidates = self._construct_dataset_with_labels_from_candidates(candidates = training_data_candidates)
        training_data = self._add_features_to_base_instances_and_return_data(base_instances = training_data_candidates)

        if create_evaluation_dataset:
            evaluation_data_candidates = self._construct_dataset_with_labels_from_candidates(candidates = evaluation_data_candidates)
            evaluation_data = self._add_features_to_base_instances_and_return_data(base_instances = evaluation_data_candidates)

        return training_data, evaluation_data

    def _construct_dataset_with_labels_from_candidates(self, candidates: pd.DataFrame) -> pd.DataFrame:
        """
        This method constructs a new dataset that uses the data from the candidates dataframe to act as the basis for creating training
        instances. Multiple new instances, one for each future time period that we would like to predict for, are created based on each
        record within the candidates dataframe.

        :param candidates (obj:`pd.DataFrame`): A dataframe that contains OHLC data
        :return (obj:`pd.DataFrame`): A dataframe in which each record represents a point in time and contains labels for the future return
        of the stock price at a specific point in the future (respective to the time of the instance)
        """

        dataset = None
        for prediction_timeframe in self.TimeframesInTheFutureInTradingDays:
            dataset_with_labels_for_future = self._create_dataset_with_future_percentage_return(candidates = candidates, trading_days_in_future = prediction_timeframe)
            if dataset is None:
                dataset = dataset_with_labels_for_future
            else:
                dataset = pd.concat([dataset, dataset_with_labels_for_future])
        dataset.replace([np.inf, -np.inf], np.nan)
        dataset.dropna(subset = [LabelName])
        return dataset

    def _create_dataset_with_future_percentage_return(self, candidates: pd.DataFrame, trading_days_in_future: float) -> pd.DataFrame:
        """
        This method is responsible for taking a source list of candidates and creating a new dataset from them, one which contains the
        percentage return of the stock given the timeframe in the future

        :param candidates (obj:`pd.DataFrame`): A dataframe that contains a list of OHLC data
        :param trading_days_in_future (float): The number of trading days in the future that we need to create labels for
        :return (obj:`pd.DataFrame`): A dataframe that contains a datetime column, the stock price, the days in the future, and the percentage return
        """

        time_at_prediction = self._get_time_for_prediction_given_timerange_in_future(trading_days_in_future = trading_days_in_future)
        forward_periods_for_percentage_return = self._get_forward_period_for_percentage_return(trading_days_in_future = trading_days_in_future)
        percentage_changes = candidates.pct_change(-forward_periods_for_percentage_return).between_time(time_at_prediction, time_at_prediction, inclusive = "both")
        percentage_changes[NameOfColumnForTradingDaysInFuture] = trading_days_in_future
        percentage_changes.rename(columns = {"Close": LabelName}, inplace = True)
        percentage_changes = percentage_changes[[LabelName, NameOfColumnForTradingDaysInFuture]]
        percentage_changes.dropna(subset = [LabelName])
        return percentage_changes

    def _get_time_for_prediction_given_timerange_in_future(self, trading_days_in_future: float) -> str:
        """
        This method calculates the time at which we are making a prediction given the parameter for trading days in the future

        :param trading_days_in_future (float): The number of trading days in the future that we need to predict for
        :return (str): The string representation of the time for which we are predicting
        """

        if trading_days_in_future < 1.0:
            number_of_minutes_until_close = round(trading_days_in_future * self.MinutesInHour * self.HoursInTradingDay, 1)
            timestamp = datetime.now()
            timestamp = timestamp.replace(hour = 16, minute = 00)
            timestamp = timestamp - timedelta(minutes = number_of_minutes_until_close)
            return datetime.strftime(timestamp, "%H:%M")

        return "09:30"

    def _get_forward_period_for_percentage_return(self, trading_days_in_future: float) -> int:
        """
        This method calculates the number of future periods in a dataframe that must be looked at, given a sub-sampled dataframe,
        in order to calculate the percentage return after `trading_days_in_future` in the future with respect to any given record

        :param trading_days_in_future (float): The number of trading days in the future that we need to predict for
        :return (int): The number of time periods in the future to look at given a sub-sampled dataframe
        """

        if trading_days_in_future >= 1.0:
            return int(round(trading_days_in_future * self.RecordsPerDay, 0))

        minutes_ahead = trading_days_in_future * self.MinutesInHour * self.HoursInTradingDay
        return int(round(minutes_ahead / self.MinuteFrequencyToSubsample, 0))