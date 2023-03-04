from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta

from price_modeling.src.main.code.constants.feature_names import Month, Day, DayName, Year, Quarter, MinutesSinceOpen, IsSameDayClose, NameOfColumnForTradingDaysInFuture, ReturnSinceLastClose, ReturnSinceOpen, ReturnInLast10Minutes, ReturnInLast30Minutes, ReturnInLast1Hour, ReturnInLast2Hours, ReturnInLast3Hours, ReturnInLast4Hours, ReturnInLast5Hours, ReturnInLast6Hours
from price_modeling.src.main.code.constants.generic_constants import LabelName, CloseColumnName


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
        training_data = self._add_features_to_base_instances_and_return_data(base_instances = training_data_candidates, source_data = source_data)

        if create_evaluation_dataset:
            evaluation_data_candidates = self._construct_dataset_with_labels_from_candidates(candidates = evaluation_data_candidates)
            evaluation_data = self._add_features_to_base_instances_and_return_data(base_instances = evaluation_data_candidates, source_data = source_data)

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
        percentage_changes.rename(columns = {CloseColumnName: LabelName}, inplace = True)
        percentage_changes = percentage_changes[[LabelName, NameOfColumnForTradingDaysInFuture]]
        percentage_changes.dropna(subset = [LabelName], inplace = True)
        percentage_changes = percentage_changes.join(candidates, how = "left")
        return percentage_changes[[CloseColumnName, NameOfColumnForTradingDaysInFuture, LabelName]]

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

    def _add_features_to_base_instances_and_return_data(self, base_instances: pd.DataFrame, source_data: pd.DataFrame) -> pd.DataFrame:
        """
        This method adds in all of the features that will be used for prediction into the provided dataset

        :param base_instances (obj:`pd.DataFrame`): A dataframe that contains labels and a column for how far in the future that is being predicted
        :param source_data (obj:`pd.DataFrame`): A dataframe that contains all the original data, useful for matching an instance to its closest closing value
        :return (obj:`pd.DataFrame`): A dataframe that contains all features and labels
        """

        def minutes_since_open(row) -> int:
            start_of_day = row.name # Here, the "name" column is actually the index, wich is the datetime of the record
            start_of_day = start_of_day.replace(hour = 9, minute = 30)
            difference = (row.name - start_of_day).total_seconds() / 60 # 60 seconds per minute
            return difference

        def get_return_since_time(time: str, base_instances: pd.DataFrame, name_of_new_column: str = "Returns") -> pd.DataFrame:
            prices_at_target_time = source_data.between_time(time, time, inclusive = "both")
            prices_at_target_time.sort_index(inplace = True)
            prices_at_target_time = pd.merge_asof(base_instances, prices_at_target_time, left_index = True, right_index = True)
            prices_at_target_time[name_of_new_column] = prices_at_target_time[CloseColumnName + "_x"] / prices_at_target_time[CloseColumnName + "_y"] - 1.0
            returns_since_target_time = prices_at_target_time[[name_of_new_column]]
            return returns_since_target_time

        def get_new_dataset_with_returns_for_specified_time(dataset: pd.DataFrame, new_column_name: str, periods: int, start_time_reset: str, end_time_reset: str) -> pd.DataFrame:
            dataset[new_column_name] = source_data.pct_change(periods)[CloseColumnName]
            index = dataset.between_time(start_time_reset, end_time_reset, inclusive = "left").index
            dataset.loc[index, new_column_name] = 0
            return dataset

        base_instances.sort_index(inplace = True)
        dataset = base_instances.sort_index()
        dataset[Month] = dataset.index.month
        dataset[Day] = dataset.index.day
        dataset[DayName] = dataset.index.day_name()
        dataset[Year] = dataset.index.year
        dataset[Quarter] = dataset.index.quarter
        dataset[MinutesSinceOpen] = dataset.apply(minutes_since_open, axis = 1)
        dataset[IsSameDayClose] = dataset[NameOfColumnForTradingDaysInFuture] <= 1.0

        returns_since_last_close = get_return_since_time("16:00", base_instances, ReturnSinceLastClose)
        dataset = dataset.join(returns_since_last_close, how = "left")
        dataset.drop_duplicates(inplace = True)

        returns_since_open = get_return_since_time("09:30", base_instances, ReturnSinceOpen)
        dataset = dataset.join(returns_since_open, how = "left")
        dataset.drop_duplicates(inplace = True)

        dataset = get_new_dataset_with_returns_for_specified_time(dataset = dataset, new_column_name = ReturnInLast10Minutes, periods = 10, start_time_reset = "9:30", end_time_reset = "9:40")
        dataset = get_new_dataset_with_returns_for_specified_time(dataset = dataset, new_column_name = ReturnInLast30Minutes, periods = 30, start_time_reset = "9:30", end_time_reset = "10:00")
        dataset = get_new_dataset_with_returns_for_specified_time(dataset = dataset, new_column_name = ReturnInLast1Hour, periods = self.MinutesInHour, start_time_reset = "9:30", end_time_reset = "10:30")
        dataset = get_new_dataset_with_returns_for_specified_time(dataset = dataset, new_column_name = ReturnInLast2Hours, periods = 2 * self.MinutesInHour, start_time_reset = "9:30", end_time_reset = "11:30")
        dataset = get_new_dataset_with_returns_for_specified_time(dataset = dataset, new_column_name = ReturnInLast3Hours, periods = 3 * self.MinutesInHour, start_time_reset = "9:30", end_time_reset = "12:30")
        dataset = get_new_dataset_with_returns_for_specified_time(dataset = dataset, new_column_name = ReturnInLast4Hours, periods = 4 * self.MinutesInHour, start_time_reset = "9:30", end_time_reset = "13:30")
        dataset = get_new_dataset_with_returns_for_specified_time(dataset = dataset, new_column_name = ReturnInLast5Hours, periods = 5 * self.MinutesInHour, start_time_reset = "9:30", end_time_reset = "14:30")
        dataset = get_new_dataset_with_returns_for_specified_time(dataset = dataset, new_column_name = ReturnInLast6Hours, periods = 6 * self.MinutesInHour, start_time_reset = "9:30", end_time_reset = "15:30")

        return dataset