from price_modeling.src.main.code.dataset_generator import DatasetGenerator



class TestDatasetGenerator():
    def setup(self):
        self.dataset_generator = DatasetGenerator()

    def test_get_time_for_prediction_given_timerange_in_future_returns_correct_time(self):
        assert "09:30" == self.dataset_generator._get_time_for_prediction_given_timerange_in_future(3.0)
        assert "09:30" == self.dataset_generator._get_time_for_prediction_given_timerange_in_future(2.0)
        assert "09:30" == self.dataset_generator._get_time_for_prediction_given_timerange_in_future(1.0)
        assert "10:00" == self.dataset_generator._get_time_for_prediction_given_timerange_in_future(6.0 / 6.5)
        assert "11:00" == self.dataset_generator._get_time_for_prediction_given_timerange_in_future(5.0 / 6.5)
        assert "12:00" == self.dataset_generator._get_time_for_prediction_given_timerange_in_future(4.0 / 6.5)
        assert "13:00" == self.dataset_generator._get_time_for_prediction_given_timerange_in_future(3.0 / 6.5)
        assert "14:00" == self.dataset_generator._get_time_for_prediction_given_timerange_in_future(2.0 / 6.5)
        assert "15:00" == self.dataset_generator._get_time_for_prediction_given_timerange_in_future(1.0 / 6.5)
        assert "15:30" == self.dataset_generator._get_time_for_prediction_given_timerange_in_future(0.5 / 6.5)

    def test_get_forward_period_for_percentage_return(self):
        assert 2 == self.dataset_generator._get_forward_period_for_percentage_return(0.5/6.5)
        assert 4 == self.dataset_generator._get_forward_period_for_percentage_return(1.0/6.5)
        assert 8 == self.dataset_generator._get_forward_period_for_percentage_return(2.0/6.5)
        assert 27 * 3 == self.dataset_generator._get_forward_period_for_percentage_return(3.0)