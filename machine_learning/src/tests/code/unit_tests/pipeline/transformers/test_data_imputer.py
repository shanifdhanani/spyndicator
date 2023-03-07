from nose.tools import raises

from ml_core.src.main.code.pipeline.transformers.data_imputer import DataImputer
import pandas as pd
import numpy as np

from ml_core.src.main.code.pipeline.transformers.exceptions.columns_not_present_exception import ColumnsNotPresentException


class TestDataImputer:
    def setup(self):
        self.imputer = DataImputer()

    def test_we_impute_null_values(self):
        nanny_frame = pd.DataFrame({'nanny': [np.nan, 1, 2, 3, np.nan]})
        assert nanny_frame.isnull().values.any()
        results = self.imputer.transform(nanny_frame)
        assert not results.isnull().values.any()

    def test_columns_not_present_in_dataset_after_training_wont_mess_up_transformation(self):
        df = pd.DataFrame(
            {
                'col1': [1, 2, 3, 4],
                'col2': [10, 11, 12, 12],
                'col3': ['a', 'b', 'c', 'd']
            }
        )

        # This initial call will fit, as well as transform, the dataframe, it's needed so we can use the imputer on later data
        self.imputer.transform(df)

        bad_dataframe = pd.DataFrame(
            {
                'col1': [np.nan, np.nan, np.nan],
                'col2': [np.nan, 1, 13],
                'col3': ['a', 'b', 'c']
            }
        )
        transformed_dataframe = self.imputer.transform(bad_dataframe)
        assert len(transformed_dataframe.columns) == 3
        assert transformed_dataframe is not None

    def test_empty_columns_not_used_to_fit_imputer(self):
        bad_dataframe = pd.DataFrame(
            {
                'col1': [np.nan, np.nan, np.nan],
                'col2': [np.nan, 1, 13]
            }
        )
        new_dataframe = self.imputer.transform(bad_dataframe)
        assert len(new_dataframe.columns) == 1

        good_dataframe = pd.DataFrame(
            {
                'col1': [1, 3, 4],
                'col2': [9, 1, 13]
            }
        )
        updated_dataframe = self.imputer.transform(good_dataframe)
        assert len(updated_dataframe.columns) == 1

    def test_get_missing_columns_works(self):
        good_dataframe = pd.DataFrame(
            {
                'col1': [1, 2, 3],
                'col2': [54, 1, 13]
            }
        )
        self.imputer.transform(good_dataframe)

        bad_dataframe = pd.DataFrame(
            {
                'col1': [1, 3, 4]
            }
        )
        missing_columns = self.imputer._get_missing_columns(bad_dataframe)
        assert 'col2' in missing_columns

    @raises(ColumnsNotPresentException)
    def test_columns_needed_to_fit_are_reported_when_not_available(self):
        good_dataframe = pd.DataFrame(
            {
                'col1': [1, 2, 3],
                'col2': [54, 1, 13]
            }
        )
        new_dataframe = self.imputer.transform(good_dataframe)

        bad_dataframe = pd.DataFrame(
            {
                'col1': [1, 3, 4]
            }
        )
        self.imputer.transform(bad_dataframe)

    def test_get_extraneous_columns_works(self):
        training_dataframe = pd.DataFrame(
            {
                'col1': [1, 2, 3]
            }
        )
        self.imputer.transform(training_dataframe)

        updated_dataframe = pd.DataFrame(
            {
                'col1': [3, 5, 5],
                'col2': [1, 3, 4]
            }
        )
        extraneous_columns = self.imputer._get_extraneous_columns(updated_dataframe)
        assert 'col2' in extraneous_columns and len(extraneous_columns) == 1

    def test_fit_transform_works(self):
        training_dataframe = pd.DataFrame(
            {
                'col1': [1, 2.3, np.nan, np.nan, np.nan],
                'col2': [1, 3.4, np.nan, 1, 13]
            }
        )

        df = self.imputer.transform(training_dataframe)
        assert len(df) == len(training_dataframe)
        assert len(df.columns) == len(training_dataframe.columns)
