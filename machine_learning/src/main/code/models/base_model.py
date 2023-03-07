import time
import warnings

import numpy as np
import pandas as pd
import shap
from scipy.stats import linregress
from sklearn import metrics
from sklearn.utils.validation import DataConversionWarning

from machine_learning.src.main.code.model_evaluation_metrics.model_evaluation_metrics_collection import ModelEvaluationMetricsCollection
from machine_learning.src.main.code.pipeline.pipeline import Pipeline
from machine_learning.src.main.code.pipeline.transformers.data_imputer import DataImputer
from machine_learning.src.main.code.pipeline.transformers.dataset_records_selector import DatasetRecordsSelector
from machine_learning.src.main.code.pipeline.transformers.one_hot_encoder import OneHotEncoder
from machine_learning.src.main.code.pipeline.transformers.outlier_robust_scaler import OutlierRobustScaler
from utils.src.main.code.logging.logger import Logger


class BaseModel:
    """
    A Base Model class extended by all their model classes
    """

    AbsoluteImpactOnPredictionKey = 'absoluteImpactOnPrediction'
    FeatureImportanceForPredictionKey = 'featureImportanceForPrediction'

    logger = Logger.get_logger(__name__)

    def __init__(self, pipeline = None):
        """
        Initialize the model

        :param pipeline: (obj:`Pipeline`) A Pipeline object used in dataset transformation
        """

        warnings.simplefilter('ignore', DataConversionWarning)

        self.model = None   # Subclasses must set this
        self.predictions_explainer = None
        self.pipeline = pipeline
        if pipeline is None:
            self._create_default_pipeline()
        self.predictions_explainer = None
        self.transformed_dataset_columns = None
        self.model_evaluation_metrics = ModelEvaluationMetricsCollection()

    def train(self, training_dataset: pd.DataFrame, training_labels, should_transform = True, should_add_explanations = False):
        """
        This is the base "train" method. It expects train and test datasets and labels and
        uses them to train the underlying learning algorithm.

        :param training_dataset: (obj:`pd.DataFrame`): A pd.DataFrame object consisting of training data
        :param training_labels: (obj:`list`): A list of labels ordered respective to the training dataset
        :param should_transform: (bool): Whether to transform the dataset using this model's pipeline
        """

        assert self.model is not None

        if training_dataset is None or not isinstance(training_dataset, pd.DataFrame):
            raise Exception("Please provide correctly formatted training dataset")

        if training_labels is None:
            raise Exception("Please provide correctly formatted training labels")

        transformation_start_time = time.time()
        if should_transform:
            training_dataset, training_labels = self.pipeline.transform(training_dataset, training_labels)
            if training_dataset is None:
                raise Exception("There was an error transforming the training dataset")
        transformation_end_time = time.time()

        self.model_evaluation_metrics.transformation_time = transformation_end_time - transformation_start_time
        self.model_evaluation_metrics.num_train_records = len(training_labels)

        training_start_time = time.time()
        self.model.fit(training_dataset, training_labels)
        training_end_time = time.time()
        training_time = training_end_time - training_start_time
        self.transformed_dataset_columns = training_dataset.columns

        # Create and store an explainer
        if should_add_explanations:
            self.predictions_explainer = shap.KernelExplainer(self.model.predict, training_dataset)

        self.model_evaluation_metrics.training_time = training_time
        self.set_feature_importances(training_dataset_columns = training_dataset.columns)

    def set_feature_importances(self, training_dataset_columns):
        """
        Set a dict of feature_name: feature_importance if this was a tree model that contains feature importances
        """

        try:
            feature_importances = self.model.feature_importances_
            training_feature_name_to_feature_importance = self._create_mapping_of_feature_name_to_feature_importance(training_dataset_columns, feature_importances)
            self.feature_importances = training_feature_name_to_feature_importance
            self.model_evaluation_metrics.model_feature_importance = training_feature_name_to_feature_importance
        except:
            self.logger.error("Failed to set feature importances for model", exc_info = True)
            self.feature_importances = None

    def train_test_and_return_model_metrics(self, training_dataset, training_labels, test_dataset, test_labels, should_transform = True, should_add_explanation = False):
        """
        This method trains the model, evaluates it on the test dataset and return the model metrics

        :param training_dataset: (obj:`pd.DataFrame`): A pd.DataFrame object consisting of training data
        :param training_labels: (obj:`list`): A list of labels ordered respective to the training dataset
        :param test_dataset: (obj:`pd.DataFrame`): A pd.DataFrame for testing data
        :param test_labels: (obj:`list`): A list of labels ordered respective to the test dataset
        :param should_transform: (bool): Whether to transform the dataset using this model's pipeline
        """

        # Train the model
        self.train(training_dataset = training_dataset, training_labels = training_labels,
                   should_transform = should_transform, should_add_explanations = should_add_explanation)

        if test_dataset is None or not isinstance(test_dataset, pd.DataFrame):
            raise Exception("Please provide correctly formatted test dataset for model evaluation")

        if test_labels is None:
            raise Exception("Please provide correctly formatted test labels for model evaluation")

        # Evaluate the model and extract and return the metrics
        model_metrics = self.evaluate(dataset = test_dataset, labels = test_labels, should_transform = should_transform,
                                      should_add_explanation = should_add_explanation)

        return model_metrics

    # Classes that extend BaseModel can override this method
    def evaluate(self, dataset, labels, should_transform = True, should_add_explanation = False):
        """
        Evaluate the performance of the model on the provided dataset and labels

        :param dataset: (obj:`pd.DataFrame`): A pd.DataFrame dataset to generate predictions on
        :param labels: (obj:`list`): A list of labels ordered respective to the dataset
        :param should_transform: (bool): Whether or not to transform the dataset using this model's pipeline
        :param should_add_explanation (bool): Whether or not to add explanations to the predictions instance
        :return: (obj:`ModelEvaluationMetrics`) A ModelEvaluationMetrics object
        """

        evaluation_start_time = time.time()

        if should_transform:
            dataset, labels = self.pipeline.transform(dataset, labels)

        model_predictions, _ = self.predict(dataset, should_transform = False)
        self._evaluate_predictions(predictions = model_predictions, labels = labels)

        evaluation_end_time = time.time()
        evaluation_time = evaluation_end_time - evaluation_start_time
        self.model_evaluation_metrics.evaluation_time = evaluation_time

        return self.model_evaluation_metrics

    # Classes that extend BaseModel can overwrite this method
    def predict(self, dataset, should_transform = True, show_probability = False, get_explanations = False):
        """
        This takes data and feeds it to the model to produce predictions

        :param dataset: (obj:`pd.DataFrame`): A pd.DataFrame dataset to generate predictions on
        :param should_transform: (bool): Whether or not to transform the dataset using this model's pipeline
        """

        assert self.model is not None

        if should_transform:
            dataset, _ = self.pipeline.transform(dataset)

        model_predictions = self.model.predict(dataset)
        if not get_explanations:
            return model_predictions, None

        prediction_explanations = self.get_feature_importance_for_prediction(dataset)
        return model_predictions, prediction_explanations

    # Classes that extend BaseModel can override this method
    def get_feature_importance_for_prediction(self, prediction_dataset):
        """
        This functions takes in a prediction dataset and feeds it to the predictions explainer to explain the prediction

        :param prediction_dataset (pd.DataFrame):
        :return:
        """

        if self.predictions_explainer is not None and self.transformed_dataset_columns is not None:
            prediction_explanations = []
            shap_values = self.predictions_explainer.shap_values(prediction_dataset, nsamples = 25, l1_reg = 'aic')
            for prediction_index in range(len(prediction_dataset)):
                shap_value_for_prediction = abs(np.asarray(shap_values[prediction_index]))
                absolute_impact_mapping = self._create_mapping_of_feature_name_to_feature_importance(self.transformed_dataset_columns, shap_value_for_prediction)
                normalized_feature_importances = shap_value_for_prediction / shap_value_for_prediction.sum()
                feature_importance_mapping = self._create_mapping_of_feature_name_to_feature_importance(self.transformed_dataset_columns, normalized_feature_importances)
                prediction_explanations.append({
                    self.AbsoluteImpactOnPredictionKey: absolute_impact_mapping,
                    self.FeatureImportanceForPredictionKey: feature_importance_mapping
                })

            return prediction_explanations

        return None

    # Classes that extend BaseModel can override this method
    def _evaluate_predictions(self, predictions, labels):
        """
        Evaluate the predictions and the labels.

        :param predictions (obj:`list<obj>`): A list of predictions
        :param labels (obj:`list<obj>`): A list of labels
        :return: (obj:`ModelEvaluationMetrics`) A ModelEvaluationMetrics object
        """

        try:
            self.model_evaluation_metrics.mean_absolute_error = metrics.mean_absolute_error(y_pred = predictions, y_true = labels)
        except:
            pass
        try:
            self.model_evaluation_metrics.mean_squared_error = metrics.mean_squared_error(y_pred = predictions, y_true = labels)
        except:
            pass
        try:
            self.model_evaluation_metrics.root_mean_squared_error = np.sqrt(self.model_evaluation_metrics.mean_squared_error)
        except:
            pass
        try:
            self.model_evaluation_metrics.predictions_standard_deviation = np.std(predictions)
        except:
            pass
        try:
            self.model_evaluation_metrics.labels_standard_deviation = np.std(labels)
        except:
            pass
        try:
            self.model_evaluation_metrics.num_test_records = len(predictions)
        except:
            pass
        try:
            self.model_evaluation_metrics.r_squared = metrics.r2_score(y_true = labels, y_pred = predictions)
        except:
            self.logger.error("Failed to extract r_squared", exc_info = True)
            pass
        try:
            self.model_evaluation_metrics.median_absolute_error = metrics.median_absolute_error(y_true = labels, y_pred = predictions)
        except:
            self.logger.error("Failed to extract median_absolute_error", exc_info = True)
            pass
        try:
            self.model_evaluation_metrics.mean_squared_log_error = metrics.mean_squared_log_error(y_true = labels, y_pred = predictions)
        except ValueError as e:
            if str(e).index("cannot be used when targets contain negative values") >= 0:
                pass
            else:
                self.logger.error("Failed to extract mean_squared_log_error", exc_info = True)
        except:
            self.logger.error("Failed to extract mean_squared_log_error", exc_info = True)
            pass
        try:
            self.model_evaluation_metrics.max_error = metrics.max_error(y_true = labels, y_pred = predictions)
        except:
            self.logger.error("Failed to extract max_error", exc_info = True)
            pass
        try:
            self.model_evaluation_metrics.explained_variance_score = metrics.explained_variance_score(y_true = labels, y_pred = predictions)
        except:
            self.logger.error("Failed to extract explained_variance_score", exc_info = True)
            pass
        try:
            self.model_evaluation_metrics.mean_absolute_percentage_error = self._get_mean_or_median_absolute_percentage_error(y_true = labels, y_pred = predictions)
        except:
            self.logger.error("Failed to extract mean_absolute_percentage_error", exc_info=True)
            pass
        try:
            self.model_evaluation_metrics.median_absolute_percentage_error = self._get_mean_or_median_absolute_percentage_error(y_true = labels, y_pred = predictions, return_median = True)
        except:
            self.logger.error("Failed to extract median_absolute_percentage_error", exc_info=True)
            pass

        # Correlation b/w predictions and labels
        try:
            slope_of_regression_line, y_intercept, correlation_coefficient, p_value, stderr = linregress(labels, predictions)
            self.model_evaluation_metrics.correlation_coefficient = correlation_coefficient
            self.model_evaluation_metrics.correlation_p_value = p_value
        except:
            self.logger.error("Failed to extract correlation metrics", exc_info = True)
            pass

    # Classes that extend BaseModel can overwrite this method
    def _create_default_pipeline(self):
        self.pipeline = Pipeline(transformers = [
            DatasetRecordsSelector(),
            OneHotEncoder(),
            DataImputer(),
            OutlierRobustScaler()
        ])

    def _get_mean_or_median_absolute_percentage_error(self, y_true, y_pred, return_median = False):
        """
        Get the mean absolute percentage error

        :param y_true: array-like of shape = (n_samples) or (n_samples, n_outputs). Ground truth (correct) target values.
        :param y_pred: array-like of shape = (n_samples) or (n_samples, n_outputs). Estimated target values.
        :return (float): The mean absolute percentage error (MAPE)
        """

        try:
            y_true, y_pred = np.array(y_true), np.array(y_pred)
            if return_median:
                return np.median(np.abs((y_true - y_pred) / y_true)) * 100
            else:
                return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        except:
            return None

    def _create_mapping_of_feature_name_to_feature_importance(self, transformed_feature_names, feature_importances):
        """
        Take lists of transformed_feature_names and feature importance and return a dict mapping from training
        feature name to feature importance

        :param feature_names: (list<str>)
        :param feature_importances: list<float>
        :return: (dict)
        """

        try:
            training_feature_name_to_feature_importance = dict()
            for training_column_name, importance in zip(list(transformed_feature_names), feature_importances):
                for feature_name in self.pipeline.ordered_features:
                    if feature_name in training_column_name:
                        if feature_name not in training_feature_name_to_feature_importance:
                            training_feature_name_to_feature_importance[feature_name] = []
                        training_feature_name_to_feature_importance[feature_name].append(importance)

            for key, value in training_feature_name_to_feature_importance.items():
                try:
                    training_feature_name_to_feature_importance[key] = np.sum(value)
                except:
                    self.logger.error("An issue in parsing feature importance", exc_info=True)
                    training_feature_name_to_feature_importance[key] = np.nan

            return training_feature_name_to_feature_importance
        except:
            self.logger.error("Failed to create feature importance mapping")
            return None
