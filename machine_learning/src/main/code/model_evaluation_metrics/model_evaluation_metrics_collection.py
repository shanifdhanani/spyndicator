from dataclasses import dataclass
import numpy as np
from copy import deepcopy


@dataclass(frozen = False)
class ModelEvaluationMetricsCollection:
    """
    This is a dataclass used to store all kinds of model evaluation metrics
    """

    num_train_records: object = None
    num_test_records: object = None
    mean_absolute_error: object = None
    mean_squared_error: object = None
    accuracy_score: object = None
    jaccard_score: object = None
    predictions_standard_deviation: object = None
    labels_standard_deviation: object = None
    log_loss: object = None
    f1_score: object = None
    confusion_matrix: object = None
    root_mean_squared_error: object = None
    training_time: object = None
    evaluation_time: object = None
    r_squared: object = None
    correlation_coefficient: object = None
    correlation_p_value: object = None
    median_absolute_error: object = None
    mean_squared_log_error: object = None
    max_error: object = None
    explained_variance_score: object = None
    mean_absolute_percentage_error: object = None
    median_absolute_percentage_error: object = None
    mean_percentage_error: object = None
    model_explainability: object = None
    model_feature_importance: object = None

    def get_in_json_format(self):
        data_dict = deepcopy(self.__dict__)
        for key, value in data_dict.items():
            try:
                if np.isnan(value):
                    data_dict[key] = None
                elif not np.isfinite(value):
                    data_dict[key] = None
                elif isinstance(value, np.ndarray):
                    data_dict[key] = value.tolist()
            except:
                continue

        return data_dict

    def get_metric_by_name(self, metric_name):
        return self.__dict__.get(metric_name, None)

