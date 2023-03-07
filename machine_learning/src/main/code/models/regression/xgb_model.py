import xgboost as xgb

from machine_learning.src.main.code.pipeline.pipeline import Pipeline
from machine_learning.src.main.code.models.base_model import BaseModel


class XgbModel(BaseModel):

    __slots__ = 'parameter_grid'

    def __init__(self, pipeline = None):
        """
        Initialize the XgbModelModel

        :param pipeline (obj:`Pipeline`): If provided - the transformation pipeline that the model should use
        """

        super().__init__(pipeline = pipeline)

        # TODO: The below is only needed once we start to optimize hyperparams
        self.parameter_grid = {
            'learning_rate': [.0001, 0.001, 0.01, 0.05, 0.1, 0.5],
            'max_depth': [2, 5, 10, 15, 20, 30, 40, 50],
            'n_estimators': [100, 200, 300, 400, 500],
            'gamma': [0, 0.1],
            'reg_lambda': [1, 0.1]
        }
        self.model = xgb.XGBRegressor()
        if pipeline is not None and isinstance(pipeline, Pipeline):
            self.pipeline = pipeline
