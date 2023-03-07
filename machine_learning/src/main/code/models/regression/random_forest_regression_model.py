from numpy import random
from sklearn.ensemble import RandomForestRegressor

from machine_learning.src.main.code.models.base_model import BaseModel
from machine_learning.src.main.code.models.model_types import ModelTypes


class RandomForestRegressionModel(BaseModel):

    ModelType = ModelTypes.RandomForestRegression

    __slots__ = 'n_estimators'

    def __init__(self, n_estimators = 50, criterion = "mae", max_depth = None, min_samples_split = 2,
                 min_samples_leaf = 1, min_weight_fraction_leaf = 0., max_features = "auto",
                 max_leaf_nodes = None, min_impurity_decrease = 0, bootstrap = True, oob_score = False,
                 n_jobs = 1, random_state = None, verbose = 0, warm_start = False, pipeline = None):

        super().__init__(pipeline = pipeline)

        self.model = RandomForestRegressor(n_estimators = n_estimators, verbose = True, random_state = random.seed(1234))
