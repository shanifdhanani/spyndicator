from sklearn.tree import DecisionTreeRegressor
from numpy import random

from machine_learning.src.main.code.models.base_model import BaseModel
from machine_learning.src.main.code.models.model_types import ModelTypes


class DecisionTreeRegressionModel(BaseModel):

    ModelType = ModelTypes.DecisionTreeRegression

    def __init__(self, pipeline = None):

        super().__init__(pipeline = pipeline)

        self.model = DecisionTreeRegressor(max_depth = 50, random_state = random.seed(1234))
