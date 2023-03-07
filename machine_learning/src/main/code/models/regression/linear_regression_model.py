from sklearn.linear_model import LinearRegression

from machine_learning.src.main.code.models.base_model import BaseModel
from machine_learning.src.main.code.models.model_types import ModelTypes


class LinearRegressionModel(BaseModel):

    ModelType = ModelTypes.LinearRegression

    __slots__ = ()

    def __init__(self, pipeline = None):
        """
        Initialize the LinearRegression model

        :param pipeline (obj:`Pipeline`): If provided - the transformation pipeline that the model should use
        """

        super().__init__(pipeline = pipeline)
        self.model = LinearRegression()
