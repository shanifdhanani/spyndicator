from sklearn.linear_model import Ridge
from numpy import random

from machine_learning.src.main.code.pipeline.pipeline import Pipeline
from machine_learning.src.main.code.models.base_model import BaseModel


class RidgeModel(BaseModel):

    __slots__ = ()

    def __init__(self, pipeline = None):
        """
        Initialize the RidgeModel

        :param pipeline (obj:`Pipeline`): If provided - the transformation pipeline that the model should use
        """

        super().__init__(pipeline = pipeline)
        self.model = Ridge(random_state = random.seed(1234))
        if pipeline is not None and isinstance(pipeline, Pipeline):
            self.pipeline = pipeline
