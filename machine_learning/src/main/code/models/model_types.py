from enum import Enum


class ModelTypes(Enum):
    """
    All the different kinds of supported model types
    """

    Lasso = 'Lasso'
    LinearRegression = 'LinearRegression'
    RandomForestRegression = 'RandomForestRegression'
    StackingRegression = 'StackingRegression'
    Ridge = 'Ridge'
    Svm = 'Svm'
    XgBoost = 'XgBoost'
    RandomForestClassification = 'RandomForestClassification'
    LogisticRegressionClassification = 'LogisticRegressionClassification'
    MlpClassification = 'MlpClassification'
    AdaBoostClassification = 'AdaBoostClassification'
    NaiveBayesClassification = 'NaiveBayesClassification'
    DecisionTreeClassification = 'DecisionTreeClassification'
    SvcClassification = 'SvcClassification'
    StackingClassification = 'StackingClassification'
    GradientBoostingClassification = 'GradientBoostingClassification'
    DiscriminantAnalysisClassification = 'DiscriminantAnalysisClassification'
    GradientBoosting = 'GradientBoosting'
    SimpleNeuralNetworkRegression = 'SimpleNeuralNetworkRegression'
    DecisionTreeRegression = 'DecisionTreeRegression'

    # Segmentation Models
    TreeRegressionForSegmentationAnalysis = 'TreeRegressionForSegmentationAnalysis'
    TreeClassificationForSegmentationAnalysis = 'TreeClassificationForSegmentationAnalysis'
    TreeClassificationModelForChurnIndicators = 'TreeClassificationModelForChurnIndicators'

    def is_regression_model(self):
        return self in [self.Lasso, self.LinearRegression, self.RandomForestRegression, self.DecisionTreeRegression,
                        self.Svm, self.GradientBoosting, self.SimpleNeuralNetworkRegression, self.TreeRegressionForSegmentationAnalysis,
                        self.StackingRegression]

    def is_classification_model(self):
        return self in [self.RandomForestClassification, self.LogisticRegressionClassification, self.MlpClassification,
                        self.DecisionTreeClassification, self.SvcClassification, self.DiscriminantAnalysisClassification,
                        self.NaiveBayesClassification, self.AdaBoostClassification, self.TreeClassificationForSegmentationAnalysis,
                        self.StackingClassification, self.GradientBoostingClassification]
