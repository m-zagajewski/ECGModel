from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import numpy as np
import pandas as pd
from base_model import ECGBaseModel

class RandomForestModel(ECGBaseModel):
    def __init__(self, n_estimators: int = 100, max_depth: int = None, 
                 min_samples_split: int = 2, min_samples_leaf: int = 1,
                 max_features: str = 'sqrt'):
        super().__init__()
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.model = None
    
    def _fit_model(self, X: np.ndarray, y: pd.Series):
        self.model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features,
            random_state=42
        )
        self.model.fit(X, y)
    
    def _predict(self, X: np.ndarray):
        return self.model.predict(X)

class GradientBoostingModel(ECGBaseModel):
    def __init__(self, n_estimators: int = 100, learning_rate: float = 0.1,
                 max_depth: int = 3, min_samples_split: int = 2,
                 subsample: float = 1.0):
        super().__init__()
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.subsample = subsample
        self.model = None
    
    def _fit_model(self, X: np.ndarray, y: pd.Series):
        self.model = GradientBoostingClassifier(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            subsample=self.subsample,
            random_state=42
        )
        self.model.fit(X, y)
    
    def _predict(self, X: np.ndarray):
        return self.model.predict(X)

class SVMModel(ECGBaseModel):
    def __init__(self, C: float = 1.0, kernel: str = 'rbf', gamma: float = 'scale'):
        super().__init__()
        self.C = C
        self.kernel = kernel
        self.gamma = gamma
        self.model = None
    
    def _fit_model(self, X: np.ndarray, y: pd.Series):
        self.model = SVC(
            C=self.C,
            kernel=self.kernel,
            gamma=self.gamma,
            random_state=42
        )
        self.model.fit(X, y)
    
    def _predict(self, X: np.ndarray):
        return self.model.predict(X)

class LogisticRegressionModel(ECGBaseModel):
    def __init__(self, C: float = 1.0, penalty: str = 'l2', solver: str = 'liblinear'):
        super().__init__()
        self.C = C
        self.penalty = penalty
        self.solver = solver
        self.model = None
    
    def _fit_model(self, X: np.ndarray, y: pd.Series):
        self.model = LogisticRegression(
            C=self.C,
            penalty=self.penalty,
            solver=self.solver,
            random_state=42,
            max_iter=1000
        )
        self.model.fit(X, y)
    
    def _predict(self, X: np.ndarray):
        return self.model.predict(X) 