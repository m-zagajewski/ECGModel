from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from catboost import CatBoostClassifier
import xgboost as xgb
import numpy as np
import pandas as pd
from src.models.base_model import ECGBaseModel

class RandomForestModel(ECGBaseModel):
    def __init__(self, n_estimators: int = 100, max_depth: int = None, 
                 min_samples_split: int = 2, min_samples_leaf: int = 1,
                 max_features: str = 'sqrt', random_state: int = 42):
        super().__init__()
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = random_state
        self.model = None
    
    def _fit_model(self, X: np.ndarray, y: pd.Series):
        self.model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features,
            random_state=self.random_state
        )
        self.model.fit(X, y)
    
    def _predict(self, X: np.ndarray):
        return self.model.predict(X)

    def _predict_proba(self, X: np.ndarray):
        return self.model.predict_proba(X)

class GradientBoostingModel(ECGBaseModel):
    def __init__(self, n_estimators: int = 100, learning_rate: float = 0.1,
                 max_depth: int = 3, min_samples_split: int = 2,
                 subsample: float = 1.0, random_state: int = 42):
        super().__init__()
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.subsample = subsample
        self.random_state = random_state
        self.model = None
    
    def _fit_model(self, X: np.ndarray, y: pd.Series):
        self.model = GradientBoostingClassifier(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            subsample=self.subsample,
            random_state=self.random_state
        )
        self.model.fit(X, y)
    
    def _predict(self, X: np.ndarray):
        return self.model.predict(X)

    def _predict_proba(self, X: np.ndarray):
        return self.model.predict_proba(X)

class SVMModel(ECGBaseModel):
    def __init__(self, C: float = 1.0, kernel: str = 'rbf', gamma: float = 'scale', probability: bool = True, random_state: int = 42):
        super().__init__()
        self.C = C
        self.kernel = kernel
        self.gamma = gamma
        self.probability = probability
        self.random_state = random_state
        self.model = None
    
    def _fit_model(self, X: np.ndarray, y: pd.Series):
        self.model = SVC(
            C=self.C,
            kernel=self.kernel,
            gamma=self.gamma,
            probability=self.probability,
            random_state=self.random_state
        )
        self.model.fit(X, y)
    
    def _predict(self, X: np.ndarray):
        return self.model.predict(X)

    def _predict_proba(self, X: np.ndarray):
        if not self.probability:
            raise AttributeError("predict_proba is not available when probability=False for SVMModel.")
        return self.model.predict_proba(X)

class LogisticRegressionModel(ECGBaseModel):
    def __init__(self, C: float = 1.0, penalty: str = 'l2', solver: str = 'liblinear', random_state: int = 42, max_iter: int = 1000):
        super().__init__()
        self.C = C
        self.penalty = penalty
        self.solver = solver
        self.random_state = random_state
        self.max_iter = max_iter
        self.model = None
    
    def _fit_model(self, X: np.ndarray, y: pd.Series):
        self.model = LogisticRegression(
            C=self.C,
            penalty=self.penalty,
            solver=self.solver,
            random_state=self.random_state,
            max_iter=self.max_iter
        )
        self.model.fit(X, y)
    
    def _predict(self, X: np.ndarray):
        return self.model.predict(X) 

    def _predict_proba(self, X: np.ndarray):
        return self.model.predict_proba(X)

class CatBoostModel(ECGBaseModel):
    def __init__(self, iterations: int = 1000, learning_rate: float = 0.1,
                 depth: int = 6, verbose: bool = False, random_seed: int = 42): # Zmieniono random_state na random_seed dla spójności z CatBoost
        super().__init__()
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.depth = depth
        self.verbose = verbose
        self.random_seed = random_seed # Użycie random_seed
        self.model = None

    def _fit_model(self, X: np.ndarray, y: pd.Series):
        self.model = CatBoostClassifier(
            iterations=self.iterations,
            learning_rate=self.learning_rate,
            depth=self.depth,
            verbose=self.verbose,
            random_seed=self.random_seed # Przekazanie random_seed
        )
        self.model.fit(X, y)

    def _predict(self, X: np.ndarray):
        return self.model.predict(X)
    
    def _predict_proba(self, X: np.ndarray):
        return self.model.predict_proba(X)
    
class XGBoostModel(ECGBaseModel):
    def __init__(self, 
                 n_estimators: int = 100,
                 learning_rate: float = 0.1,
                 max_depth: int = 3,
                 subsample: float = 1.0,
                 colsample_bytree: float = 1.0,
                 random_state: int = 42): # Dodano random_state
        super().__init__()
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.random_state = random_state # Zapisanie random_state
        self.model = None
    
    def _fit_model(self, X: np.ndarray, y: pd.Series):
        self.model = xgb.XGBClassifier(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
           # use_label_encoder=False,
            eval_metric='logloss',
            random_state=self.random_state # Przekazanie random_state
        )
        self.model.fit(X, y)
    
    def _predict(self, X: np.ndarray):
        return self.model.predict(X)

    def _predict_proba(self, X: np.ndarray):
        return self.model.predict_proba(X)