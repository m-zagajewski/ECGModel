from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from typing import Optional, Dict, Any
from sklearn.pipeline import make_pipeline


class ECGBaseModel(ABC, BaseEstimator, ClassifierMixin):
    def __init__(self, scaler: Optional[BaseEstimator] = None):
        self.scaler = scaler if scaler is not None else StandardScaler()
        self.model = None
        
    def fit(self, X: pd.DataFrame, y: pd.Series):
        """Fit the model with the provided data."""
        X_scaled = self.scaler.fit_transform(X)
        self._fit_model(X_scaled, y)
        return self
        
    def predict(self, X: pd.DataFrame):
        """Make predictions on new data."""
        X_scaled = self.scaler.transform(X)
        return self._predict(X_scaled)
    
    def predict_proba(self, X: pd.DataFrame):
        """Make probability predictions on new data."""
        if not hasattr(self.model, 'predict_proba'):
            raise AttributeError(f"Model {self.model.__class__.__name__} does not support predict_proba.")
        X_scaled = self.scaler.transform(X)
        return self._predict_proba(X_scaled)

    def evaluate(self, X: pd.DataFrame, y: pd.Series, cv: int = 5, scoring: str = 'accuracy') -> Dict[str, Any]:
        """Evaluate model performance using cross-validation."""
        pipeline = make_pipeline(self.scaler, self.model)
        scores = cross_val_score(pipeline, X, y, cv=cv, scoring=scoring)
        
        results = {
            'model_name': self.__class__.__name__,
            'mean_score': np.mean(scores),
            'std_score': np.std(scores),
            'all_scores': scores,
            'scoring_metric': scoring
        }
        return results
        
    # Implementacja metody score wymaganej przez ClassifierMixin
    def score(self, X, y):
        """Return the accuracy score on the given test data and labels."""
        from sklearn.metrics import accuracy_score
        return accuracy_score(y, self.predict(X))
    
    @abstractmethod
    def _fit_model(self, X: np.ndarray, y: pd.Series):
        """Implementation specific model fitting."""
        pass
    
    @abstractmethod
    def _predict(self, X: np.ndarray):
        """Implementation specific prediction."""
        pass 

    @abstractmethod
    def _predict_proba(self, X: np.ndarray):
        """Implementation specific probability prediction."""
        pass
