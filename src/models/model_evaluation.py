import pandas as pd
import numpy as np
from typing import List, Dict, Any
from sklearn.model_selection import train_test_split
from base_model import ECGBaseModel

class ModelEvaluator:
    def __init__(self, test_size: float = 0.2, random_state: int = 42):
        self.test_size = test_size
        self.random_state = random_state
        self.results = []
        
    def evaluate_models(self, models: List[ECGBaseModel], X: pd.DataFrame, y: pd.Series, 
                       cv: int = 5, scoring: str = 'accuracy') -> pd.DataFrame:
        """
        Evaluate multiple models and return their performance metrics
        """
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )
        
        all_results = []
        for model in models:
            # Fit the model
            model.fit(X_train, y_train)
            
            # Get cross-validation results
            cv_results = model.evaluate(X_train, y_train, cv=cv, scoring=scoring)
            
            # Get test set performance
            y_pred = model.predict(X_test)
            test_score = self._calculate_score(y_test, y_pred, scoring)
            
            results = {
                'model_name': model.__class__.__name__,
                'cv_mean_score': cv_results['mean_score'],
                'cv_std_score': cv_results['std_score'],
                'test_score': test_score,
                'scoring_metric': scoring
            }
            all_results.append(results)
        
        results_df = pd.DataFrame(all_results)
        self.results = results_df
        return results_df
    
    def print_summary(self):
        """Print a formatted summary of model performances"""
        if len(self.results) == 0:
            print("No evaluation results available. Run evaluate_models first.")
            return
        
        print("\nModel Evaluation Summary:")
        print("=" * 80)
        print(f"{'Model Name':<30} {'CV Score':<20} {'Test Score':<20}")
        print("-" * 80)
        
        for _, row in self.results.iterrows():
            cv_score = f"{row['cv_mean_score']:.4f} ± {row['cv_std_score']:.4f}"
            test_score = f"{row['test_score']:.4f}"
            print(f"{row['model_name']:<30} {cv_score:<20} {test_score:<20}")
        print("=" * 80)
    
    def _calculate_score(self, y_true: pd.Series, y_pred: np.ndarray, scoring: str) -> float:
        """Calculate the score based on the scoring metric"""
        from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
        
        if scoring == 'accuracy':
            return accuracy_score(y_true, y_pred)
        elif scoring == 'f1':
            return f1_score(y_true, y_pred)
        elif scoring == 'roc_auc':
            return roc_auc_score(y_true, y_pred)
        else:
            raise ValueError(f"Scoring metric {scoring} not implemented") 