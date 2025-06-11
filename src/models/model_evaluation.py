import pandas as pd
import numpy as np
from typing import List, Dict, Any
from sklearn.model_selection import train_test_split
from src.models.base_model import ECGBaseModel

class ModelEvaluator:
    def __init__(self, test_size: float = 0.2, random_state: int = 34):
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
            # CV dla roc_auc powinno również używać predict_proba, co jest obsługiwane przez pipeline w ECGBaseModel.evaluate
            cv_results = model.evaluate(X_train, y_train, cv=cv, scoring=scoring)
            
            # Get test set performance
            y_pred_labels = model.predict(X_test) # Etykiety klas dla metryk takich jak accuracy, f1
            
            test_score_value = np.nan # Domyślnie NaN, jeśli nie można obliczyć

            if scoring == 'roc_auc':
                try:
                    y_pred_probs = model.predict_proba(X_test)[:, 1] # Prawdopodobieństwa dla klasy pozytywnej
                    test_score_value = self._calculate_score(y_test, y_pred_probs, scoring)
                except AttributeError:
                    print(f"Ostrzeżenie: Model {model.__class__.__name__} nie wspiera predict_proba. Nie można obliczyć roc_auc na zbiorze testowym.")
                except Exception as e:
                    print(f"Błąd podczas obliczania roc_auc dla {model.__class__.__name__} na zbiorze testowym: {e}")
            else: # Dla innych metryk używamy etykiet
                test_score_value = self._calculate_score(y_test, y_pred_labels, scoring)
            
            results = {
                'model_name': model.__class__.__name__,
                'cv_mean_score': cv_results['mean_score'],
                'cv_std_score': cv_results['std_score'],
                'test_score': test_score_value,
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
            # y_pred powinny być etykietami
            return accuracy_score(y_true, y_pred)
        elif scoring == 'f1':
            # y_pred powinny być etykietami
            return f1_score(y_true, y_pred, zero_division=0)
        elif scoring == 'roc_auc':
            # y_pred powinny być prawdopodobieństwami dla klasy pozytywnej
            # Sprawdzenie, czy y_pred to prawdopodobieństwa (1D array)
            if y_pred.ndim != 1:
                # To nie powinno się zdarzyć, jeśli evaluate_models działa poprawnie
                raise ValueError(f"Dla roc_auc, y_pred musi być jednowymiarową tablicą prawdopodobieństw, otrzymano kształt {y_pred.shape}")
            return roc_auc_score(y_true, y_pred)
        else:
            raise ValueError(f"Scoring metric {scoring} not implemented")