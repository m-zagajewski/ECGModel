import pandas as pd
import numpy as np
from typing import List, Dict, Any, Union
from sklearn.model_selection import train_test_split
from sklearn.base import clone
from src.models.base_model import ECGBaseModel

class ModelEvaluator:
    def __init__(self, test_size: float = 0.2):
        self.test_size = test_size
        self.results_summary = pd.DataFrame() # To store aggregated results
        self.all_run_results = pd.DataFrame() # To store results from all runs

    def evaluate_models(self, models: List[ECGBaseModel], X: pd.DataFrame, y: pd.Series,
                       random_states: Union[int, List[int]],
                       cv: int = 5, scoring: str = 'accuracy') -> pd.DataFrame:
        """
        Evaluate multiple models over one or more random states for train/test split
        and return their aggregated performance metrics.
        """
        if isinstance(random_states, int):
            random_states = [random_states]

        detailed_results_list = []

        for seed in random_states:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=self.test_size, random_state=seed, stratify=y
            )

            for model_prototype in models:
                # Clone the model to ensure a fresh state for each run
                model = clone(model_prototype)
                
                # Fit the model
                model.fit(X_train, y_train)
                
                # Get cross-validation results
                cv_results = model.evaluate(X_train, y_train, cv=cv, scoring=scoring)
                
                # Get test set performance
                if scoring == 'roc_auc':
                    if hasattr(model, 'predict_proba'):
                        y_pred_proba = model.predict_proba(X_test)
                        # Ensure y_pred_proba is 2D and take probabilities for the positive class
                        y_pred_for_score = y_pred_proba[:, 1] if y_pred_proba.ndim == 2 else y_pred_proba
                    else:
                        # Fallback or raise error if predict_proba is needed but not available
                        # For simplicity, we'll use predict, but this might not be ideal for AUC
                        print(f"Warning: Model {model.__class__.__name__} does not have predict_proba for roc_auc. Using predict().")
                        y_pred_for_score = model.predict(X_test)
                else:
                    y_pred_for_score = model.predict(X_test)
                
                test_score = self._calculate_score(y_test, y_pred_for_score, scoring)
                
                run_result = {
                    'model_name': model.__class__.__name__,
                    'random_state_run': seed,
                    'cv_mean_score': cv_results['mean_score'],
                    'cv_std_score': cv_results['std_score'],
                    'test_score': test_score,
                    'scoring_metric': scoring
                }
                detailed_results_list.append(run_result)
        
        self.all_run_results = pd.DataFrame(detailed_results_list)
        
        # Aggregate results
        if not self.all_run_results.empty:
            self.results_summary = self.all_run_results.groupby(['model_name', 'scoring_metric']).agg(
                mean_cv_score=('cv_mean_score', 'mean'),
                std_cv_score=('cv_mean_score', 'std'), # Std of the CV means from different runs
                mean_cv_std_dev=('cv_std_score', 'mean'), # Mean of the CV std_devs from different runs
                mean_test_score=('test_score', 'mean'),
                std_test_score=('test_score', 'std')
            ).reset_index()
        else:
            self.results_summary = pd.DataFrame()
            
        return self.results_summary

    def print_summary(self):
        """Print a formatted summary of aggregated model performances."""
        if self.results_summary.empty:
            print("No evaluation results available. Run evaluate_models first.")
            return
        
        print("\nModel Evaluation Summary (Aggregated over all random states):")
        print("=" * 120)
        header = f"{'Model Name':<30} {'Scoring Metric':<15} {'Mean CV Score':<20} {'Mean Test Score':<20} {'Std Test Score':<20}"
        print(header)
        print("-" * 120)
        
        for _, row in self.results_summary.iterrows():
            mean_cv_score_str = f"{row['mean_cv_score']:.4f} ± {row['std_cv_score']:.4f}" if pd.notna(row['std_cv_score']) else f"{row['mean_cv_score']:.4f}"
            mean_test_score_str = f"{row['mean_test_score']:.4f}"
            std_test_score_str = f"{row['std_test_score']:.4f}" if pd.notna(row['std_test_score']) else "N/A"
            
            print(f"{row['model_name']:<30} {row['scoring_metric']:<15} {mean_cv_score_str:<20} {mean_test_score_str:<20} {std_test_score_str:<20}")
        print("=" * 120)
        if len(self.all_run_results['random_state_run'].unique()) > 1:
            print(f"Number of random states used: {len(self.all_run_results['random_state_run'].unique())}")

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