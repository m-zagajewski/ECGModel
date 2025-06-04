import pandas as pd
import numpy as np
from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical
import os
from typing import Dict, Any, Type
from src.models.base_model import ECGBaseModel
from src.models.model_implementations import (
    RandomForestModel,
    GradientBoostingModel,
    SVMModel,
    LogisticRegressionModel,
    CatBoostModel,
    XGBoostModel
)
from sklearn.model_selection import StratifiedKFold

class ModelOptimizer:
    # Define search spaces for different models
    SEARCH_SPACES = {
        RandomForestModel: {
            'n_estimators': Integer(50, 300),
            'max_depth': Integer(5, 30),
            'min_samples_split': Integer(2, 10),
            'min_samples_leaf': Integer(1, 4),
            'max_features': Categorical(['sqrt', 'log2', None])
        },
        GradientBoostingModel: {
            'n_estimators': Integer(50, 300),
            'learning_rate': Real(0.001, 0.3, prior='log-uniform'),
            'max_depth': Integer(3, 10),
            'min_samples_split': Integer(2, 10),
            'subsample': Real(0.5, 1.0, prior='uniform')
        },
        SVMModel: {
            'C': Real(0.1, 10, prior='log-uniform'),
            'kernel': Categorical(['rbf', 'linear']),
            'gamma': Real(1e-4, 1, prior='log-uniform')
        },
        LogisticRegressionModel: {
            'C': Real(0.1, 10, prior='log-uniform'),
            'penalty': Categorical(['l1', 'l2']),
            'solver': Categorical(['liblinear', 'saga'])
        },
        CatBoostModel: {
            'iterations': Integer(200, 2000),
            'learning_rate': Real(0.005, 0.3, prior='log-uniform'),
            'depth': Integer(4, 10),
            # 'l2_leaf_reg': Real(1, 10, prior='log-uniform'),
            'verbose': Categorical([False])
        },
        XGBoostModel: {
            'n_estimators': Integer(50, 300),
            'learning_rate': Real(0.001, 0.3, prior='log-uniform'),
            'max_depth': Integer(3, 10),
            'min_child_weight': Integer(1, 10),
            'subsample': Real(0.5, 1.0, prior='uniform'),
            'colsample_bytree': Real(0.5, 1.0, prior='uniform')
        }
    }

    def __init__(self, 
                 model_class: Type[ECGBaseModel],
                 n_iter: int = 50,
                 cv: int = 5,
                 n_jobs: int = -1,
                 random_state: int = 42):
        self.model_class = model_class
        self.n_iter = n_iter
        self.cv = cv
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.best_params_ = None
        self.best_score_ = None
        self.search_results_ = None

    def optimize(self, X: pd.DataFrame, y: pd.Series, scoring: str = 'f1') -> Dict[str, Any]:
        """
        Perform Bayesian optimization to find the best hyperparameters.
        """
        # Get search space for the model
        search_space = self.SEARCH_SPACES.get(self.model_class)
        if search_space is None:
            raise ValueError(f"No search space defined for model class {self.model_class.__name__}")

        # Create base model instance
        base_model = self.model_class()

        # Configure the search
        opt = BayesSearchCV(
            base_model,
            search_space,
            n_iter=self.n_iter,
            cv=StratifiedKFold(n_splits=self.cv, shuffle=True, random_state=self.random_state),
            n_jobs=self.n_jobs,
            scoring=scoring,
            random_state=self.random_state,
            verbose=1
        )

        # Perform the search
        opt.fit(X, y)

        # Store results
        self.best_params_ = opt.best_params_
        self.best_score_ = opt.best_score_
        self.search_results_ = pd.DataFrame(opt.cv_results_)

        return {
            'best_params': self.best_params_,
            'best_score': self.best_score_,
            'all_results': self.search_results_
        }

    def get_best_model(self) -> ECGBaseModel:
        """
        Return a new model instance with the best found parameters.
        """
        if self.best_params_ is None:
            raise ValueError("Must run optimize() first to find best parameters")
        return self.model_class(**self.best_params_)


def main():
    # Load the data
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
    data_path = os.path.join(project_root, 'data/processed_features.csv')
    data = pd.read_csv(data_path)

    # Prepare the data
    y = data['ZGON']
    X = data.drop(columns=['ZGON'])

    # Dictionary mapping model names to classes for user selection
    MODEL_CLASSES = {
        'random_forest': RandomForestModel,
        'gradient_boosting': GradientBoostingModel,
        'svm': SVMModel,
        'logistic_regression': LogisticRegressionModel,
        'catboost': CatBoostModel,
        'xgboost': XGBoostModel
    }

    # Get user input for model selection
    print("\nAvailable models:")
    for name in MODEL_CLASSES.keys():
        print(f"- {name}")
    
    model_name = input("\nEnter the model name to optimize: ").lower()
        
    if model_name not in MODEL_CLASSES:
        raise ValueError(f"Invalid model name. Choose from: {', '.join(MODEL_CLASSES.keys())}")

    # Create optimizer
    optimizer = ModelOptimizer(
        model_class=MODEL_CLASSES[model_name],
        n_iter=50,  # Number of optimization iterations
        cv=5,       # Number of cross-validation folds
        n_jobs=-1   # Use all available CPU cores
    )

    # Run optimization
    print(f"\nOptimizing {model_name} model...")
    results = optimizer.optimize(X, y, scoring='f1')

    # Print results
    print("\nOptimization Results:")
    print("=" * 80)
    print(f"Best {model_name} parameters:")
    for param, value in results['best_params'].items():
        print(f"{param}: {value}")
    print(f"\nBest cross-validation score (F1): {results['best_score']:.4f}")

    # Save results
    output_dir = os.path.join(project_root, 'model_optimization_results')
    os.makedirs(output_dir, exist_ok=True)
    
    # Save detailed results to CSV
    results_path = os.path.join(output_dir, f'{model_name}_optimization_results.csv')
    results['all_results'].to_csv(results_path, index=False)
    print(f"\nDetailed results saved to: {results_path}")

    # Get the best model
    best_model = optimizer.get_best_model()
    
    # Train the best model on the full dataset
    print("\nTraining final model with best parameters...")
    best_model.fit(X, y)
    
    # Save the best model
    import joblib
    model_path = os.path.join(output_dir, f'{model_name}_best_model.joblib')
    joblib.dump(best_model, model_path)
    print(f"Best model saved to: {model_path}")

if __name__ == "__main__":
    main()