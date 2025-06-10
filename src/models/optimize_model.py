import pandas as pd
import numpy as np
from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical
import os
import json
from typing import Dict, Any, Type
from tqdm.auto import tqdm 
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
from skopt.callbacks import VerboseCallback

class ModelOptimizer:
    # Define search spaces for different models
    SEARCH_SPACES = {
        RandomForestModel: {
            'n_estimators': Integer(50, 500),
            'max_depth': Integer(5, 50),
            'min_samples_split': Integer(2, 20),
            'min_samples_leaf': Integer(1, 10),
            'max_features': Categorical(['sqrt', 'log2', None]),
            'class_weight': Categorical(['balanced', None])
        },
        GradientBoostingModel: {
            'n_estimators': Integer(50, 500),
            'learning_rate': Real(0.001, 0.3, prior='log-uniform'),
            'max_depth': Integer(2, 15),
            'min_samples_split': Integer(2, 20),
            'min_samples_leaf': Integer(1, 10),
            'subsample': Real(0.5, 1.0, prior='uniform'),
            'max_features': Categorical(['sqrt', 'log2', None])
        },
        SVMModel: {
            'C': Real(0.01, 100, prior='log-uniform'),
            'kernel': Categorical(['rbf', 'linear', 'poly', 'sigmoid']),
            'gamma': Real(1e-5, 10, prior='log-uniform'),
            'degree': Integer(2, 5),
            'coef0': Real(0.0, 10.0, prior='uniform'),
            'probability': Categorical([True])
        },
        LogisticRegressionModel: {
            'C': Real(0.01, 100, prior='log-uniform'),
            'penalty': Categorical(['l1', 'l2', 'elasticnet']),
            'solver': Categorical(['liblinear', 'saga']),
            'max_iter': Integer(1000, 5000), # Zwiększono zakres z (500, 2000)
            'class_weight': Categorical(['balanced', None]),
            'l1_ratio': Real(0.0, 1.0, prior='uniform')
        },
        CatBoostModel: {
            'iterations': Integer(100, 3000),
            'learning_rate': Real(0.001, 0.5, prior='log-uniform'),
            'depth': Integer(3, 12),
            'l2_leaf_reg': Real(0.1, 10, prior='log-uniform'),
            'random_strength': Real(0.1, 10, prior='log-uniform'),
            'bagging_temperature': Real(0.0, 10.0, prior='uniform'),
            'border_count': Integer(32, 255),
            'verbose': Categorical([False])
        },
        XGBoostModel: {
            'n_estimators': Integer(50, 1000),
            'learning_rate': Real(0.001, 0.5, prior='log-uniform'),
            'max_depth': Integer(2, 15),
            'min_child_weight': Integer(1, 20),
            'gamma': Real(0.0, 10.0, prior='uniform'),
            'subsample': Real(0.5, 1.0, prior='uniform'),
            'colsample_bytree': Real(0.3, 1.0, prior='uniform'),
            'reg_alpha': Real(1e-5, 10.0, prior='log-uniform'),
            'reg_lambda': Real(0.1, 10.0, prior='log-uniform')
        }
    }

    def __init__(self,
                 model_class: Type[ECGBaseModel],
                 n_iter: int = 100,
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
        search_space = self.SEARCH_SPACES.get(self.model_class)
        if search_space is None:
            raise ValueError(f"No search space defined for model class {self.model_class.__name__}")

        base_model = self.model_class()

        # Utworzenie paska postępu dla wewnętrznych iteracji BayesSearchCV
        # `leave=False` oznacza, że pasek zniknie po zakończeniu.
        inner_pbar = tqdm(total=self.n_iter, desc="Iteracje opt.", unit="iter", leave=False, position=1)

        def tqdm_skopt_callback(res):
            """Callback to update tqdm progress bar for skopt."""
            inner_pbar.update(1)
            return False # Kontynuuj optymalizację

        opt = BayesSearchCV(
            base_model,
            search_space,
            n_iter=self.n_iter,
            cv=StratifiedKFold(n_splits=self.cv, shuffle=True, random_state=self.random_state),
            n_jobs=self.n_jobs,
            scoring=scoring,
            random_state=self.random_state,
            verbose=0 # Wyłączamy standardowe logowanie skopt/sklearn
            # callbacks=[tqdm_skopt_callback] # Usunięto stąd
        )
        
        # Informacja przed rozpoczęciem fit, ale nie używamy print() bezpośrednio,
        # aby nie kolidowało z tqdm. tqdm.write() jest lepsze.
        # tqdm.write(f"Rozpoczynanie optymalizacji Bayesowskiej dla {self.model_class.__name__} ({self.n_iter} iteracji)...")
        
        try:
            opt.fit(X, y, callback=[tqdm_skopt_callback]) 
        finally:
            inner_pbar.close() 

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
    data_path = os.path.join(project_root, 'data/selected_features.csv')
    data = pd.read_csv(data_path)

    # Prepare the data
    y = data['ZGON']
    X = data.drop(columns=['ZGON'])

    # Dictionary mapping model names to classes for optimization
    MODEL_CLASSES = {
        #'random_forest': RandomForestModel,
        #'gradient_boosting': GradientBoostingModel,
        #'svm': SVMModel,
        #'logistic_regression': LogisticRegressionModel,
        #'catboost': CatBoostModel,
        'xgboost': XGBoostModel
    }

    output_dir = os.path.join(project_root, 'model_optimization_results')
    os.makedirs(output_dir, exist_ok=True)

    # Główny pasek postępu dla modeli, position=0 (domyślnie)
    # Użyj bar_format dla lepszej kontroli nad wyglądem i dodaj postfix
    main_pbar = tqdm(MODEL_CLASSES.items(), desc="Modele", unit="model", position=0, 
                     bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]')
    
    for model_name, model_class in main_pbar:
        main_pbar.set_postfix_str(f"Przetwarzanie: {model_name.upper()}", refresh=True)
        
        # Usunięto tqdm.write dla nagłówków modelu, aby uniknąć konfliktów z renderowaniem
        
        optimizer = ModelOptimizer(
            model_class=model_class,
            n_iter=100, 
            cv=5,
            n_jobs=-1
        )

        # Przekazanie X, y; model_name_for_tqdm nie jest już potrzebny w optimize, bo desc jest generyczny
        results = optimizer.optimize(X, y, scoring='f1') 

        # Wypisywanie wyników po zakończeniu optymalizacji danego modelu
        tqdm.write(f"\n--- Wyniki dla: {model_name.upper()} ---")
        tqdm.write(f"Najlepsze parametry:")
        for param, value in results['best_params'].items():
            tqdm.write(f"  {param}: {value}")
        tqdm.write(f"Najlepszy wynik CV (F1): {results['best_score']:.4f}")

        # Save detailed results to CSV
        results_path = os.path.join(output_dir, f'{model_name}_optimization_results.csv')
        results['all_results'].to_csv(results_path, index=False)
        tqdm.write(f"Szczegółowe wyniki zapisano do: {results_path}")

        # Save best parameters to JSON
        best_params_path = os.path.join(output_dir, f'{model_name}_best_params.json')
        with open(best_params_path, 'w') as f:
            json.dump(results['best_params'], f, indent=4)
        tqdm.write(f"Najlepsze parametry zapisano do: {best_params_path}")

        # Get the best model and train on the full dataset
        tqdm.write("Trenowanie finalnego modelu z najlepszymi parametrami...")
        best_model = optimizer.get_best_model()
        best_model.fit(X, y)

        # Save the best model
        import joblib
        model_path = os.path.join(output_dir, f'{model_name}_best_model.joblib')
        joblib.dump(best_model, model_path)
        tqdm.write(f"Najlepszy model zapisano do: {model_path}")
        tqdm.write("-" * 80) # Separator dla czytelności

    print("\nOptymalizacja wszystkich modeli zakończona.")


if __name__ == "__main__":
    main()