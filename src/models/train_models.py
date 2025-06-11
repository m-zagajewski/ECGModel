import pandas as pd
import os
import json
import sys
from src.models.model_implementations import (
    RandomForestModel,
    GradientBoostingModel,
    SVMModel,
    LogisticRegressionModel,
    CatBoostModel,
    XGBoostModel
)
from src.models.model_evaluation import ModelEvaluator

def main():
    # Load the preprocessed data
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
    data_path = os.path.join(project_root, 'data/selected_features.csv')
    
    # Jeśli istnieje plik quickstart_input_data.csv, użyj go zamiast selected_features.csv
    quickstart_data_path = os.path.join(project_root, 'data/quickstart_input_data.csv')
    if os.path.exists(quickstart_data_path):
        data_path = quickstart_data_path
        print(f"Używam danych z quickstart: {quickstart_data_path}")
    
    data = pd.read_csv(data_path)

    # Sprawdź, czy pierwsza kolumna to indeks i ustaw ją
    if data.columns[0].startswith('Unnamed') or data.columns[0].lower() in ['id', 'case_id', 'kg']:
        data.set_index(data.columns[0], inplace=True)

    # Assuming the last column is the target variable
    if 'ZGON' not in data.columns:
        print(f"Błąd: Kolumna 'ZGON' nie istnieje w danych z {data_path}")
        sys.exit(1)
    
    y = data['ZGON']
    X = data.drop(columns=['ZGON'])

    print(f"Wczytano dane o kształcie: {X.shape}")

    # Mapowanie nazw modeli do klas modeli
    MODEL_CLASSES = {
        'random_forest': RandomForestModel,
        'gradient_boosting': GradientBoostingModel,
        'svm': SVMModel,
        'logistic_regression': LogisticRegressionModel,
        'catboost': CatBoostModel,
        'xgboost': XGBoostModel
    }

    # Lista modeli do ewaluacji (wszystkie dostępne)
    model_names_to_evaluate = list(MODEL_CLASSES.keys())
    
    # Inicjalizacja pustej listy modeli
    models = []

    # Folder z zapisanymi zoptymalizowanymi parametrami
    optimization_dir = os.path.join(project_root, 'model_optimization_results')
    
    # Dla każdego modelu spróbuj wczytać zoptymalizowane parametry
    for model_name in model_names_to_evaluate:
        params_file = os.path.join(optimization_dir, f'{model_name}_best_params.json')
        
        model_class = MODEL_CLASSES[model_name]
        model_instance = None
        
        try:
            if os.path.exists(params_file):
                with open(params_file, 'r') as f:
                    params = json.load(f)
                print(f"Wczytano zoptymalizowane parametry dla {model_name}: {params}")
                model_instance = model_class(**params)
            else:
                print(f"Nie znaleziono pliku z parametrami dla {model_name}. Używam domyślnych parametrów.")
                # Domyślne parametry dla każdego modelu - bazujące na oryginalnych wartościach
                default_params = {
                    'random_forest': {'n_estimators': 100, 'max_depth': 10, 'random_state': 34}, # Zmieniono z 36 na 34
                    'gradient_boosting': {'n_estimators': 200, 'learning_rate': 0.05, 'random_state': 34},
                    'svm': {'kernel': 'rbf', 'C': 0.60, 'gamma': 'scale', 'probability': True, 'random_state': 34},
                    'logistic_regression': {'C': 1.0, 'penalty': 'l2', 'random_state': 34, 'max_iter': 1000},
                    'catboost': {'iterations': 1000, 'learning_rate': 0.1, 'depth': 6, 'random_seed': 34, 'verbose': False},
                    'xgboost': {'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 3, 'random_state': 34} # Zmieniono z 42 na 34
                }
                
                model_params = default_params.get(model_name, {})
                model_instance = model_class(**model_params)
        
            models.append(model_instance)
        except Exception as e:
            print(f"Błąd inicjalizacji modelu {model_name}: {e}")
    
    if not models:
        print("Błąd: Nie można było zainicjalizować żadnego modelu. Sprawdź parametry i pliki JSON.")
        sys.exit(1)
    
    print(f"Zainicjalizowano {len(models)} modeli do ewaluacji")

    # Initialize the evaluator
    evaluator = ModelEvaluator(test_size=0.2, random_state=34)

    # Evaluate all models
    results = evaluator.evaluate_models(
        models=models,
        X=X,
        y=y,
        cv=5,
        scoring='f1'  # You can change this to 'accuracy' or 'roc_auc' if needed
    )

    # Print the results summary
    evaluator.print_summary()

    # Save results to CSV
    output_path = os.path.join(project_root, 'model_comparison_results.csv')
    results.to_csv(output_path, index=False)
    print(f"Zapisano wyniki porównania modeli do: {output_path}")

if __name__ == "__main__":
    main()