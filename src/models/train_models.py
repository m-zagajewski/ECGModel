import pandas as pd
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
    import os
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
    data_path = os.path.join(project_root, 'data/selected_features.csv')
    
    print(f"Loading data from: {data_path}")
    if not os.path.exists(data_path):
        print(f"Error: Data file not found at {data_path}")
        print("Please ensure 'selected_features.csv' exists. You might need to run 'select_the_most_important_features.py' first.")
        return
        
    data = pd.read_csv(data_path)

    # Assuming the last column is the target variable
    if 'ZGON' not in data.columns:
        print("Error: Target column 'ZGON' not found in the data.")
        return
        
    y = data['ZGON']
    X = data.drop(columns=['ZGON'])
    
    print(f"Data loaded. X shape: {X.shape}, y shape: {y.shape}")
    print(f"Target variable distribution:\n{y.value_counts(normalize=True)}")

    # Define a list of random states for splitting the data
    # Using multiple random states gives a more robust evaluation
    random_states_for_evaluation = [42, 123, 789, 101, 202] 
    # random_states_for_evaluation = 42 # For a single run

    # Initialize models with different parameters
    # Note: random_state within models is for their internal stochastic processes (e.g., tree building)
    # The random_states_for_evaluation is for the train/test split.
    models = [
        RandomForestModel(n_estimators=100, max_depth=10, random_state=42),
        RandomForestModel(n_estimators=200, max_depth=None, random_state=42),
        GradientBoostingModel(n_estimators=100, learning_rate=0.1, random_state=42),
        GradientBoostingModel(n_estimators=200, learning_rate=0.05, max_depth=5, random_state=42),
        SVMModel(kernel='rbf', C=1.0, probability=True, random_state=42), # Ensure probability=True for roc_auc
        SVMModel(kernel='linear', C=0.5, probability=True, random_state=42),
        LogisticRegressionModel(C=0.5, penalty='l1', solver='saga', max_iter=2000, random_state=42), # saga for l1
        CatBoostModel(iterations=200, learning_rate=0.1, depth=6, random_seed=42, verbose=0),
        CatBoostModel(iterations=500, learning_rate=0.05, depth=8, random_seed=42, verbose=0),
        XGBoostModel(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42),
        XGBoostModel(n_estimators=200, learning_rate=0.05, max_depth=7, random_state=42)
    ]

    # Initialize the evaluator
    evaluator = ModelEvaluator(test_size=0.25) # Increased test_size slightly

    # Define scoring metric
    # Options: 'accuracy', 'f1', 'roc_auc', 'precision', 'recall'
    # Note: For 'roc_auc', models should support predict_proba.
    # SVMModel needs probability=True.
    scoring_metric = 'f1'
    
    print(f"\nStarting model evaluation with scoring: {scoring_metric}")
    print(f"Using {len(random_states_for_evaluation)} random state(s) for train/test split: {random_states_for_evaluation}")

    # Evaluate all models
    results_summary = evaluator.evaluate_models(
        models=models,
        X=X,
        y=y,
        random_states=random_states_for_evaluation,
        cv=5, # 5-fold cross-validation on the training set
        scoring=scoring_metric
    )

    # Print the results summary
    evaluator.print_summary()

    # Save detailed run results and summary results to CSV
    detailed_results_output_path = os.path.join(project_root, f'model_evaluation_detailed_runs_{scoring_metric}.csv')
    evaluator.all_run_results.to_csv(detailed_results_output_path, index=False)
    print(f"\nDetailed run results saved to: {detailed_results_output_path}")
    
    summary_output_path = os.path.join(project_root, f'model_evaluation_summary_{scoring_metric}.csv')
    results_summary.to_csv(summary_output_path, index=False)
    print(f"Aggregated summary results saved to: {summary_output_path}")

if __name__ == "__main__":
    main()
