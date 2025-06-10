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
    data_path = os.path.join(project_root, 'data/quickstart_input_data.csv')
    data = pd.read_csv(data_path)

    # Assuming the last column is the target variable
    y = data['ZGON']
    X = data.drop(columns=['ZGON'])

    # Initialize models with different parameters
    models = [
        RandomForestModel(),
        RandomForestModel(),
        GradientBoostingModel(),
        GradientBoostingModel(),
        SVMModel(),
        LogisticRegressionModel(),
        CatBoostModel(),
        XGBoostModel()
    ]

    # Initialize the evaluator
    evaluator = ModelEvaluator(test_size=0.2, random_state=42)

    # Evaluate all models
    results = evaluator.evaluate_models(
        models=models,
        X=X,
        y=y,
        cv=5,
        scoring='f1'  # You can change this to 'f1' or 'roc_auc' if needed
    )

    # Print the results summary
    evaluator.print_summary()

    # Save results to CSV
    output_path = os.path.join(project_root, 'model_comparison_results.csv')
    results.to_csv(output_path, index=False)

if __name__ == "__main__":
    main()