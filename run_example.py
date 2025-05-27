#!/usr/bin/env python
"""
Example script to demonstrate the full pipeline of the ECG and Clinical Data Mortality Prediction Model.

This script:
1. Checks if the data exists
2. Creates a simplified model configuration
3. Trains the model on a small subset of data
4. Evaluates the model
5. Prints the results

Usage:
    python run_example.py
"""

import os
import json
import shutil
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split

from src.data.ecg_processor import ECGDataset
from src.data.tabular_processor import TabularDataset, CombinedDataset
from src.models.fusion_model import MultimodalMortalityPredictor
from src.train import train_model
from src.evaluate import evaluate_model, calculate_metrics, plot_roc_curve

# Set paths
ecg_data_dir = "data/ecg"
tabular_data_file = "data/clinical_data.xlsx"
output_dir = "example_output"
config_file = "config/example_config.json"

# Create output directory
os.makedirs(output_dir, exist_ok=True)

# Check if data exists
print("Checking if data exists...")
if not os.path.exists(ecg_data_dir):
    print(f"Error: ECG data directory '{ecg_data_dir}' does not exist.")
    print("Please create the directory and add ECG JSON files.")
    exit(1)

if not os.path.exists(tabular_data_file):
    print(f"Error: Tabular data file '{tabular_data_file}' does not exist.")
    print("Please add the clinical data Excel file.")
    exit(1)

# Count ECG files
ecg_files = [f for f in os.listdir(ecg_data_dir) if f.endswith('.json')]
print(f"Found {len(ecg_files)} ECG files in '{ecg_data_dir}'")

# Create a simplified model configuration
print("Creating simplified model configuration...")
config = {
    "ecg_data_dir": ecg_data_dir,
    "tabular_data_file": tabular_data_file,
    "model_dir": output_dir,
    "batch_size": 8,
    "learning_rate": 0.0001,  # Reduced learning rate to avoid 'nan' loss
    "weight_decay": 1e-5,
    "num_epochs": 5,  # Reduced epochs for this example
    "patience": 3,
    "n_splits": 2,  # Reduced folds for this example
    "random_state": 42,
    "model": {
        "ecg_config": {
            "model_type": "cnn",  # Simpler model type
            "in_channels": 12,
            "output_dim": 64  # Reduced dimension
        },
        "tabular_config": {
            "model_type": "mlp",  # Simpler model type
            "input_dim": 20,  # Will be updated based on actual data
            "output_dim": 64,  # Reduced dimension
            "hidden_dims": [128, 64],  # Smaller hidden layers
            "dropout_rate": 0.3,
            "batch_norm": True
        },
        "fusion_config": {
            "model_type": "concatenation",  # Simpler fusion type
            "fusion_dim": 128,  # Reduced dimension
            "dropout_rate": 0.3
        }
    }
}

# Save configuration
with open(config_file, 'w') as f:
    json.dump(config, f, indent=4)
print(f"Saved simplified configuration to '{config_file}'")

# Create datasets
print("Creating datasets...")
try:
    # Create datasets
    ecg_dataset = ECGDataset(ecg_data_dir, verbose=False)
    tabular_dataset = TabularDataset(tabular_data_file)
    combined_dataset = CombinedDataset(ecg_dataset, tabular_dataset)

    # Print summary of NaN values in ECG data
    ecg_dataset.print_nan_summary()
    print(f"Combined dataset created with {len(combined_dataset)} samples")

    # Split dataset into train, validation, and test
    train_size = int(0.7 * len(combined_dataset))
    val_size = int(0.15 * len(combined_dataset))
    test_size = len(combined_dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(
        combined_dataset, [train_size, val_size, test_size]
    )

    # Create data loaders
    batch_size = config["batch_size"]
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # Get tabular input dimension
    tabular_dim = tabular_dataset.get_feature_dimensions()
    config["model"]["tabular_config"]["input_dim"] = tabular_dim
    print(f"Tabular feature dimension: {tabular_dim}")

    # Create model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    model = MultimodalMortalityPredictor(
        ecg_config=config["model"]["ecg_config"],
        tabular_config=config["model"]["tabular_config"],
        fusion_config=config["model"]["fusion_config"]
    ).to(device)

    # Create loss function and optimizer
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=config["learning_rate"], 
        weight_decay=config["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2
    )

    # Train model
    print("Training model...")
    model_save_path = os.path.join(output_dir, "example_model.pt")
    model, history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        num_epochs=config["num_epochs"],
        patience=config["patience"],
        model_save_path=model_save_path
    )

    # Plot training history
    print("Plotting training history...")
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')

    plt.subplot(1, 2, 2)
    val_auroc = [metrics['auroc'] for metrics in history['val_metrics']]
    val_f1 = [metrics['f1'] for metrics in history['val_metrics']]
    plt.plot(val_auroc, label='Validation AUROC')
    plt.plot(val_f1, label='Validation F1')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.legend()
    plt.title('Validation Metrics')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_history.png'))

    # Evaluate model
    print("Evaluating model...")
    predictions, targets, feature_importance = evaluate_model(model, test_loader, device)
    metrics = calculate_metrics(predictions, targets)

    # Print metrics
    print("\nEvaluation Metrics:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")

    # Save metrics
    with open(os.path.join(output_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=4)

    # Plot ROC curve
    plot_roc_curve(targets, predictions['probabilities'], output_dir)

    # Save predictions
    df = pd.DataFrame({
        'patient_id': predictions['patient_ids'],
        'true_label': targets,
        'predicted_probability': predictions['probabilities'],
        'predicted_label': predictions['predictions']
    })
    df.to_csv(os.path.join(output_dir, 'predictions.csv'), index=False)

    print(f"\nExample completed successfully! Results saved to '{output_dir}'")
    print("To view the results, check the following files:")
    print(f"  - {os.path.join(output_dir, 'training_history.png')}")
    print(f"  - {os.path.join(output_dir, 'metrics.json')}")
    print(f"  - {os.path.join(output_dir, 'roc_curve.png')}")
    print(f"  - {os.path.join(output_dir, 'predictions.csv')}")

except Exception as e:
    print(f"Error: {e}")
