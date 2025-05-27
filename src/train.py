"""
Training script for the multimodal mortality prediction model.

This script handles the training pipeline, including data loading, model initialization,
loss function, optimization, and evaluation metrics.
"""

import os
import argparse
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
from sklearn.model_selection import KFold

from src.data.ecg_processor import ECGDataset
from src.data.tabular_processor import TabularDataset, CombinedDataset
from src.models.ecg_encoder import ECGEncoder, ResNetECGEncoder
from src.models.tabular_model import TabularMLP, TabularFeatureExtractor, TabularClassifier
from src.models.fusion_model import MultimodalMortalityPredictor


def train_epoch(model, dataloader, criterion, optimizer, device):
    """
    Train the model for one epoch.

    Args:
        model (nn.Module): Model to train
        dataloader (DataLoader): DataLoader for training data
        criterion (nn.Module): Loss function
        optimizer (optim.Optimizer): Optimizer
        device (torch.device): Device to use for training

    Returns:
        float: Average loss for the epoch
    """
    model.train()
    total_loss = 0.0

    for batch in dataloader:
        # Get data
        ecg = batch['ecg'].to(device)
        tabular = batch['tabular'].to(device)
        targets = batch['target'].to(device)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(ecg, tabular)

        # Handle different output types (with or without feature importance)
        if isinstance(outputs, tuple):
            logits, _ = outputs
        else:
            logits = outputs

        # Calculate loss
        loss = criterion(logits, targets.unsqueeze(1))

        # Backward pass and optimize
        loss.backward()
        # Apply gradient clipping to prevent 'nan' loss
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # Update total loss
        total_loss += loss.item() * ecg.size(0)

    # Calculate average loss
    avg_loss = total_loss / len(dataloader.dataset)

    return avg_loss


def evaluate(model, dataloader, criterion, device):
    """
    Evaluate the model.

    Args:
        model (nn.Module): Model to evaluate
        dataloader (DataLoader): DataLoader for evaluation data
        criterion (nn.Module): Loss function
        device (torch.device): Device to use for evaluation

    Returns:
        tuple: (average loss, metrics dictionary)
    """
    model.eval()
    total_loss = 0.0
    all_targets = []
    all_probs = []

    with torch.no_grad():
        for batch in dataloader:
            # Get data
            ecg = batch['ecg'].to(device)
            tabular = batch['tabular'].to(device)
            targets = batch['target'].to(device)

            # Forward pass
            outputs = model.predict_proba(ecg, tabular)

            # Handle different output types (with or without feature importance)
            if isinstance(outputs, tuple):
                probs, _ = outputs
            else:
                probs = outputs

            # Calculate loss
            loss = criterion(torch.log(probs + 1e-8), targets.unsqueeze(1))

            # Update total loss
            total_loss += loss.item() * ecg.size(0)

            # Store targets and probabilities for metric calculation
            all_targets.extend(targets.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    # Calculate average loss
    avg_loss = total_loss / len(dataloader.dataset)

    # Convert to numpy arrays
    all_targets = np.array(all_targets)
    all_probs = np.array(all_probs).flatten()

    # Calculate metrics
    all_preds = (all_probs > 0.5).astype(int)
    auroc = roc_auc_score(all_targets, all_probs)
    f1 = f1_score(all_targets, all_preds)
    precision = precision_score(all_targets, all_preds)
    recall = recall_score(all_targets, all_preds)

    # Create metrics dictionary
    metrics = {
        'loss': avg_loss,
        'auroc': auroc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

    return avg_loss, metrics


def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, 
               device, num_epochs=50, patience=10, model_save_path=None):
    """
    Train the model with early stopping.

    Args:
        model (nn.Module): Model to train
        train_loader (DataLoader): DataLoader for training data
        val_loader (DataLoader): DataLoader for validation data
        criterion (nn.Module): Loss function
        optimizer (optim.Optimizer): Optimizer
        scheduler (optim.lr_scheduler._LRScheduler): Learning rate scheduler
        device (torch.device): Device to use for training
        num_epochs (int, optional): Maximum number of epochs to train
        patience (int, optional): Number of epochs to wait for improvement before early stopping
        model_save_path (str, optional): Path to save the best model

    Returns:
        tuple: (trained model, training history)
    """
    # Initialize variables for early stopping
    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0

    # Initialize training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_metrics': []
    }

    # Training loop
    for epoch in range(num_epochs):
        # Train for one epoch
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)

        # Evaluate on validation set
        val_loss, val_metrics = evaluate(model, val_loader, criterion, device)

        # Update learning rate
        if scheduler is not None:
            scheduler.step(val_loss)

        # Update training history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_metrics'].append(val_metrics)

        # Print progress
        print(f"Epoch {epoch+1}/{num_epochs} - "
              f"Train Loss: {train_loss:.4f} - "
              f"Val Loss: {val_loss:.4f} - "
              f"Val AUROC: {val_metrics['auroc']:.4f} - "
              f"Val F1: {val_metrics['f1']:.4f}")

        # Check for improvement
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            patience_counter = 0

            # Save the best model
            if model_save_path is not None:
                torch.save(model.state_dict(), model_save_path)
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= patience:
            print(f"Early stopping after {epoch+1} epochs")
            break

    # Load the best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model, history


def cross_validation(ecg_data_dir, tabular_data_file, config, n_splits=5, 
                    random_state=42, device='cuda'):
    """
    Perform k-fold cross-validation.

    Args:
        ecg_data_dir (str): Directory containing ECG JSON files
        tabular_data_file (str): Path to the Excel file containing clinical data
        config (dict): Model configuration
        n_splits (int, optional): Number of folds for cross-validation
        random_state (int, optional): Random seed for reproducibility
        device (str, optional): Device to use for training ('cuda' or 'cpu')

    Returns:
        dict: Cross-validation results
    """
    # Set random seed
    torch.manual_seed(random_state)
    np.random.seed(random_state)

    # Load tabular data to get patient IDs
    tabular_data = pd.read_excel(tabular_data_file)
    patient_ids = tabular_data['patient_id'].values

    # Initialize k-fold cross-validation
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    # Initialize results
    cv_results = {
        'fold_metrics': [],
        'mean_metrics': {},
        'std_metrics': {}
    }

    # Cross-validation loop
    for fold, (train_idx, test_idx) in enumerate(kf.split(patient_ids)):
        print(f"\nFold {fold+1}/{n_splits}")

        # Get patient IDs for this fold
        train_patient_ids = patient_ids[train_idx]
        test_patient_ids = patient_ids[test_idx]

        # Create datasets
        ecg_train_dataset = ECGDataset(ecg_data_dir, patient_ids=train_patient_ids, verbose=False)
        ecg_test_dataset = ECGDataset(ecg_data_dir, patient_ids=test_patient_ids, verbose=False)

        # Print summary of NaN values in ECG data for the first fold only
        if fold == 0:
            print("\nNaN values in training ECG data:")
            ecg_train_dataset.print_nan_summary()
            print("\nNaN values in test ECG data:")
            ecg_test_dataset.print_nan_summary()

        tabular_train_dataset = TabularDataset(tabular_data_file, patient_ids=train_patient_ids)
        tabular_test_dataset = TabularDataset(tabular_data_file, patient_ids=test_patient_ids)

        # Create combined datasets
        train_dataset = CombinedDataset(ecg_train_dataset, tabular_train_dataset)
        test_dataset = CombinedDataset(ecg_test_dataset, tabular_test_dataset)

        # Split train dataset into train and validation
        train_size = int(0.8 * len(train_dataset))
        val_size = len(train_dataset) - train_size
        train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=config['batch_size'])
        test_loader = DataLoader(test_dataset, batch_size=config['batch_size'])

        # Update tabular input dimension in config
        tabular_dim = tabular_train_dataset.get_feature_dimensions()
        config['model']['tabular_config']['input_dim'] = tabular_dim

        # Create model
        model = MultimodalMortalityPredictor(
            ecg_config=config['model']['ecg_config'],
            tabular_config=config['model']['tabular_config'],
            fusion_config=config['model']['fusion_config']
        ).to(device)

        # Create loss function, optimizer, and scheduler
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'], 
                              weight_decay=config['weight_decay'])
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                                                       factor=0.5, patience=5)

        # Train model
        model_save_path = os.path.join(config['model_dir'], f"model_fold_{fold+1}.pt")
        model, history = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            num_epochs=config['num_epochs'],
            patience=config['patience'],
            model_save_path=model_save_path
        )

        # Evaluate on test set
        _, test_metrics = evaluate(model, test_loader, criterion, device)

        # Store results for this fold
        cv_results['fold_metrics'].append(test_metrics)

        print(f"Fold {fold+1} Test Metrics:")
        for metric, value in test_metrics.items():
            print(f"  {metric}: {value:.4f}")

    # Calculate mean and standard deviation of metrics across folds
    for metric in cv_results['fold_metrics'][0].keys():
        values = [fold_metrics[metric] for fold_metrics in cv_results['fold_metrics']]
        cv_results['mean_metrics'][metric] = np.mean(values)
        cv_results['std_metrics'][metric] = np.std(values)

    # Print overall results
    print("\nCross-Validation Results:")
    for metric, mean_value in cv_results['mean_metrics'].items():
        std_value = cv_results['std_metrics'][metric]
        print(f"  {metric}: {mean_value:.4f} ± {std_value:.4f}")

    return cv_results


def main():
    """Main function to run the training script."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Train multimodal mortality prediction model')
    parser.add_argument('--config', type=str, default='config/model_config.json', help='Path to configuration file (default: config/model_config.json)')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use for training')
    args = parser.parse_args()

    # Check if configuration file exists
    if not os.path.exists(args.config):
        print(f"Error: Configuration file '{args.config}' not found.")
        print("Please provide a valid configuration file path using the --config argument.")
        print("Example: python src/train.py --config config/model_config.json")
        exit(1)

    # Load configuration
    try:
        with open(args.config, 'r') as f:
            config = json.load(f)
    except json.JSONDecodeError:
        print(f"Error: Configuration file '{args.config}' is not a valid JSON file.")
        exit(1)

    # Check if CUDA is available
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA is not available. Using CPU instead.")
        device = 'cpu'
    else:
        device = args.device

    # Create model directory if it doesn't exist
    os.makedirs(config['model_dir'], exist_ok=True)

    # Perform cross-validation
    cv_results = cross_validation(
        ecg_data_dir=config['ecg_data_dir'],
        tabular_data_file=config['tabular_data_file'],
        config=config,
        n_splits=config['n_splits'],
        random_state=config['random_state'],
        device=device
    )

    # Save cross-validation results
    results_path = os.path.join(config['model_dir'], 'cv_results.json')
    with open(results_path, 'w') as f:
        json.dump(cv_results, f, indent=4)

    print(f"Cross-validation results saved to {results_path}")


if __name__ == '__main__':
    main()
