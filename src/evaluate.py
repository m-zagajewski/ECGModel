"""
Evaluation script for the multimodal mortality prediction model.

This script handles model evaluation on test data and visualization of feature importance.
"""

import os
import argparse
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_curve, precision_recall_curve, auc, confusion_matrix
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, accuracy_score

from src.data.ecg_processor import ECGDataset
from src.data.tabular_processor import TabularDataset, CombinedDataset
from src.models.fusion_model import MultimodalMortalityPredictor


def load_model(model_path, config, device):
    """
    Load a trained model.

    Args:
        model_path (str): Path to the saved model
        config (dict): Model configuration
        device (str): Device to use for inference

    Returns:
        nn.Module: Loaded model
    """
    # Create model
    model = MultimodalMortalityPredictor(
        ecg_config=config['model']['ecg_config'],
        tabular_config=config['model']['tabular_config'],
        fusion_config=config['model']['fusion_config']
    ).to(device)

    # Load model weights
    model.load_state_dict(torch.load(model_path, map_location=device))

    # Set model to evaluation mode
    model.eval()

    return model


def evaluate_model(model, dataloader, device):
    """
    Evaluate the model on a dataset.

    Args:
        model (nn.Module): Model to evaluate
        dataloader (DataLoader): DataLoader for evaluation data
        device (str): Device to use for inference

    Returns:
        tuple: (predictions, targets, feature_importance)
            - predictions (dict): Dictionary containing predicted probabilities and binary predictions
            - targets (numpy.ndarray): Ground truth labels
            - feature_importance (dict): Dictionary containing feature importance weights (if available)
    """
    model.eval()
    all_probs = []
    all_targets = []
    all_patient_ids = []
    all_feature_importance = []

    with torch.no_grad():
        for batch in dataloader:
            # Get data
            ecg = batch['ecg'].to(device)
            tabular = batch['tabular'].to(device)
            targets = batch['target'].to(device)
            patient_ids = batch['patient_id']

            # Forward pass
            outputs = model.predict_proba(ecg, tabular)

            # Handle different output types (with or without feature importance)
            if isinstance(outputs, tuple):
                probs, feature_importance = outputs
                all_feature_importance.append(feature_importance)
            else:
                probs = outputs

            # Store predictions and targets
            all_probs.extend(probs.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            all_patient_ids.extend(patient_ids)

    # Convert to numpy arrays
    all_probs = np.array(all_probs).flatten()
    all_targets = np.array(all_targets)

    # Create binary predictions
    threshold = 0.5
    all_preds = (all_probs > threshold).astype(int)

    # Combine predictions
    predictions = {
        'probabilities': all_probs,
        'predictions': all_preds,
        'patient_ids': all_patient_ids
    }

    # Combine feature importance if available
    if all_feature_importance:
        combined_importance = {}
        for key in all_feature_importance[0].keys():
            combined_importance[key] = torch.cat([imp[key] for imp in all_feature_importance], dim=0).cpu().numpy()
        return predictions, all_targets, combined_importance
    else:
        return predictions, all_targets, None


def calculate_metrics(predictions, targets):
    """
    Calculate evaluation metrics.

    Args:
        predictions (dict): Dictionary containing predicted probabilities and binary predictions
        targets (numpy.ndarray): Ground truth labels

    Returns:
        dict: Dictionary containing evaluation metrics
    """
    probs = predictions['probabilities']
    preds = predictions['predictions']

    # Calculate metrics
    accuracy = accuracy_score(targets, preds)
    auroc = roc_auc_score(targets, probs)
    f1 = f1_score(targets, preds)
    precision = precision_score(targets, preds)
    recall = recall_score(targets, preds)

    # Create confusion matrix
    tn, fp, fn, tp = confusion_matrix(targets, preds).ravel()
    specificity = tn / (tn + fp)

    # Create metrics dictionary
    metrics = {
        'accuracy': accuracy,
        'auroc': auroc,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'specificity': specificity
    }

    return metrics


def plot_roc_curve(targets, probabilities, output_dir):
    """
    Plot ROC curve.

    Args:
        targets (numpy.ndarray): Ground truth labels
        probabilities (numpy.ndarray): Predicted probabilities
        output_dir (str): Directory to save the plot
    """
    # Calculate ROC curve
    fpr, tpr, _ = roc_curve(targets, probabilities)
    roc_auc = auc(fpr, tpr)

    # Plot ROC curve
    plt.figure(figsize=(8, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')

    # Save plot
    plt.savefig(os.path.join(output_dir, 'roc_curve.png'), dpi=300, bbox_inches='tight')
    plt.close()


def plot_precision_recall_curve(targets, probabilities, output_dir):
    """
    Plot precision-recall curve.

    Args:
        targets (numpy.ndarray): Ground truth labels
        probabilities (numpy.ndarray): Predicted probabilities
        output_dir (str): Directory to save the plot
    """
    # Calculate precision-recall curve
    precision, recall, _ = precision_recall_curve(targets, probabilities)
    pr_auc = auc(recall, precision)

    # Plot precision-recall curve
    plt.figure(figsize=(8, 8))
    plt.plot(recall, precision, color='darkorange', lw=2, label=f'PR curve (AUC = {pr_auc:.3f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc='lower left')

    # Save plot
    plt.savefig(os.path.join(output_dir, 'precision_recall_curve.png'), dpi=300, bbox_inches='tight')
    plt.close()


def plot_confusion_matrix(targets, predictions, output_dir):
    """
    Plot confusion matrix.

    Args:
        targets (numpy.ndarray): Ground truth labels
        predictions (numpy.ndarray): Binary predictions
        output_dir (str): Directory to save the plot
    """
    # Calculate confusion matrix
    cm = confusion_matrix(targets, predictions)

    # Plot confusion matrix
    plt.figure(figsize=(8, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.xticks([0.5, 1.5], ['Survived', 'Died'])
    plt.yticks([0.5, 1.5], ['Survived', 'Died'])

    # Save plot
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
    plt.close()


def plot_tabular_feature_importance(feature_importance, feature_names, output_dir):
    """
    Plot tabular feature importance.

    Args:
        feature_importance (numpy.ndarray): Feature importance weights
        feature_names (list): Names of the features
        output_dir (str): Directory to save the plot
    """
    # Calculate mean feature importance across samples
    mean_importance = np.mean(feature_importance, axis=0)

    # Sort features by importance
    sorted_idx = np.argsort(mean_importance)
    sorted_names = [feature_names[i] for i in sorted_idx]
    sorted_importance = mean_importance[sorted_idx]

    # Plot feature importance
    plt.figure(figsize=(10, 8))
    plt.barh(range(len(sorted_names)), sorted_importance, align='center')
    plt.yticks(range(len(sorted_names)), sorted_names)
    plt.xlabel('Mean Feature Importance')
    plt.title('Tabular Feature Importance')

    # Save plot
    plt.savefig(os.path.join(output_dir, 'tabular_feature_importance.png'), dpi=300, bbox_inches='tight')
    plt.close()


def plot_modality_importance(ecg_weight, tabular_weight, output_dir):
    """
    Plot modality importance.

    Args:
        ecg_weight (numpy.ndarray): ECG modality weights
        tabular_weight (numpy.ndarray): Tabular modality weights
        output_dir (str): Directory to save the plot
    """
    # Calculate mean modality importance across samples
    mean_ecg_weight = np.mean(ecg_weight)
    mean_tabular_weight = np.mean(tabular_weight)

    # Plot modality importance
    plt.figure(figsize=(8, 6))
    plt.bar(['ECG', 'Tabular'], [mean_ecg_weight, mean_tabular_weight])
    plt.ylabel('Mean Modality Weight')
    plt.title('Modality Importance')

    # Save plot
    plt.savefig(os.path.join(output_dir, 'modality_importance.png'), dpi=300, bbox_inches='tight')
    plt.close()


def save_predictions(predictions, targets, patient_ids, output_dir):
    """
    Save predictions to a CSV file.

    Args:
        predictions (dict): Dictionary containing predicted probabilities and binary predictions
        targets (numpy.ndarray): Ground truth labels
        patient_ids (list): List of patient IDs
        output_dir (str): Directory to save the CSV file
    """
    # Create DataFrame
    df = pd.DataFrame({
        'patient_id': patient_ids,
        'true_label': targets,
        'predicted_probability': predictions['probabilities'],
        'predicted_label': predictions['predictions']
    })

    # Save to CSV
    df.to_csv(os.path.join(output_dir, 'predictions.csv'), index=False)


def main():
    """Main function to run the evaluation script."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Evaluate multimodal mortality prediction model')
    parser.add_argument('--config', type=str, required=True, help='Path to configuration file')
    parser.add_argument('--model', type=str, required=True, help='Path to trained model')
    parser.add_argument('--output', type=str, required=True, help='Directory to save evaluation results')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use for inference')
    args = parser.parse_args()

    # Load configuration
    with open(args.config, 'r') as f:
        config = json.load(f)

    # Check if CUDA is available
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA is not available. Using CPU instead.")
        device = 'cpu'
    else:
        device = args.device

    # Create output directory if it doesn't exist
    os.makedirs(args.output, exist_ok=True)

    # Load tabular data to get feature names
    tabular_data = pd.read_excel(config['tabular_data_file'])

    # Create datasets
    ecg_dataset = ECGDataset(config['ecg_data_dir'], verbose=False)
    tabular_dataset = TabularDataset(config['tabular_data_file'])
    combined_dataset = CombinedDataset(ecg_dataset, tabular_dataset)

    # Print summary of NaN values in ECG data
    print("\nNaN values in ECG data:")
    ecg_dataset.print_nan_summary()

    # Create data loader
    dataloader = DataLoader(combined_dataset, batch_size=config['batch_size'])

    # Update tabular input dimension in config
    tabular_dim = tabular_dataset.get_feature_dimensions()
    config['model']['tabular_config']['input_dim'] = tabular_dim

    # Load model
    model = load_model(args.model, config, device)

    # Evaluate model
    predictions, targets, feature_importance = evaluate_model(model, dataloader, device)

    # Calculate metrics
    metrics = calculate_metrics(predictions, targets)

    # Print metrics
    print("Evaluation Metrics:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")

    # Save metrics
    with open(os.path.join(args.output, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=4)

    # Plot ROC curve
    plot_roc_curve(targets, predictions['probabilities'], args.output)

    # Plot precision-recall curve
    plot_precision_recall_curve(targets, predictions['probabilities'], args.output)

    # Plot confusion matrix
    plot_confusion_matrix(targets, predictions['predictions'], args.output)

    # Plot feature importance if available
    if feature_importance is not None:
        # Get feature names
        feature_names = tabular_dataset.get_feature_names()

        # Plot tabular feature importance
        plot_tabular_feature_importance(
            feature_importance['tabular_feature_importance'],
            feature_names,
            args.output
        )

        # Plot modality importance
        plot_modality_importance(
            feature_importance['ecg_weight'],
            feature_importance['tabular_weight'],
            args.output
        )

    # Save predictions
    save_predictions(
        predictions,
        targets,
        predictions['patient_ids'],
        args.output
    )

    print(f"Evaluation results saved to {args.output}")


if __name__ == '__main__':
    main()
