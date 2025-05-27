import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

from src.data.ecg_processor import ECGDataset
from src.data.tabular_processor import TabularDataset, CombinedDataset
from src.models.fusion_model import MultimodalMortalityPredictor

# Set paths
ecg_data_dir = "data/ecg"
tabular_data_file = "data/clinical_data.xlsx"

# Check if data exists
print(f"ECG data directory exists: {os.path.exists(ecg_data_dir)}")
print(f"Tabular data file exists: {os.path.exists(tabular_data_file)}")

# Create datasets
try:
    print("Creating datasets...")
    ecg_dataset = ECGDataset(ecg_data_dir, verbose=False)

    # Print summary of NaN values in ECG data
    print("\nChecking for NaN values in ECG data:")
    ecg_dataset.print_nan_summary()

    tabular_dataset = TabularDataset(tabular_data_file)
    combined_dataset = CombinedDataset(ecg_dataset, tabular_dataset)
    print(f"Combined dataset created with {len(combined_dataset)} samples")

    # Split dataset into train and validation
    train_size = int(0.8 * len(combined_dataset))
    val_size = len(combined_dataset) - train_size
    train_dataset, val_dataset = random_split(combined_dataset, [train_size, val_size])

    # Create data loaders
    batch_size = 2  # Small batch size for testing
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # Get tabular input dimension
    tabular_dim = tabular_dataset.get_feature_dimensions()
    print(f"Tabular feature dimension: {tabular_dim}")

    # Create a small model for testing
    model_config = {
        'ecg_config': {
            'model_type': 'cnn',
            'in_channels': 12,
            'output_dim': 32
        },
        'tabular_config': {
            'model_type': 'mlp',
            'input_dim': tabular_dim,
            'output_dim': 32,
            'hidden_dims': [64, 32],
            'dropout_rate': 0.3,
            'batch_norm': True
        },
        'fusion_config': {
            'model_type': 'concatenation',
            'fusion_dim': 64,
            'dropout_rate': 0.3
        }
    }

    # Create model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    model = MultimodalMortalityPredictor(
        ecg_config=model_config['ecg_config'],
        tabular_config=model_config['tabular_config'],
        fusion_config=model_config['fusion_config']
    ).to(device)

    # Create loss function and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Try to train for one batch
    print("Training for one batch...")
    model.train()
    for batch in train_loader:
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

        print(f"Batch trained successfully! Loss: {loss.item()}")
        break  # Just train on one batch for testing

    print("Test completed successfully!")

except Exception as e:
    print(f"Error: {e}")
