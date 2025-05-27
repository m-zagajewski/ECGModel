"""
Module for loading and preprocessing tabular clinical data from Excel files.

This module provides functionality to:
1. Load tabular data from Excel files
2. Preprocess clinical features (normalization, encoding, handling missing values)
3. Create PyTorch datasets for tabular data
"""

import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from src.data.ecg_processor import ECGDataset


class TabularDataset(Dataset):
    """Dataset class for tabular clinical data."""

    def __init__(self, data_file, patient_ids=None, target_col='ZGON wewnątrzszpitalnie', 
                transform=None, categorical_cols=None, numerical_cols=None):
        """
        Initialize the tabular dataset.

        Args:
            data_file (str): Path to the Excel file containing clinical data
            patient_ids (list, optional): List of patient IDs to include. If None, include all.
            target_col (str, optional): Name of the target column (mortality within 30 days)
            transform (callable, optional): Optional transform to be applied on a sample
            categorical_cols (list, optional): List of categorical column names
            numerical_cols (list, optional): List of numerical column names
        """
        self.data_file = data_file
        self.transform = transform
        self.target_col = target_col

        # Load data from Excel file
        self.data = pd.read_excel(data_file)

        # Use the KG column as patient_id (it appears to contain unique identifiers)
        if 'KG' in self.data.columns:
            self.data['patient_id'] = self.data['KG']
        else:
            # Create a patient_id column if KG doesn't exist
            self.data['patient_id'] = self.data.index.astype(str)

        # Filter by patient IDs if provided
        if patient_ids is not None:
            self.data = self.data[self.data['patient_id'].isin(patient_ids)]

        # Store patient IDs
        self.patient_ids = self.data['patient_id'].values

        # Convert target column to binary (0/1)
        if target_col in self.data.columns:
            # Assuming 'NIE' means 'No' (no death) and anything else means 'Yes' (death)
            self.data[target_col] = (self.data[target_col] != 'NIE').astype(int)

        # Define categorical and numerical columns if not provided
        if categorical_cols is None:
            # Use only categorical columns that exist in the data
            self.categorical_cols = ['PLEC']  # 'PLEC' appears to be gender
        else:
            self.categorical_cols = categorical_cols

        if numerical_cols is None:
            # Use only numerical columns that exist in the data
            self.numerical_cols = ['male sex', 'WIEK']  # 'male sex' is already binary, 'WIEK' is likely age
        else:
            self.numerical_cols = numerical_cols

        # Preprocess data
        self._preprocess_data()

    def _preprocess_data(self):
        """Preprocess the tabular data."""
        # Handle missing values in numerical columns
        num_imputer = SimpleImputer(strategy='mean')
        self.data[self.numerical_cols] = num_imputer.fit_transform(self.data[self.numerical_cols])

        # Normalize numerical features
        self.scaler = StandardScaler()
        self.data[self.numerical_cols] = self.scaler.fit_transform(self.data[self.numerical_cols])

        # Handle missing values and encode categorical columns
        for col in self.categorical_cols:
            # Fill missing values with the most frequent value
            self.data[col] = self.data[col].fillna(self.data[col].mode()[0])

            # Convert to string to ensure proper encoding
            self.data[col] = self.data[col].astype(str)

        # One-hot encode categorical features
        # Use sparse_output=False for newer scikit-learn versions, fallback to sparse=False for older versions
        try:
            self.encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        except TypeError:
            self.encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')

        encoded_cats = self.encoder.fit_transform(self.data[self.categorical_cols])

        # Create feature names for encoded categorical columns
        encoded_feature_names = []
        for i, col in enumerate(self.categorical_cols):
            categories = self.encoder.categories_[i]
            for cat in categories:
                encoded_feature_names.append(f"{col}_{cat}")

        # Convert encoded categories to DataFrame
        encoded_df = pd.DataFrame(encoded_cats, columns=encoded_feature_names, index=self.data.index)

        # Combine numerical and encoded categorical features
        self.features = pd.concat([self.data[self.numerical_cols], encoded_df], axis=1)

        # Extract target variable
        self.targets = self.data[self.target_col].values

    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.data)

    def __getitem__(self, idx):
        """
        Get a sample from the dataset.

        Args:
            idx (int): Index of the sample

        Returns:
            dict: A dictionary containing:
                - 'tabular': Preprocessed tabular features tensor
                - 'target': Target variable (mortality within 30 days)
                - 'patient_id': Patient ID
        """
        patient_id = self.patient_ids[idx]
        features = self.features.iloc[idx].values
        target = self.targets[idx]

        # Check for NaN values in features
        if np.isnan(features).any():
            print(f"Warning: NaN values found in features for patient {patient_id}. Replacing with zeros.")
            features = np.nan_to_num(features, nan=0.0)

        # Check for NaN values in target
        if np.isnan(target):
            print(f"Warning: NaN value found in target for patient {patient_id}. Replacing with 0.")
            target = 0.0

        # Convert to tensors
        features_tensor = torch.tensor(features, dtype=torch.float32)
        target_tensor = torch.tensor(target, dtype=torch.float32)

        sample = {
            'tabular': features_tensor,
            'target': target_tensor,
            'patient_id': patient_id
        }

        if self.transform:
            sample = self.transform(sample)

        return sample

    def get_feature_names(self):
        """Return the names of the features after preprocessing."""
        return self.features.columns.tolist()

    def get_feature_dimensions(self):
        """Return the dimensions of the features after preprocessing."""
        return len(self.features.columns)


def create_tabular_dataloader(data_file, patient_ids=None, batch_size=32, shuffle=True, 
                             num_workers=4, target_col='mortality_30d'):
    """
    Create a DataLoader for tabular clinical data.

    Args:
        data_file (str): Path to the Excel file containing clinical data
        patient_ids (list, optional): List of patient IDs to include. If None, include all.
        batch_size (int, optional): Batch size for the DataLoader
        shuffle (bool, optional): Whether to shuffle the data
        num_workers (int, optional): Number of worker processes for data loading
        target_col (str, optional): Name of the target column (mortality within 30 days)

    Returns:
        torch.utils.data.DataLoader: DataLoader for tabular data
    """
    dataset = TabularDataset(data_file, patient_ids, target_col=target_col)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, 
                           num_workers=num_workers)
    return dataloader


class CombinedDataset(Dataset):
    """Dataset class that combines ECG and tabular data."""

    def __init__(self, ecg_dataset, tabular_dataset):
        """
        Initialize the combined dataset.

        Args:
            ecg_dataset (ECGDataset): Dataset for ECG data
            tabular_dataset (TabularDataset): Dataset for tabular data
        """
        self.ecg_dataset = ecg_dataset
        self.tabular_dataset = tabular_dataset

        # Get patient IDs from both datasets
        ecg_patient_ids = set(ecg_dataset.patient_ids)
        tabular_patient_ids = set(tabular_dataset.patient_ids)

        # Find common patient IDs
        common_patient_ids = ecg_patient_ids.intersection(tabular_patient_ids)

        # Create mapping from patient ID to indices in both datasets
        self.ecg_idx_map = {pid: i for i, pid in enumerate(ecg_dataset.patient_ids)}
        self.tabular_idx_map = {pid: i for i, pid in enumerate(tabular_dataset.patient_ids)}

        # Store common patient IDs
        self.patient_ids = list(common_patient_ids)

    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.patient_ids)

    def __getitem__(self, idx):
        """
        Get a sample from the dataset.

        Args:
            idx (int): Index of the sample

        Returns:
            dict: A dictionary containing:
                - 'ecg': Preprocessed ECG signal tensor
                - 'tabular': Preprocessed tabular features tensor
                - 'target': Target variable (mortality within 30 days)
                - 'patient_id': Patient ID
        """
        patient_id = self.patient_ids[idx]

        # Get ECG data
        ecg_idx = self.ecg_idx_map[patient_id]
        ecg_sample = self.ecg_dataset[ecg_idx]

        # Get tabular data
        tabular_idx = self.tabular_idx_map[patient_id]
        tabular_sample = self.tabular_dataset[tabular_idx]

        # Combine samples
        sample = {
            'ecg': ecg_sample['ecg'],
            'tabular': tabular_sample['tabular'],
            'target': tabular_sample['target'],
            'patient_id': patient_id
        }

        return sample


def create_combined_dataloader(ecg_data_dir, tabular_data_file, batch_size=32, shuffle=True, 
                              num_workers=4, target_col='mortality_30d', target_length=5000,
                              verbose=False, print_summary=True):
    """
    Create a DataLoader that combines ECG and tabular data.

    Args:
        ecg_data_dir (str): Directory containing ECG JSON files
        tabular_data_file (str): Path to the Excel file containing clinical data
        batch_size (int, optional): Batch size for the DataLoader
        shuffle (bool, optional): Whether to shuffle the data
        num_workers (int, optional): Number of worker processes for data loading
        target_col (str, optional): Name of the target column (mortality within 30 days)
        target_length (int, optional): Target length for ECG signals after preprocessing
        verbose (bool, optional): Whether to print detailed warnings for each file
        print_summary (bool, optional): Whether to print a summary of NaN values

    Returns:
        torch.utils.data.DataLoader: DataLoader for combined data
    """
    # Create individual datasets
    ecg_dataset = ECGDataset(ecg_data_dir, target_length=target_length, verbose=verbose)
    tabular_dataset = TabularDataset(tabular_data_file, target_col=target_col)

    # Print summary of NaN values in ECG data if requested
    if print_summary:
        ecg_dataset.print_nan_summary()

    # Create combined dataset
    combined_dataset = CombinedDataset(ecg_dataset, tabular_dataset)

    # Create DataLoader
    dataloader = DataLoader(combined_dataset, batch_size=batch_size, shuffle=shuffle, 
                           num_workers=num_workers)

    return dataloader
