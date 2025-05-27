"""
Module for loading and preprocessing ECG data from JSON files.

This module provides functionality to:
1. Load ECG data from JSON files
2. Preprocess ECG signals (normalization, padding/cropping, etc.)
3. Create PyTorch datasets for ECG data
"""

import json
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class ECGDataset(Dataset):
    """Dataset class for ECG data."""

    def __init__(self, data_dir, patient_ids=None, transform=None, target_length=5000, verbose=False):
        """
        Initialize the ECG dataset.

        Args:
            data_dir (str): Directory containing ECG JSON files
            patient_ids (list, optional): List of patient IDs to include. If None, include all.
            transform (callable, optional): Optional transform to be applied on a sample
            target_length (int, optional): Target length for ECG signals after preprocessing
            verbose (bool, optional): Whether to print detailed warnings for each file
        """
        self.data_dir = data_dir
        self.transform = transform
        self.target_length = target_length
        self.verbose = verbose

        # Statistics for NaN values
        self.nan_stats = {
            'files_with_nans': 0,
            'total_nans': 0,
            'leads_with_nans': [0] * 12,
            'nans_after_normalization': 0
        }

        # Get all JSON files in the directory if patient_ids is None
        if patient_ids is None:
            self.file_paths = [os.path.join(data_dir, f) for f in os.listdir(data_dir) 
                              if f.endswith('.json')]
        else:
            # Convert patient_ids from "number/year" format to "number_year" format for file matching
            formatted_patient_ids = [patient_id.replace('/', '_') for patient_id in patient_ids]
            self.file_paths = [os.path.join(data_dir, f"{formatted_id}.json") for formatted_id in formatted_patient_ids
                              if os.path.exists(os.path.join(data_dir, f"{formatted_id}.json"))]

        # Extract patient IDs from file paths and convert from "number_year" format to "number/year" format
        self.patient_ids = [os.path.splitext(os.path.basename(fp))[0].replace('_', '/') for fp in self.file_paths]

    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.file_paths)

    def __getitem__(self, idx):
        """
        Get a sample from the dataset.

        Args:
            idx (int): Index of the sample

        Returns:
            dict: A dictionary containing:
                - 'ecg': Preprocessed ECG signal tensor of shape [12, target_length]
                - 'patient_id': Patient ID
        """
        file_path = self.file_paths[idx]
        patient_id = self.patient_ids[idx]

        # Load ECG data from JSON file
        with open(file_path, 'r') as f:
            ecg_data = json.load(f)

        # Extract ECG signals (assuming a specific structure in the JSON)
        # This will need to be adapted based on the actual JSON structure
        ecg_signals = self._extract_signals_from_json(ecg_data)

        # Preprocess ECG signals
        ecg_signals = self._preprocess_signals(ecg_signals)

        # Convert to tensor
        ecg_tensor = torch.tensor(ecg_signals, dtype=torch.float32)

        sample = {
            'ecg': ecg_tensor,
            'patient_id': patient_id
        }

        if self.transform:
            sample = self.transform(sample)

        return sample

    def _extract_signals_from_json(self, ecg_data):
        """
        Extract ECG signals from JSON data.

        Args:
            ecg_data (dict): JSON data containing ECG signals

        Returns:
            numpy.ndarray: ECG signals of shape [12, time_steps]
        """
        # Extract signals from the 'leads' list
        if 'leads' in ecg_data and isinstance(ecg_data['leads'], list):
            leads = ecg_data['leads']

            # Check if we have 12 leads
            if len(leads) != 12:
                print(f"Warning: Expected 12 leads, but found {len(leads)}. Using available leads.")

            # Extract signals from each lead
            raw_signals = []
            for lead in leads:
                if 'signal' in lead and isinstance(lead['signal'], list):
                    raw_signals.append(lead['signal'])
                else:
                    # If signal is missing or not a list, use zeros
                    print(f"Warning: Signal missing or invalid for lead {lead.get('lead_name', 'unknown')}. Using zeros.")
                    # Use a default length of 5000 if we don't have any signals yet
                    length = len(raw_signals[0]) if raw_signals else 5000
                    raw_signals.append([0.0] * length)

            # Ensure all signals have the same length
            if raw_signals:
                # Find the most common length
                lengths = [len(signal) for signal in raw_signals]
                common_length = max(set(lengths), key=lengths.count)

                # Adjust all signals to the common length
                signals = []
                for signal in raw_signals:
                    if len(signal) != common_length:
                        # Pad or truncate the signal to match the common length
                        if len(signal) > common_length:
                            # Truncate
                            signals.append(signal[:common_length])
                        else:
                            # Pad with zeros
                            padded_signal = signal + [0.0] * (common_length - len(signal))
                            signals.append(padded_signal)
                    else:
                        signals.append(signal)

                # Convert to numpy array
                signals = np.array(signals, dtype=np.float32)

                # Ensure we have exactly 12 leads
                if len(signals) < 12:
                    # Pad with zero signals if we have fewer than 12 leads
                    zero_signals = np.zeros((12 - len(signals), common_length), dtype=np.float32)
                    signals = np.vstack((signals, zero_signals))
                elif len(signals) > 12:
                    # Truncate if we have more than 12 leads
                    signals = signals[:12]

                return signals
            else:
                # If no valid signals were found, return zeros
                return np.zeros((12, 5000), dtype=np.float32)
        else:
            # If 'leads' is missing or not a list, return an empty array
            print("Warning: 'leads' key missing or not a list in ECG data. Using zeros.")
            return np.zeros((12, 5000), dtype=np.float32)

    def _preprocess_signals(self, signals):
        """
        Preprocess ECG signals.

        Args:
            signals (numpy.ndarray): Raw ECG signals of shape [12, time_steps]

        Returns:
            numpy.ndarray: Preprocessed ECG signals of shape [12, target_length]
        """
        # Normalize each lead
        signals = self._normalize_signals(signals)

        # Adjust length to target_length
        signals = self._adjust_length(signals, self.target_length)

        return signals

    def _normalize_signals(self, signals):
        """
        Normalize ECG signals.

        Args:
            signals (numpy.ndarray): ECG signals of shape [12, time_steps]

        Returns:
            numpy.ndarray: Normalized ECG signals
        """
        # Check for NaN values
        if np.isnan(signals).any():
            self.nan_stats['files_with_nans'] += 1
            nan_count = np.isnan(signals).sum()
            self.nan_stats['total_nans'] += nan_count

            # Count NaNs per lead
            for i in range(signals.shape[0]):
                lead_nan_count = np.isnan(signals[i]).sum()
                if lead_nan_count > 0:
                    self.nan_stats['leads_with_nans'][i] += 1

            if self.verbose:
                print(f"Warning: {nan_count} NaN values found in ECG signals. Replacing with zeros.")

            signals = np.nan_to_num(signals, nan=0.0)

        # Normalize each lead independently
        normalized_signals = np.zeros_like(signals, dtype=np.float32)

        for i in range(signals.shape[0]):
            # Z-score normalization
            mean = np.mean(signals[i])
            std = np.std(signals[i])
            if std > 0:
                normalized_signals[i] = (signals[i] - mean) / std
            else:
                normalized_signals[i] = signals[i] - mean

            # Check for NaN values after normalization
            if np.isnan(normalized_signals[i]).any():
                nan_count = np.isnan(normalized_signals[i]).sum()
                self.nan_stats['nans_after_normalization'] += nan_count

                if self.verbose:
                    print(f"Warning: {nan_count} NaN values found after normalization in lead {i}. Replacing with zeros.")

                normalized_signals[i] = np.nan_to_num(normalized_signals[i], nan=0.0)

        return normalized_signals

    def _adjust_length(self, signals, target_length):
        """
        Adjust the length of ECG signals to the target length.

        Args:
            signals (numpy.ndarray): ECG signals of shape [12, time_steps]
            target_length (int): Target length for the signals

        Returns:
            numpy.ndarray: ECG signals of shape [12, target_length]
        """
        current_length = signals.shape[1]

        if current_length == target_length:
            return signals

        adjusted_signals = np.zeros((signals.shape[0], target_length), dtype=np.float32)

        if current_length > target_length:
            # Crop signals
            start = (current_length - target_length) // 2
            adjusted_signals = signals[:, start:start+target_length]
        else:
            # Pad signals
            start = (target_length - current_length) // 2
            adjusted_signals[:, start:start+current_length] = signals

        return adjusted_signals


    def print_nan_summary(self):
        """
        Print a summary of NaN values found during processing.
        """
        if self.nan_stats['files_with_nans'] > 0:
            print("\nECG Data NaN Summary:")
            print(f"- Files with NaN values: {self.nan_stats['files_with_nans']} out of {len(self.file_paths)} ({self.nan_stats['files_with_nans']/len(self.file_paths)*100:.1f}%)")
            print(f"- Total NaN values found: {self.nan_stats['total_nans']}")
            print(f"- NaN values found after normalization: {self.nan_stats['nans_after_normalization']}")

            # Print leads with NaNs
            leads_with_nans = [(i, count) for i, count in enumerate(self.nan_stats['leads_with_nans']) if count > 0]
            if leads_with_nans:
                print("- Leads with NaN values:")
                for lead_idx, count in leads_with_nans:
                    print(f"  - Lead {lead_idx}: {count} files ({count/len(self.file_paths)*100:.1f}%)")

            print("All NaN values have been replaced with zeros.")
        else:
            print("\nNo NaN values found in ECG data.")


def create_ecg_dataloader(data_dir, patient_ids=None, batch_size=32, shuffle=True, 
                         num_workers=4, target_length=5000, verbose=False, print_summary=True):
    """
    Create a DataLoader for ECG data.

    Args:
        data_dir (str): Directory containing ECG JSON files
        patient_ids (list, optional): List of patient IDs to include. If None, include all.
        batch_size (int, optional): Batch size for the DataLoader
        shuffle (bool, optional): Whether to shuffle the data
        num_workers (int, optional): Number of worker processes for data loading
        target_length (int, optional): Target length for ECG signals after preprocessing
        verbose (bool, optional): Whether to print detailed warnings for each file
        print_summary (bool, optional): Whether to print a summary of NaN values

    Returns:
        torch.utils.data.DataLoader: DataLoader for ECG data
    """
    dataset = ECGDataset(data_dir, patient_ids, target_length=target_length, verbose=verbose)

    # Create DataLoader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, 
                           num_workers=num_workers)

    # Print summary of NaN values if requested
    if print_summary:
        dataset.print_nan_summary()

    return dataloader
