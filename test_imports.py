import os
import torch
from src.data.ecg_processor import ECGDataset
from src.data.tabular_processor import TabularDataset, CombinedDataset

# Check if the data directories exist
ecg_data_dir = "data/ecg"
tabular_data_file = "data/clinical_data.xlsx"

print(f"ECG data directory exists: {os.path.exists(ecg_data_dir)}")
print(f"Tabular data file exists: {os.path.exists(tabular_data_file)}")

# Try to create dataset instances
try:
    # This might fail if there are no ECG files in the directory, but it should import correctly
    print("Creating ECGDataset instance...")
    ecg_dataset = ECGDataset(ecg_data_dir, verbose=False)
    print("ECGDataset created successfully!")

    # Print summary of NaN values in ECG data
    print("\nChecking for NaN values in ECG data:")
    ecg_dataset.print_nan_summary()
except Exception as e:
    print(f"Error creating ECGDataset: {e}")

try:
    # This should work if the tabular data file exists
    print("Creating TabularDataset instance...")
    tabular_dataset = TabularDataset(tabular_data_file)
    print("TabularDataset created successfully!")
except Exception as e:
    print(f"Error creating TabularDataset: {e}")

try:
    # This might fail if either of the above fails, but the import should be correct
    print("Creating CombinedDataset instance...")
    combined_dataset = CombinedDataset(ecg_dataset, tabular_dataset)
    print("CombinedDataset created successfully!")
except Exception as e:
    print(f"Error creating CombinedDataset: {e}")

print("Import test completed!")
