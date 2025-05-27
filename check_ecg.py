import os
import json
import numpy as np

# Set path to ECG data directory
ecg_data_dir = "data/ecg"

# Get a list of ECG JSON files
ecg_files = [f for f in os.listdir(ecg_data_dir) if f.endswith('.json')]

if not ecg_files:
    print("No ECG JSON files found in the directory.")
    exit()

# Select the first file for inspection
first_file = os.path.join(ecg_data_dir, ecg_files[0])
print(f"Inspecting file: {first_file}")

try:
    # Load the JSON file
    with open(first_file, 'r') as f:
        ecg_data = json.load(f)
    
    # Print the top-level keys in the JSON
    print("\nTop-level keys in the JSON:")
    for key in ecg_data.keys():
        print(f"  - {key}")
    
    # Check if 'signals' key exists
    if 'signals' in ecg_data:
        signals = ecg_data['signals']
        print(f"\nSignals data type: {type(signals)}")
        
        if isinstance(signals, list):
            print(f"Number of signals: {len(signals)}")
            if len(signals) > 0:
                print(f"First signal data type: {type(signals[0])}")
                print(f"First signal length: {len(signals[0]) if isinstance(signals[0], list) else 'N/A'}")
        elif isinstance(signals, dict):
            print(f"Signal keys: {signals.keys()}")
    else:
        print("\n'signals' key not found. Exploring the structure further...")
        
        # Try to find arrays or lists that might contain ECG signals
        for key, value in ecg_data.items():
            if isinstance(value, list) and len(value) > 0:
                print(f"\nKey: {key}")
                print(f"  Type: {type(value)}")
                print(f"  Length: {len(value)}")
                print(f"  First element type: {type(value[0])}")
                if isinstance(value[0], list) and len(value[0]) > 0:
                    print(f"  First element length: {len(value[0])}")
                    print(f"  First element first value: {value[0][0]}")
                elif isinstance(value[0], dict):
                    print(f"  First element keys: {value[0].keys()}")
            elif isinstance(value, dict):
                print(f"\nKey: {key}")
                print(f"  Type: {type(value)}")
                print(f"  Keys: {value.keys()}")
    
    # Print a sample of the data structure
    print("\nSample of the data structure (first 100 characters):")
    print(str(ecg_data)[:100] + "...")
    
except Exception as e:
    print(f"Error inspecting ECG file: {e}")