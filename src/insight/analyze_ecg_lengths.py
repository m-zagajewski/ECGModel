import os
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis
from scipy.fft import fft
from scipy.signal import find_peaks



def analyze_ecg_lengths(file_path):
    """Analyze ECG signal length distribution in a directory"""
    lengths = []

    for file_name in os.listdir(file_path):
        if not file_name.endswith('.json'):
            continue

        with open(os.path.join(file_path, file_name), 'r') as f:
            try:
                data = json.load(f)
                for lead in data['leads']:
                    lengths.append(len(lead['signal']))
            except:
                continue

    print(f"Total ECGs analyzed: {len(lengths)}")
    print(f"Min length: {np.min(lengths)} samples")
    print(f"Max length: {np.max(lengths)} samples")
    print(f"Median length: {np.median(lengths)} samples")
    print(f"95th percentile: {np.percentile(lengths, 95)} samples")

    plt.figure(figsize=(10, 6))
    plt.hist(lengths, bins=50, alpha=0.7, color='steelblue')
    plt.axvline(np.median(lengths), color='red', linestyle='dashed', linewidth=2,
                label=f'Median: {np.median(lengths)}')
    plt.axvline(np.percentile(lengths, 95), color='purple', linestyle='dashed', linewidth=2,
                label=f'95th %ile: {np.percentile(lengths, 95)}')
    plt.title('ECG Signal Length Distribution')
    plt.xlabel('Number of Samples')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True)
    plt.show()

    return lengths

if __name__ == "__main__":
    # Set the path to your ECG data directory
    ecg_data_dir = "../../data/ecg"

    if not os.path.exists(ecg_data_dir):
        print(f"Directory {ecg_data_dir} does not exist.")
    else:
        lengths = analyze_ecg_lengths(ecg_data_dir)
        print(f"Lengths of ECG signals: {lengths[:10]}... (showing first 10 lengths)")