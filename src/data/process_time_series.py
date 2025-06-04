import os
import json
import numpy as np
import pandas as pd
import neurokit2 as nk
from scipy import stats
from tqdm import tqdm
import warnings
import concurrent.futures
import multiprocessing

# Global constants for the module
ECG_LENGTH = 3382  # 95th percentile length
FS = 320  # Sampling frequency (Hz)
LEADS = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
SIGNAL_QUALITY_THRESHOLD = 0.3  # Threshold for acceptable signal quality

# Suppress specific warnings
warnings.filterwarnings("ignore", message="The behavior of obj\\[i:j\\] with a float-dtype index is deprecated")
warnings.filterwarnings("ignore", category=RuntimeWarning, message="Mean of empty slice")
warnings.filterwarnings("ignore", category=RuntimeWarning, message="invalid value encountered in")
warnings.filterwarnings("ignore", category=RuntimeWarning, message="divide by zero encountered in divide")

def cleaned_up(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean up the DataFrame by removing columns with 'error' or 'insufficient' or empty in their names.

    Args:
        df (pd.DataFrame): The DataFrame to clean up.

    Returns:
        pd.DataFrame: The cleaned DataFrame.
    """
    # Remove columns that contain 'error' or 'insufficient' in their names
    cols_to_remove = [col for col in df.columns if
                      'error' in col.lower() or 'insufficient' in col.lower() or df[col].isnull().all()]
    df_cleaned = df.drop(columns=cols_to_remove, errors='ignore')
    # Remove columns that are completely empty
    df_cleaned = df_cleaned.dropna(axis=1, how='all')
    # Remove columns that are completely constant
    df_cleaned = df_cleaned.loc[:, (df_cleaned != df_cleaned.iloc[0]).any()]
    # Remove columns that are not numeric
    numeric_cols = df_cleaned.select_dtypes(include=[np.number]).columns
    df_cleaned = df_cleaned[numeric_cols]
    #remove from index _
    df_cleaned.index = df_cleaned.index.str.replace('_', '', regex=False)
    return df_cleaned

# Helper functions for safe mathematical operations
def safe_mean(arr, default_val=np.nan):
    """Safe calculation of mean with protection against empty arrays"""
    if arr is None or len(arr) == 0:
        return default_val
    return np.nanmean(arr)

def safe_variance(arr, default_val=0):
    """Safe calculation of variance with protection against empty arrays"""
    if arr is None or len(arr) == 0 or len(arr) == 1:
        return default_val
    return np.nanvar(arr)

def safe_division(numerator, denominator, default_val=np.nan):
    """Safe division with protection against division by zero"""
    if denominator == 0 or np.isnan(denominator):
        return default_val
    return numerator / denominator

# Helper function for safe conversion to int
def safe_int_conversion(values):
    """Safe conversion to int64 with prior removal of NaN values"""
    if values is None:
        return np.array([], dtype=np.int64)

    # Convert to numpy array if not already
    if not isinstance(values, np.ndarray):
        values = np.array(values)

    # Remove NaN values
    mask = ~np.isnan(values)
    filtered = values[mask]

    # Remove negative or too large values
    mask2 = (filtered >= 0) & (filtered < 1e6)  # Reasonable limit for ECG indices
    filtered = filtered[mask2]

    # Safe conversion to int64
    try:
        return filtered.astype(np.int64)
    except (ValueError, TypeError):
        return np.array([], dtype=np.int64)

def is_valid_signal(signal_arr):
    """Enhanced signal quality check"""
    if len(signal_arr) < 100:
        return False
    if np.std(signal_arr) < 0.01:  # Flat line detection
        return False
    zero_crossings = np.sum(np.diff(np.sign(signal_arr)) != 0)
    if zero_crossings / len(signal_arr) < 0.01:  # Low variability
        return False

    # Entropy check - używamy bezpiecznego dostępu zamiast wywoływania funkcji
    try:
        # Sprawdzenie czy funkcja entropy_shannon istnieje i jest wywoływalna
        if hasattr(nk, "entropy_shannon") and callable(nk.entropy_shannon):
            entropy_result = nk.entropy_shannon(signal_arr)
            entropy_value = entropy_result[0] if isinstance(entropy_result, tuple) else entropy_result
            if entropy_value < 0.5:
                return False
    except Exception:
        # W razie problemu z entropią, pomijamy ten test
        pass

    return True

def extract_lead_features(lead_signal: np.ndarray, debug=False) -> dict:
    """Extract advanced clinical features from a single lead ECG signal"""
    features = {'signal_quality': 1.0 if is_valid_signal(lead_signal) else 0.0}

    # Skip feature extraction for poor quality signals
    if features['signal_quality'] < 0.5:
        return features

    try:
        # 1. Advanced preprocessing
        signal_clean = nk.ecg_clean(lead_signal, sampling_rate=FS, method="neurokit")
        signal_clean = nk.signal_detrend(signal_clean)

        # 2. Basic statistical features
        features.update({
            'mean': np.nanmean(signal_clean),
            'std': np.nanstd(signal_clean),
            'skewness': stats.skew(signal_clean, nan_policy='omit'),
            'kurtosis': stats.kurtosis(signal_clean, nan_policy='omit'),
            'rms': np.sqrt(np.nanmean(np.square(signal_clean))),
            'zero_crossings': len(nk.signal_zerocrossings(signal_clean))
        })

        # 3. Heart rate and rhythm analysis with robust peak detection
        try:
            _, rpeaks = nk.ecg_peaks(signal_clean, sampling_rate=FS, correct_artifacts=True)

            # Safe processing of R-peaks
            if 'ECG_R_Peaks' in rpeaks:
                # Safe conversion of R-peaks to int64 with removal of NaN values
                rpeaks_valid = safe_int_conversion(rpeaks['ECG_R_Peaks'])

                if len(rpeaks_valid) > 5:
                    # Calculate RR intervals with protection
                    rri = np.diff(rpeaks_valid) / FS * 1000  # RR intervals in ms
                    rri_sec = rri / 1000  # Convert to seconds

                    # HRV features with safe input
                    try:
                        hrv_time = nk.hrv_time(rpeaks_valid, sampling_rate=FS, show=False)
                        hrv_freq = nk.hrv_frequency(rpeaks_valid, sampling_rate=FS, show=False)
                    except Exception as e:
                        if debug:
                            print(f"HRV analysis failed: {str(e)}")
                        hrv_time = pd.DataFrame()
                        hrv_freq = pd.DataFrame()

                    # Waveform analysis with safe input
                    waves = {}
                    try:
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore", category=FutureWarning)
                            _, waves = nk.ecg_delineate(
                                signal_clean,
                                rpeaks_valid,  # Already safely converted
                                sampling_rate=FS,
                                check=False
                            )
                    except Exception as e:
                        if debug:
                            print(f"ECG delineation failed: {str(e)}")
                        waves = {}

                    # ------------------------------------------
                    # 4. Key clinical feature extraction
                    # ------------------------------------------
                    features.update({
                        'heart_rate': safe_division(60 * FS, safe_mean(rri)) if len(rri) > 0 else np.nan,
                        'hrv_rmssd': hrv_time['HRV_RMSSD'].iloc[0] if not hrv_time.empty else np.nan,
                        'hrv_sdnn': hrv_time['HRV_SDNN'].iloc[0] if not hrv_time.empty else np.nan,
                        'lf_hf_ratio': hrv_freq['HRV_LFHF'].iloc[0] if not hrv_freq.empty else np.nan
                    })

                    # P-wave analysis
                    try:
                        if 'ECG_P_Peaks' in waves and len(waves['ECG_P_Peaks']) > 2:
                            # Safe conversion of all indices
                            p_onsets = safe_int_conversion(waves['ECG_P_Onsets'])
                            p_offsets = safe_int_conversion(waves['ECG_P_Offsets'])
                            p_peaks = safe_int_conversion(waves['ECG_P_Peaks'])

                            # Further protection against incorrect indices
                            valid_indices = []
                            for i in range(len(p_peaks)):
                                if (i < len(p_onsets) and i < len(p_offsets) and
                                        0 <= p_onsets[i] < len(signal_clean) and
                                        0 <= p_offsets[i] < len(signal_clean) and
                                        0 <= p_peaks[i] < len(signal_clean)):
                                    valid_indices.append(i)

                            if valid_indices:
                                # Select only valid indices
                                p_durations = [p_offsets[i] - p_onsets[i] for i in valid_indices]
                                p_amplitudes = [signal_clean[p_peaks[i]] for i in valid_indices]

                                if p_durations and p_amplitudes:
                                    features.update({
                                        'p_duration': safe_mean(p_durations),
                                        'p_amplitude': safe_mean(p_amplitudes),
                                        'p_amplitude_var': safe_variance(p_amplitudes)
                                    })
                    except Exception as e:
                        if debug:
                            print(f"P-wave analysis failed: {str(e)}")

                    # QRS complex analysis with safe processing
                    try:
                        qrs_durations = []
                        r_amplitudes = []

                        # Safely convert Q and S peaks
                        q_peaks = safe_int_conversion(waves.get('ECG_Q_Peaks', []))
                        s_peaks = safe_int_conversion(waves.get('ECG_S_Peaks', []))

                        # Analysis only when we have both Q and S peaks
                        if len(q_peaks) > 0 and len(s_peaks) > 0:
                            # Find safe length for iteration
                            min_len = min(len(q_peaks), len(s_peaks), len(rpeaks_valid))

                            # Safely calculate QRS duration and R amplitude
                            for i in range(min_len):
                                if (i < len(q_peaks) and i < len(s_peaks) and i < len(rpeaks_valid) and
                                        0 <= q_peaks[i] < len(signal_clean) and
                                        0 <= s_peaks[i] < len(signal_clean) and
                                        0 <= rpeaks_valid[i] < len(signal_clean) and
                                        q_peaks[i] < s_peaks[i]):
                                    qrs_durations.append(s_peaks[i] - q_peaks[i])
                                    r_amplitudes.append(signal_clean[rpeaks_valid[i]])

                        # Add R-amplitudes even if there are no Q/S peaks
                        if not r_amplitudes:
                            for r_peak in rpeaks_valid:
                                if 0 <= r_peak < len(signal_clean):
                                    r_amplitudes.append(signal_clean[r_peak])

                        if qrs_durations:
                            features.update({
                                'qrs_duration': safe_mean(qrs_durations),
                                'qrs_duration_var': safe_variance(qrs_durations)
                            })

                        if r_amplitudes:
                            features['r_amplitude'] = safe_mean(r_amplitudes)
                    except Exception as e:
                        if debug:
                            print(f"QRS analysis error: {str(e)}")

                    # ST-T segment analysis
                    try:
                        st_levels = []
                        t_amplitudes = []

                        # Safely convert T peaks
                        t_peaks = safe_int_conversion(waves.get('ECG_T_Peaks', []))

                        for r_peak in rpeaks_valid:
                            if 0 <= r_peak < len(signal_clean):
                                st_start = r_peak + int(0.08 * FS)
                                st_end = st_start + int(0.04 * FS)

                                # Safe calculation of ST level
                                if 0 <= st_start < st_end < len(signal_clean):
                                    st_level = np.mean(signal_clean[st_start:st_end])
                                    st_levels.append(st_level)

                        # Safe calculation of T amplitude
                        for t_peak in t_peaks:
                            if 0 <= t_peak < len(signal_clean):
                                t_amplitudes.append(signal_clean[t_peak])

                        if st_levels:
                            features.update({
                                'st_level': safe_mean(st_levels),
                                'st_slope': stats.linregress(np.arange(len(st_levels)), st_levels).slope if len(
                                    st_levels) > 2 else 0,
                                't_amplitude': safe_mean(t_amplitudes) if t_amplitudes else np.nan
                            })
                    except Exception as e:
                        if debug:
                            print(f"ST-T analysis error: {str(e)}")

                    # QT interval analysis
                    try:
                        qt_intervals = []
                        qtc_values = []

                        # Safe conversion of Q peaks and T offsets
                        q_peaks = safe_int_conversion(waves.get('ECG_Q_Peaks', []))
                        t_offsets = safe_int_conversion(waves.get('ECG_T_Offsets', []))

                        # QT analysis only when we have appropriate data
                        min_len = min(len(q_peaks), len(t_offsets), len(rpeaks_valid) - 1) if len(
                            rpeaks_valid) > 0 else 0
                        for i in range(min_len):
                            if (0 <= q_peaks[i] < len(signal_clean) and
                                    0 <= t_offsets[i] < len(signal_clean) and
                                    q_peaks[i] < t_offsets[i] and i < len(rri_sec)):

                                qt_intervals.append(t_offsets[i] - q_peaks[i])
                                # Convert to seconds for QTc calculation
                                qt_sec = (t_offsets[i] - q_peaks[i]) / FS
                                rr_sec = rri_sec[i]
                                if rr_sec > 0:  # Protection against division by zero
                                    qtc_values.append(qt_sec / np.sqrt(rr_sec))

                        if qtc_values:
                            features.update({
                                'qtc_mean': safe_mean(qtc_values),
                                'qtc_sd': np.nanstd(qtc_values) if len(qtc_values) > 1 else 0
                            })
                    except Exception as e:
                        if debug:
                            print(f"QT analysis error: {str(e)}")

                    # 5. Advanced features
                    # T-wave alternans
                    try:
                        if len(t_amplitudes) > 30:
                            even = t_amplitudes[::2]
                            odd = t_amplitudes[1::2]
                            if len(even) > 15 and len(odd) > 15:
                                features['twa'] = np.abs(np.mean(even) - np.mean(odd))
                    except Exception as e:
                        if debug:
                            print(f"T-wave alternans analysis failed: {str(e)}")

                    # Entropy measures - we use Shannon entropy from scipy instead of problematic functions
                    try:
                        from scipy.stats import entropy

                        # Prepare data for entropy calculation
                        hist, _ = np.histogram(signal_clean, bins=50, density=True)
                        if len(hist) > 0 and np.sum(hist) > 0:
                            features['shannon_entropy'] = entropy(hist)

                        # Calculate our own version of signal complexity
                        diffs = np.diff(signal_clean)
                        if len(diffs) > 10:
                            mean_abs_diff = np.mean(np.abs(diffs))
                            if mean_abs_diff > 0:  # Protection against division by zero
                                features['signal_complexity'] = np.std(diffs) / mean_abs_diff
                            else:
                                features['signal_complexity'] = 0

                    except Exception as e:
                        if debug:
                            print(f"Entropy analysis alternative failed: {str(e)}")

                    # Signal quality metrics
                    try:
                        noise_level = np.std(signal_clean - nk.signal_smooth(signal_clean, size=int(FS / 10)))
                        signal_var = np.var(signal_clean)
                        if signal_var > 0 and noise_level > 0:
                            features['snr_db'] = 10 * np.log10(signal_var / (noise_level ** 2 + 1e-10))
                        else:
                            features['snr_db'] = 0
                    except Exception as e:
                        if debug:
                            print(f"SNR calculation failed: {str(e)}")
        except Exception as e:
            if debug:
                print(f"Heart rhythm analysis failed: {str(e)}")
    except Exception as e:
        if debug:
            print(f"General feature extraction failed: {str(e)}")
        features['extraction_error'] = 1.0

    return features

def process_single_file(file_path, file_name, debug=False):
    """Process a single ECG file and return extracted data and case_id."""
    file_full_path = os.path.join(file_path, file_name)

    try:
        with open(file_full_path, 'r') as f:
            data = json.load(f)

        # Extract case_id from filename (without extension)
        case_id = os.path.splitext(file_name)[0]

        # Process leads in standard order
        lead_signals = []
        for lead in LEADS:
            signal_data = next((obj for obj in data['leads'] if obj['lead_name'] == lead), None)
            waveform = np.zeros(ECG_LENGTH, dtype=np.float32)

            if signal_data:
                # Clean and pad signal
                raw_signal = np.array(
                    [x if x is not None and not np.isnan(x) else 0.0
                     for x in signal_data['signal']],
                    dtype=np.float32
                )
                valid_length = min(len(raw_signal), ECG_LENGTH)
                waveform[:valid_length] = raw_signal[:valid_length]

            lead_signals.append(waveform)

        return case_id, lead_signals

    except (KeyError, json.JSONDecodeError) as e:
        if debug:
            print(f"Skipping {file_name}: {str(e)}")
        return None

def extract_case_features(case_data, debug=False):
    """Extract features for a single case with all its leads."""
    if case_data is None:
        return None

    case_id, lead_signals = case_data
    features = {"case_id": case_id}

    for lead_idx, lead_signal in enumerate(lead_signals):
        lead_name = LEADS[lead_idx]
        lead_features = extract_lead_features(lead_signal, debug=debug)

        for k, v in lead_features.items():
            features[f"{lead_name}_{k}"] = v

    return features

def process_time_series_data(file_path="../../data/ecg", debug=False, show_progress=True, n_jobs=None) -> pd.DataFrame:
    """
    Process time series data from ECG files.

    Args:
        file_path (str): Path to the directory with ECG files.
        debug (bool): If True, prints debug information and processes only 10 files.
        show_progress (bool): If True, shows progress bars.
        n_jobs (int, optional): Number of CPU cores to use. If None, uses all available cores - 1.

    Returns:
        pd.DataFrame: DataFrame with extracted features.
    """
    # Determine number of processes to use
    if n_jobs is None:
        n_jobs = max(1, multiprocessing.cpu_count() - 1)  # Use all cores - 1 by default
    else:
        n_jobs = min(max(1, n_jobs), multiprocessing.cpu_count())  # Limit to available cores

    if debug:
        print(f"Using {n_jobs} CPU cores for ECG processing.")

    # Main processing pipeline with parallel execution
    json_files = [f for f in os.listdir(file_path) if f.endswith('.json')]

    if debug:
        print(f"Found {len(json_files)} JSON files in {file_path}")
        json_files = json_files[:10]  # Process only 10 files in debug mode
    elif show_progress:
        print(f"Found {len(json_files)} JSON files in {file_path}")

    processed_cases = []

    # First phase: Process ECG files in parallel
    with tqdm(total=len(json_files), desc="Processing ECGs", disable=not show_progress) as pbar:
        # Check if multiprocessing should be used
        if n_jobs > 1:
            with concurrent.futures.ProcessPoolExecutor(max_workers=n_jobs) as executor:
                # Add file_path parameter to each function
                futures = [
                    executor.submit(process_single_file, file_path, file_name, debug) 
                    for file_name in json_files
                ]
                for future in concurrent.futures.as_completed(futures):
                    result = future.result()
                    if result is not None:
                        processed_cases.append(result)
                    pbar.update(1)
        else:
            # Sequential processing if n_jobs=1
            for file_name in json_files:
                result = process_single_file(file_path, file_name, debug)
                if result is not None:
                    processed_cases.append(result)
                pbar.update(1)

    # Second phase: Extract features from processed cases in parallel
    feature_rows = []

    with tqdm(total=len(processed_cases), desc="Extracting features", disable=not show_progress) as pbar:
        # Check if multiprocessing should be used
        if n_jobs > 1:
            with concurrent.futures.ProcessPoolExecutor(max_workers=n_jobs) as executor:
                futures = [
                    executor.submit(extract_case_features, case_data, debug)
                    for case_data in processed_cases
                ]
                for future in concurrent.futures.as_completed(futures):
                    result = future.result()
                    if result is not None:
                        feature_rows.append(result)
                    pbar.update(1)
        else:
            # Sequential processing if n_jobs=1
            for case_data in processed_cases:
                result = extract_case_features(case_data, debug)
                if result is not None:
                    feature_rows.append(result)
                pbar.update(1)

    # Create DataFrame with case_id as index
    df = pd.DataFrame(feature_rows)
    if 'case_id' in df.columns:
        df.set_index('case_id', inplace=True)

    # Impute missing values
    non_error_cols = [c for c in df.columns if 'error' not in c and 'insufficient' not in c]
    for col in non_error_cols:
        if df[col].dtype.kind in 'f':  # Only float columns
            df[col] = df[col].fillna(df[col].median())

    df = cleaned_up(df)

    if debug:
        print(f"Extracted {len(df)} cases with {len(df.columns)} features")
    elif show_progress:
        print(f"Extracted {len(df)} cases with {len(df.columns)} features")

    return df


if __name__ == "__main__":
    # Auto-detect number of cores
    cpu_count = multiprocessing.cpu_count()
    n_jobs = max(1, cpu_count - 1)  # Leave one core free
    print(f"Detected {cpu_count} CPU cores. Using {n_jobs} for processing.")

    df_ecg = process_time_series_data(file_path="../../data/ecg", debug=False, show_progress=True, n_jobs=n_jobs)
    df_ecg.to_csv("../../data/processed_ecg_features.csv")
    print(f"ECG features saved to processed_ecg_features.csv. Shape: {df_ecg.shape}")
