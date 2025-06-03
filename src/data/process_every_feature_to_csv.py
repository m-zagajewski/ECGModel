from src.data.process_excel_data import process_excel_data
from src.data.process_time_series import process_time_series_data
import pandas as pd
import os
import multiprocessing


def normalize_id(id_str):
    """
    Normalize IDs by replacing separators to enable matching between Excel and time series data.
    
    Args:
        id_str: The ID string to normalize
        
    Returns:
        str: Normalized ID with consistent separators
    """
    # Convert to string if not already
    id_str = str(id_str).strip().lower()
    
    # Replace slashes with underscores (Excel uses /, time series uses _)
    normalized = id_str.replace('/', '_')
    
    return normalized


def every_feature_to_csv(renew_calc=False, debug=False, n_jobs=None):
    """
    Process all features from the Excel file and time series data, and save them to CSV files.
    Args:
        renew_calc (bool): If True, forces recalculation of features even if CSV files already exist.
        debug (bool): If True, prints additional debug information.
        n_jobs (int, optional): Number of CPU cores to use for time series processing. 
                               If None, uses all available cores - 1.
    Returns:
        pd.DataFrame: The combined features DataFrame.
    """
    # Determine number of cores to use if not specified
    if n_jobs is None:
        n_jobs = max(1, multiprocessing.cpu_count() - 1)
    
    if debug:
        print(f"Using {n_jobs} CPU cores for parallel processing.")

    if renew_calc or not os.path.exists("../../data/processed_tabular.csv") or not os.path.exists(
            "../../data/processed_ecg_features.csv") or not os.path.exists("../../data/processed_features.csv"):
        # Process the Excel data
        df = process_excel_data(debug=debug)
        if debug:
            print(f"Processed Excel data shape: {df.shape}")
            print(f"Excel data index type: {type(df.index[0]) if len(df.index) > 0 else 'empty'}")
            print(f"First 5 Excel IDs: {df.index[:5].tolist() if len(df.index) >= 5 else df.index.tolist()}")
        
        # Save processed Excel file to separate CSV file
        df.to_csv("../../data/processed_tabular.csv")
        if debug:
            print("Excel data saved to ../../data/processed_tabular.csv")

        # Process the time series data with multiprocessing
        df_time_series = process_time_series_data(debug=debug, n_jobs=n_jobs)
        if debug:
            print(f"Processed time series data shape: {df_time_series.shape}")
            print(f"Time series data index type: {type(df_time_series.index[0]) if len(df_time_series.index) > 0 else 'empty'}")
            print(f"First 5 Time Series IDs: {df_time_series.index[:5].tolist() if len(df_time_series.index) >= 5 else df_time_series.index.tolist()}")
            
        # Save processed time series data to separate CSV file
        df_time_series.to_csv("../../data/processed_ecg_features.csv")
        if debug:
            print("ECG features saved to ../../data/processed_ecg_features.csv")

        # Create copies of dataframes with normalized indices
        df_excel_normalized = df.copy()
        df_time_series_normalized = df_time_series.copy()
        
        # Convert indices to string and normalize (Excel uses / as separator, while time_series uses _)
        df_excel_normalized.index = df_excel_normalized.index.map(normalize_id)
        df_time_series_normalized.index = df_time_series_normalized.index.map(normalize_id)
        
        if debug:
            print("\nID Normalization:")
            print("Original Excel IDs (first 3):", df.index[:3].tolist() if len(df.index) >= 3 else df.index.tolist())
            print("Normalized Excel IDs (first 3):", df_excel_normalized.index[:3].tolist() if len(df_excel_normalized.index) >= 3 else df_excel_normalized.index.tolist())
            print("Original Time Series IDs (first 3):", df_time_series.index[:3].tolist() if len(df_time_series.index) >= 3 else df_time_series.index.tolist())
            print("Normalized Time Series IDs (first 3):", df_time_series_normalized.index[:3].tolist() if len(df_time_series_normalized.index) >= 3 else df_time_series_normalized.index.tolist())

        # Find common identifiers using normalized indices
        common_ids_normalized = set(df_excel_normalized.index) & set(df_time_series_normalized.index)
        
        if debug:
            print(f"\nNumber of common IDs after normalization: {len(common_ids_normalized)}")
            print(f"Examples of common normalized IDs: {list(common_ids_normalized)[:5] if len(common_ids_normalized) >= 5 else list(common_ids_normalized)}")
        
        if len(common_ids_normalized) == 0:
            print("ERROR: No matching IDs found between Excel and ECG data after normalization!")
            print("Sample Excel IDs (normalized):", df_excel_normalized.index[:10].tolist())
            print("Sample ECG IDs (normalized):", df_time_series_normalized.index[:10].tolist())
            return pd.DataFrame()
            
        # Create mapping between normalized and original indices
        excel_id_map = {normalize_id(idx): idx for idx in df.index}
        timeseries_id_map = {normalize_id(idx): idx for idx in df_time_series.index}
        
        # Filter data to keep only common identifiers
        excel_original_ids = [excel_id_map[norm_id] for norm_id in common_ids_normalized if norm_id in excel_id_map]
        timeseries_original_ids = [timeseries_id_map[norm_id] for norm_id in common_ids_normalized if norm_id in timeseries_id_map]
        
        # Filter original dataframes based on matched identifiers
        df_filtered = df.loc[excel_original_ids]
        df_time_series_filtered = df_time_series.loc[timeseries_original_ids]
        
        if debug:
            print(f"\nFiltered Excel data shape: {df_filtered.shape}")
            print(f"Filtered time series data shape: {df_time_series_filtered.shape}")
            
        # Check if indices are unique after filtering
        if not df_filtered.index.is_unique:
            if debug:
                print("Warning: Filtered Excel data has duplicate indices. Making them unique...")
            df_filtered = df_filtered[~df_filtered.index.duplicated(keep='first')]
            
        if not df_time_series_filtered.index.is_unique:
            if debug:
                print("Warning: Filtered time series data has duplicate indices. Making them unique...")
            df_time_series_filtered = df_time_series_filtered[~df_time_series_filtered.index.duplicated(keep='first')]
        
        # Convert indices to common format for joining
        # Choose format with underscores as standard
        df_filtered.index = df_filtered.index.map(normalize_id)
        df_time_series_filtered.index = df_time_series_filtered.index.map(normalize_id)
        
        # Join data using join on normalized indices
        df_combined = df_filtered.join(df_time_series_filtered, how='inner')
        
        if debug:
            print(f"\nCombined data shape after join: {df_combined.shape}")
            print(f"Verification: all rows should have data from both sources")
            print(f"Number of output rows: {len(df_combined)}, should equal number of common IDs: {len(common_ids_normalized)}")
            
        # Save to CSV
        df_combined.to_csv("../../data/processed_features.csv")
        print(f"Processed features saved to ../../data/processed_features.csv with {len(df_combined)} records")
        
        return df_combined
    else:
        # Load from CSV if files exist
        try:
            df_combined = pd.read_csv("../../data/processed_features.csv", index_col='id')
            if debug:
                print("Loaded existing processed features from CSV.")
            return df_combined
        except Exception as e:
            print(f"Error loading processed_features.csv: {e}")
            return pd.DataFrame()


if __name__ == "__main__":
    # Get number of available cores 
    cpu_count = multiprocessing.cpu_count()
    recommended_jobs = max(1, cpu_count - 1)  # Leave one core free for system tasks
    
    print(f"Detected {cpu_count} CPU cores. Using {recommended_jobs} for processing.")
    result_df = every_feature_to_csv(renew_calc=True, debug=True, n_jobs=recommended_jobs)
    print(f"All features processed and saved to CSV files. Final shape: {result_df.shape}")
