#!/usr/bin/env python
"""
Script for processing all input data and saving them as CSV files.
"""

import sys
import time
import argparse
import multiprocessing
from src.data import every_feature_to_csv, set_debug_mode, set_multiprocessing

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Processing clinical data and ECG.')
    parser.add_argument('--cores', type=int, default=None, 
                        help='Number of cores to use. Default: all available - 1')
    parser.add_argument('--debug', action='store_true', 
                        help='Debug mode with additional information.')
    parser.add_argument('--no-renew', action='store_false', dest='renew',
                        help='Do not reprocess data if files already exist.')
    
    args = parser.parse_args()
    
    # Global configuration based on arguments
    set_debug_mode(args.debug)
    
    if args.cores is None:
        n_jobs = set_multiprocessing(None)
        cpu_count = multiprocessing.cpu_count()
        print(f"Detected {cpu_count} CPU cores. Using {n_jobs} cores for processing.")
    else:
        n_jobs = set_multiprocessing(args.cores)
        print(f"Using {n_jobs} cores for processing as specified.")
    
    start_time = time.time()
    print("Starting to process all data...")
    
    try:
        result_df = every_feature_to_csv(renew_calc=args.renew, debug=args.debug, n_jobs=n_jobs)
        
        elapsed_time = time.time() - start_time
        minutes = int(elapsed_time // 60)
        seconds = int(elapsed_time % 60)
        
        print(f"Processing completed successfully in {minutes} min {seconds} s!")
        print(f"Processed {len(result_df)} records with {result_df.shape[1]} features.")
    except Exception as e:
        print(f"ERROR: Processing failed: {str(e)}", file=sys.stderr)
        sys.exit(1)
