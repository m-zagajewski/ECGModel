"""
Data processing module for ECGModel project.

Contains functions for processing clinical data and ECG signals.
"""

import os
from src.data.process_excel_data import process_excel_data
from src.data.process_time_series import process_time_series_data
from src.data.process_every_feature_to_csv import every_feature_to_csv

# Exported functions
__all__ = [
    'process_excel_data',
    'process_time_series_data',
    'every_feature_to_csv',
    'set_debug_mode',
    'is_debug_mode',
    'set_multiprocessing',
    'get_multiprocessing_jobs'
]

# Global configuration
_debug_mode = False
_n_jobs = None  # None means auto-detection (CPU_COUNT - 1)

def set_debug_mode(enable=True):
    """
    Enable or disable debug mode for the entire module.
    
    Args:
        enable (bool, optional): Whether to enable debug mode. Default: True.
    """
    global _debug_mode
    _debug_mode = enable
    return _debug_mode

def is_debug_mode():
    """
    Check if debug mode is enabled.
    
    Returns:
        bool: True if debug mode is enabled, False otherwise.
    """
    return _debug_mode

def set_multiprocessing(n_jobs=None):
    """
    Set the number of processes to use for multiprocessing.
    
    Args:
        n_jobs (int, optional): Number of processes. None means auto-detection (CPU_COUNT - 1).
            0 or 1 means no multiprocessing. Default: None.
    
    Returns:
        int: The set number of processes.
    """
    import multiprocessing
    global _n_jobs
    
    if n_jobs is None:
        _n_jobs = max(1, multiprocessing.cpu_count() - 1)
    else:
        _n_jobs = max(1, min(n_jobs, multiprocessing.cpu_count()))
    
    return _n_jobs

def get_multiprocessing_jobs():
    """
    Return the number of processes set for multiprocessing.
    
    Returns:
        int: The number of processes.
    """
    if _n_jobs is None:
        return set_multiprocessing(None)
    return _n_jobs

# Initialize default multiprocessing settings
set_multiprocessing()

# Detect debug mode based on environment variable
if os.environ.get('ECGMODEL_DEBUG', '').lower() in ('1', 'true', 'yes'):
    set_debug_mode(True)
