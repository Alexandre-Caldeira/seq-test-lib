# seqtestlib/data/preprocessing.py
"""
Handles preprocessing of raw EEG data, including filtering and artifact rejection.
"""

import numpy as np
from scipy.signal import butter, filtfilt
from .. import config  # Import global settings

def preprocess_experimental_eeg(eeg_windows: np.ndarray, fs: float) -> np.ndarray:
    """
    Applies bandpass filtering, demeaning, and artifact rejection to raw EEG windows.

    This function encapsulates the preprocessing steps from the original experimental
    data processing script (10022025_genExp_v5_FIXEDvSLIDING_noimgs.ipynb).

    Args:
        eeg_windows (np.ndarray): Raw EEG data with shape (n_windows, n_samples).
        fs (float): The sampling frequency of the EEG data.

    Returns:
        np.ndarray: The preprocessed and cleaned EEG data. Returns an empty array
                    if the number of clean windows is below a threshold.
    """
    # 1. Apply Butterworth bandpass filter
    b, a = butter(
        config.FILTER_ORDER,
        [config.LOW_CUT, config.HIGH_CUT],
        btype='bandpass',
        fs=fs
    )
    filtered_windows = filtfilt(b, a, eeg_windows, axis=1)

    # 2. Demean each window
    demeaned_windows = filtered_windows - filtered_windows.mean(axis=1, keepdims=True)

    # 3. Discard the first two windows (often affected by filter artifacts)
    stable_windows = demeaned_windows[2:, :]
    if stable_windows.shape[0] == 0:
        return np.array([]) # Return empty if no windows are left

    # 4. Artifact rejection based on amplitude threshold
    max_abs_per_window = np.max(np.abs(stable_windows), axis=1)
    clean_windows = stable_windows[max_abs_per_window < config.ARTIFACT_THRESHOLD, :]

    # 5. Check if enough clean windows remain
    if clean_windows.shape[0] < 10:
        # Returning an empty array signals to the loader to skip this file
        return np.array([])

    return clean_windows