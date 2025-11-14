# seqtestlib/evaluation/metrics.py
"""
Functions for calculating final performance metrics like TPR, FPR, and TTFD
based on the output of sequential testers.
"""

import numpy as np

def calculate_trial_performance(
    detections: np.ndarray,
    signal_bins: list,
    noise_bins: list,
    m_value: int,
    detection_method: str,
    num_windows: int
) -> tuple:
    """
    Calculates key performance indicators for a batch of trials.

    This function computes:
    - True Positive Rate (TPR): Whether a signal was correctly detected in a trial.
    - False Positive Rate (FPR): Whether a false alarm occurred in a trial.
    - Time to First Detection (TTFD): The window index of the first correct detection.
      Trials with no correct detection are assigned the maximum possible duration as a penalty.

    Args:
        detections (np.ndarray): Boolean matrix of detections from a tester.
                                 Shape: (num_trials, time_axis_len, num_freqs).
        signal_bins (list): Indices of signal frequency bins.
        noise_bins (list): Indices of noise frequency bins.
        m_value (int): The M-value (window or block size) used.
        detection_method (str): 'Per-Window' or 'Per-M-Block'.
        num_windows (int): The total number of windows in the original trial data.

    Returns:
        tuple: A tuple containing (tpr_flags, fpr_flags, ttfd_values), where each
               is a 1D numpy array with results for each trial.
    """
    num_trials, time_axis_len, _ = detections.shape

    # --- 1. Calculate Trial-level TPR and FPR flags ---
    # A true positive occurs if *any* detection happens in *any* signal bin at *any* time.
    tpr_flags = np.any(detections[:, :, signal_bins], axis=(1, 2))

    # A false positive occurs if *any* detection happens in *any* noise bin at *any* time.
    fpr_flags = np.any(detections[:, :, noise_bins], axis=(1, 2))

    # --- 2. Calculate TTFD with Maximum Duration Penalty ---
    # Determine the maximum duration based on the detection method.
    if detection_method == 'Per-M-Block':
        # Max duration is the end of the last full block.
        max_duration = (num_windows // m_value) * m_value
    else:  # Per-Window
        max_duration = num_windows

    # Isolate detection events that occurred in signal bins.
    true_positive_events = np.any(detections[:, :, signal_bins], axis=2)  # Shape: (num_trials, time_axis_len)

    # Initialize TTFD for all trials with the max duration penalty.
    ttfd_values = np.full(num_trials, float(max_duration))

    # Find which trials had at least one true positive detection.
    detected_trials_mask = np.any(true_positive_events, axis=1)

    if np.any(detected_trials_mask):
        # For detected trials, get the index of the *first* detection event.
        first_detection_indices = np.argmax(true_positive_events[detected_trials_mask], axis=1)

        # Convert the detection index to the corresponding window number.
        if detection_method == 'Per-M-Block':
            # A detection at block index `k` occurs at the end of window `(k+1) * m_value`.
            detection_times = (first_detection_indices + 1) * m_value
        else: # Per-Window
            # A detection at index `k` is simply window `k`.
            detection_times = first_detection_indices

        # Update the TTFD for only those trials where a detection occurred.
        ttfd_values[detected_trials_mask] = detection_times

    return tpr_flags, fpr_flags, ttfd_values