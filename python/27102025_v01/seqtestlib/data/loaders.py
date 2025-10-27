# seqtestlib/data/loaders.py
"""
Data loaders for simulated HDF5 files and experimental .mat files.
Provides a consistent interface for accessing different data sources.
"""

import os
import re
import ast
import h5py
import numpy as np
from collections import defaultdict

from .. import config
from .preprocessing import preprocess_experimental_eeg

# --- Helper function from genExp_v5 notebook ---
def parse_filename(filepath: str) -> tuple:
    """Extracts the volunteer ID and intensity from a .mat filename."""
    filename = os.path.basename(filepath)
    match = re.match(r"([A-Za-z]+)(\d+dB|ESP)\.mat", filename)
    if match:
        return match.group(1), match.group(2)
    return None, None

# --- Main Loader Classes ---

class ExperimentalLoader:
    """
    Loads and preprocesses data from experimental .mat files.
    """
    def __init__(self, data_dir: str = config.EXPERIMENTAL_RAW_DATA_DIR):
        """
        Initializes the loader by discovering all .mat files in the directory.

        Args:
            data_dir (str): Path to the directory containing experimental .mat files.
        """
        self.data_dir = data_dir
        self.file_paths = {os.path.basename(p): p for p in os.listdir(data_dir) if p.endswith('.mat')}
        print(f"ExperimentalLoader found {len(self.file_paths)} .mat files in '{data_dir}'.")

    def load_exam(self, volunteer_id: str, intensity: str) -> dict:
        """
        Loads, preprocesses, and returns data for a single experimental exam.

        Args:
            volunteer_id (str): The identifier for the volunteer (e.g., 'Wr').
            intensity (str): The stimulus intensity level (e.g., '30dB').

        Returns:
            dict: A dictionary containing 'clean_windows', 'fs', 'nfft', and 'filepath',
                  or an empty dict if the file is not found or has insufficient clean data.
        """
        filename = f"{volunteer_id}{intensity}.mat"
        filepath = os.path.join(self.data_dir, filename)

        if not os.path.exists(filepath):
            print(f"Warning: File not found at '{filepath}'.")
            return {}

        try:
            with h5py.File(filepath, 'r') as f:
                raw_data = f['x'][:]
                fs = f['Fs'][0, 0]
                nfft = raw_data.shape[2]

            # Select electrode and preprocess
            eeg_windows = raw_data[config.ELECTRODE_TO_USE, :, :]
            clean_windows = preprocess_experimental_eeg(eeg_windows, fs)

            if clean_windows.shape[0] == 0:
                print(f"Warning: Insufficient clean windows for '{filename}'. Skipping.")
                return {}

            return {
                'clean_windows': clean_windows,
                'fs': fs,
                'nfft': nfft,
                'filepath': filepath
            }
        except Exception as e:
            print(f"Error processing file '{filename}': {e}")
            return {}


class SimulatedLoader:
    """
    Loads and prepares data from pre-generated simulated feature HDF5 files.
    """
    def __init__(self, filepath: str = config.SIMULATED_FEATURES_FILE):
        """
        Initializes the loader with the path to the simulated features HDF5 file.

        Args:
            filepath (str): Path to the HDF5 file.
        """
        self.filepath = filepath
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Simulated data file not found: {filepath}")
        self.num_trials, self.signal_bins, self.noise_bins, self.dsets = self._get_file_params()
        self.snr_keys = self._get_snr_keys()

    def _get_file_params(self) -> tuple:
        """Extracts metadata from the HDF5 file."""
        with h5py.File(self.filepath, 'r') as f:
            attrs = f.attrs
            sig_freqs = ast.literal_eval(attrs['parameters_signal_freqs'])
            noise_freqs = ast.literal_eval(attrs['parameters_noise_freqs'])
            all_freqs = sig_freqs + noise_freqs
            signal_bins = [all_freqs.index(freq) for freq in sig_freqs]
            noise_bins = [all_freqs.index(freq) for freq in noise_freqs]
            snr_keys = [k for k in f.keys() if k.startswith('snr_')]
            dsets = list(f[snr_keys[0]].keys()) if snr_keys else []
            num_trials = f[snr_keys[0]][dsets[0]].shape[0] if (snr_keys and dsets) else 0
        return num_trials, signal_bins, noise_bins, dsets

    def _get_snr_keys(self) -> list:
        """Returns a sorted list of SNR keys from the file."""
        with h5py.File(self.filepath, 'r') as f:
            return sorted([k for k in f.keys() if k.startswith('snr_')], key=lambda x: float(x.split('_')[1]))

    def get_data_for_ml(self, m_val: int, feature_idx: int) -> tuple:
        """
        Loads and prepares features and labels for ML model training, applying sampling rules.

        Args:
            m_val (int): The window size (M) to create features from.
            feature_idx (int): The index of the base ORD feature to use (0=MSC, 1=CSM).

        Returns:
            tuple: A tuple of (X, y_class, y_domain) for training.
        """
        hdf5_dset_name = self._get_hdf5_dset_name(m_val, "Sliding")
        all_features, all_class_labels, all_domain_labels = [], [], []

        with h5py.File(self.filepath, 'r') as f:
            snr_map = {float(k.replace('snr_', '')): i for i, k in enumerate(self.snr_keys)}
            domain_data = defaultdict(lambda: {'features': [], 'labels': []})

            for snr_key in self.snr_keys:
                ord_values = f[snr_key][hdf5_dset_name][:config.MAX_SIM_TRIALS_FOR_ML_TRAINING, :, :, feature_idx]

                sig_ts = ord_values[:, :, self.signal_bins].transpose(0, 2, 1).reshape(-1, ord_values.shape[1])
                noise_ts = ord_values[:, :, self.noise_bins].transpose(0, 2, 1).reshape(-1, ord_values.shape[1])

                sig_feats = self._create_sliding_windows(sig_ts, m_val)
                noise_feats = self._create_sliding_windows(noise_ts, m_val)

                if sig_feats.size > 0 or noise_feats.size > 0:
                    domain_idx = snr_map[float(snr_key.replace('snr_', ''))]
                    snr_X = np.vstack([sig_feats, noise_feats]).astype(np.float32)
                    snr_y = np.array([1]*len(sig_feats) + [0]*len(noise_feats), dtype=np.float32)
                    domain_data[domain_idx]['features'].append(snr_X)
                    domain_data[domain_idx]['labels'].append(snr_y)

        # Apply sampling per SNR
        for domain_idx, data in domain_data.items():
            X_domain = np.vstack(data['features'])
            y_domain_labels = np.concatenate(data['labels'])
            n_samples = min(config.MAX_TRAIN_SAMPLES_PER_SNR_ML, len(X_domain))
            indices = np.random.choice(len(X_domain), n_samples, replace=False)

            all_features.append(X_domain[indices])
            all_class_labels.append(y_domain_labels[indices])
            all_domain_labels.extend([domain_idx] * n_samples)

        X, y_class = np.vstack(all_features), np.concatenate(all_class_labels)
        y_domain = np.array(all_domain_labels)

        # Apply total sampling limit
        if len(X) > config.MAX_TOTAL_TRAIN_SAMPLES_ML:
            indices = np.random.choice(len(X), config.MAX_TOTAL_TRAIN_SAMPLES_ML, replace=False)
            X, y_class, y_domain = X[indices], y_class[indices], y_domain[indices]

        if X.shape[0] == 0:
            raise ValueError("No features were loaded for ML training.")

        return X, y_class, y_domain

    def _get_hdf5_dset_name(self, m_val: int, window_type: str) -> str:
        """Finds the correct dataset name in the HDF5 file."""
        pattern = f"{window_type.lower()}_m{m_val}"
        for name in self.dsets:
            if pattern in name:
                return name
        raise ValueError(f"Could not find dataset for M={m_val}, Window={window_type}")

    @staticmethod
    def _create_sliding_windows(data_ts: np.ndarray, m_val: int) -> np.ndarray:
        """Creates sliding window features from a time series of ORD values."""
        n_windows = data_ts.shape[1]
        n_strides = n_windows - m_val + 1
        if n_strides <= 0:
            return np.array([])
        
        shape = (data_ts.shape[0] * n_strides, m_val)
        strides = (data_ts.strides[0], data_ts.strides[1], data_ts.strides[1])
        sub_windows = np.lib.stride_tricks.as_strided(
            data_ts, shape=(data_ts.shape[0], n_strides, m_val), strides=strides
        )
        return sub_windows.reshape(shape)