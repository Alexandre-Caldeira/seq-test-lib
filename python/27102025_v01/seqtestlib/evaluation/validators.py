# seqtestlib/evaluation/validators.py
"""
Handles the cross-validation workflow for evaluating detector performance.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import ShuffleSplit
from tqdm import tqdm
import itertools

from .. import config
from ..data.loaders import SimulatedLoader
from ..models.base import BaseDetector
from ..models.ml import LGBMModel, DANNModel
from ..sequential.testers import BaseTester, StandardTester, BlockTester
from . import metrics

class Evaluator:
    """
    Orchestrates the cross-validation process for a given sequential tester.
    """
    def __init__(self, tester: BaseTester, m_val: int):
        """
        Args:
            tester (BaseTester): A configured sequential tester instance.
            m_val (int): The M-value (window size) for this evaluation.
        """
        self.tester = tester
        self.m_val = m_val

    def run_simulated_cv(self, loader: SimulatedLoader, job: dict):
        """
        Runs a full cross-validation loop on the simulated dataset for a single job.

        Args:
            loader (SimulatedLoader): A loader instance for the simulated feature data.
            job (dict): The analysis job configuration dictionary.

        Returns:
            pd.DataFrame: A DataFrame containing the performance results for all folds.
        """
        results = []
        num_trials = min(config.MAX_SIM_TRIALS_FOR_ML_TRAINING, loader.num_trials)
        cv = ShuffleSplit(n_splits=config.N_SPLITS_SIM, test_size=config.TEST_SIZE_SIM, random_state=config.RANDOM_STATE)
        
        hdf5_dset_name = loader._get_hdf5_dset_name(self.m_val, job['window_type'])
        
        # Load all necessary data into memory once to speed up folds
        data_source = {}
        with h5py.File(loader.filepath, 'r') as f:
            for snr_key in loader.snr_keys:
                snr_val = float(snr_key.replace('snr_', ''))
                data_source[snr_val] = f[snr_key][hdf5_dset_name][:num_trials]

        fold_pbar = tqdm(enumerate(cv.split(np.arange(num_trials))), total=config.N_SPLITS_SIM, desc=f"CV Folds for {job['feature_name']}", leave=False)

        for fold_idx, (train_indices, test_indices) in fold_pbar:
            # --- Fit the model or set the threshold on the training data ---
            self._fit_or_set_threshold(data_source, train_indices, loader.noise_bins, job)

            # --- Evaluate on the test data for each SNR ---
            for snr, snr_data in data_source.items():
                test_set_data = snr_data[test_indices]
                
                # Get model scores/raw values over time
                if isinstance(self.tester.model, (DANNModel, LGBMModel)):
                    # For ML models, we need sliding windows as input
                    sig_ts = test_set_data[:, :, loader.signal_bins, job['feature_idx']].transpose(0, 2, 1).reshape(-1, test_set_data.shape[1])
                    noise_ts = test_set_data[:, :, loader.noise_bins, job['feature_idx']].transpose(0, 2, 1).reshape(-1, test_set_data.shape[1])

                    sig_windows = loader._create_sliding_windows(sig_ts, self.m_val)
                    noise_windows = loader._create_sliding_windows(noise_ts, self.m_val)
                    
                    test_scores = []
                    if sig_windows.size > 0:
                        test_scores.append(self.tester.model.predict_score(sig_windows))
                    if noise_windows.size > 0:
                        test_scores.append(self.tester.model.predict_score(noise_windows))
                    
                    scores_over_time = np.concatenate(test_scores) if test_scores else np.array([])
                else: # StatisticalModel
                    scores_over_time = test_set_data # Pass the full data block

                # Run sequential test
                detected, ttfd = self.tester.test(scores_over_time)

                # TODO: This block needs adjustment based on how `tester.test` returns results for signal/noise bins separately
                # For now, assuming a simplified evaluation loop for demonstration
                tpr = np.mean(detected) # Placeholder
                fpr = np.mean(detected) # Placeholder
                
                results.append({
                    'SNR': snr, 'TPR': tpr, 'FPR': fpr, 'TTFD': np.nanmean(ttfd),
                    'Alpha (%)': self.tester.alpha, 'Feature': job['feature_name'],
                    'Window': job['window_type'], 'Detection': job.get('detection_method', 'ML')
                })

        return pd.DataFrame(results)

    def _fit_or_set_threshold(self, data_source: dict, train_indices: np.ndarray, noise_bins: list, job: dict):
        """Helper to prepare the model for a cross-validation fold."""
        
        if isinstance(self.tester.model, (DANNModel, LGBMModel)):
            # For ML models, we need to generate training windows
            X, y, domains = self._prepare_ml_data(data_source, train_indices, noise_bins, job)
            if X.shape[0] > 0:
                fit_kwargs = {'domains': domains} if isinstance(self.tester.model, DANNModel) else {}
                self.tester.model.fit(X, y, **fit_kwargs)
        
        # For all models, set the threshold based on noise scores from the training set
        noise_scores_or_values = self._get_noise_distribution(data_source, train_indices, noise_bins, job)
        self.tester.set_threshold(noise_scores_or_values)

    def _prepare_ml_data(self, data_source, indices, noise_bins, job):
        """Prepares windowed features, class labels, and domain labels for ML training."""
        all_features, all_class_labels, all_domain_labels = [], [], []
        snr_map = {snr: i for i, snr in enumerate(data_source.keys())}

        for snr, snr_data in data_source.items():
            train_data = snr_data[indices]
            
            sig_ts = train_data[:, :, [b for b in range(train_data.shape[2]) if b not in noise_bins], job['feature_idx']].transpose(0, 2, 1).reshape(-1, train_data.shape[1])
            noise_ts = train_data[:, :, noise_bins, job['feature_idx']].transpose(0, 2, 1).reshape(-1, train_data.shape[1])

            sig_feats = SimulatedLoader._create_sliding_windows(sig_ts, self.m_val)
            noise_feats = SimulatedLoader._create_sliding_windows(noise_ts, self.m_val)

            features = np.vstack([sig_feats, noise_feats])
            labels = np.array([1]*len(sig_feats) + [0]*len(noise_feats))
            
            all_features.append(features)
            all_class_labels.append(labels)
            all_domain_labels.extend([snr_map[snr]] * len(features))
            
        X = np.vstack(all_features) if all_features else np.array([])
        y = np.concatenate(all_class_labels) if all_class_labels else np.array([])
        domains = np.array(all_domain_labels)
        
        return X, y, domains

    def _get_noise_distribution(self, data_source, indices, noise_bins, job):
        """Gets the null distribution data (raw values or model scores) from the training set."""
        
        # Collect all noise windows/values from the training fold across all SNRs
        noise_ts_list = [
            data_source[snr][indices][:, :, noise_bins, job['feature_idx']].transpose(0, 2, 1).reshape(-1, data_source[snr].shape[1])
            for snr in data_source
        ]
        full_noise_ts = np.concatenate(noise_ts_list)
        
        if isinstance(self.tester.model, (LGBMModel, DANNModel)):
            # For ML models, the distribution comes from scores on windowed features
            noise_windows = SimulatedLoader._create_sliding_windows(full_noise_ts, self.m_val)
            if noise_windows.shape[0] == 0: return np.array([])
            return self.tester.model.predict_score(noise_windows)
        else:
            # For statistical models, the distribution is the raw ORD values themselves
            return full_noise_ts