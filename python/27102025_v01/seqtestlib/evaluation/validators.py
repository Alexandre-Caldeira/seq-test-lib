# seqtestlib/evaluation/validators.py (Corrected)
"""
Handles the cross-validation workflow for evaluating detector performance.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import ShuffleSplit
from tqdm import tqdm
import h5py

from .. import config
from ..data.loaders import SimulatedLoader
from ..models.base import BaseDetector
from ..models.ml import LGBMModel 
from ..models.dl import DANNModel
from ..sequential.testers import BaseTester
from . import metrics

class Evaluator:
    """
    Orchestrates the cross-validation process for a given sequential tester.
    """
    def __init__(self, tester: BaseTester, m_val: int):
        self.tester = tester
        self.m_val = m_val

    def run_simulated_cv(self, loader: SimulatedLoader, job: dict):
        results = []
        num_trials_for_cv = min(config.MAX_SIM_TRIALS_FOR_ML_TRAINING, loader.num_trials)
        cv = ShuffleSplit(n_splits=config.N_SPLITS_SIM, test_size=config.TEST_SIZE_SIM, random_state=config.RANDOM_STATE)
        
        hdf5_dset_name = loader._get_hdf5_dset_name(job['m_val'], job['window_type'])
        
        data_source = {}
        with h5py.File(loader.filepath, 'r') as f:
            for snr_key in loader.snr_keys:
                snr_val = float(snr_key.replace('snr_', ''))
                data_source[snr_val] = f[snr_key][hdf5_dset_name][:num_trials_for_cv]

        fold_pbar = tqdm(enumerate(cv.split(np.arange(num_trials_for_cv))), total=config.N_SPLITS_SIM, desc=f"CV Folds for {job['feature_name']}", leave=False)

        for fold_idx, (train_indices, test_indices) in fold_pbar:
            self._fit_or_set_threshold(data_source, train_indices, loader.noise_bins, job)

            for snr, snr_data in data_source.items():
                test_data_fold = snr_data[test_indices]
                if test_data_fold.shape[0] == 0: continue

                # The tester now handles the complexity of processing the data block
                _, detections_over_time, ttfd_per_bin = self.tester.test(test_data_fold, job['feature_idx'])
                
                # metrics.py calculates the final per-trial outcomes
                tpr_flags, fpr_flags, ttfd_values = metrics.calculate_trial_performance(
                    detections=detections_over_time,
                    signal_bins=loader.signal_bins,
                    noise_bins=loader.noise_bins,
                    m_value=job['m_val'],
                    detection_method=job['detection_method'],
                    num_windows=test_data_fold.shape[1]
                )
                
                # Append per-trial results
                for i in range(len(test_data_fold)):
                    results.append({
                        'SNR': snr, 'TPR': tpr_flags[i], 'FPR': fpr_flags[i], 'TTFD': ttfd_values[i],
                        'Alpha (%)': self.tester.alpha, 'Feature': job['feature_name'],
                        'Window': job['window_type'], 'Detection': job.get('detection_method', 'ML')
                    })

        return pd.DataFrame(results)

    # _fit_or_set_threshold and its helpers remain the same as the previous version
    def _fit_or_set_threshold(self, data_source: dict, train_indices: np.ndarray, noise_bins: list, job: dict):
        """Helper to prepare the model for a cross-validation fold."""
        
        if isinstance(self.tester.model, (DANNModel, LGBMModel)):
            X, y, domains = self._prepare_ml_data(data_source, train_indices, noise_bins, job)
            if X.shape[0] > 0:
                fit_kwargs = {'domains': domains} if isinstance(self.tester.model, DANNModel) else {}
                self.tester.model.fit(X, y, **fit_kwargs)
        
        noise_dist_data = self._get_noise_distribution(data_source, train_indices, noise_bins, job)
        self.tester.set_threshold(noise_dist_data)

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

            if sig_feats.size > 0 or noise_feats.size > 0:
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
        
        noise_ts_list = [
            data_source[snr][indices][:, :, noise_bins, job['feature_idx']].transpose(0, 2, 1).reshape(-1, data_source[snr].shape[1])
            for snr in data_source
        ]
        full_noise_ts = np.concatenate(noise_ts_list)
        
        if isinstance(self.tester.model, (LGBMModel, DANNModel)):
            noise_windows = SimulatedLoader._create_sliding_windows(full_noise_ts, self.m_val)
            if noise_windows.shape[0] == 0: return np.array([])
            return self.tester.model.predict_score(noise_windows)
        else:
            return full_noise_ts