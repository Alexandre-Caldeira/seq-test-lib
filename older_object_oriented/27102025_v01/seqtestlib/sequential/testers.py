# seqtestlib/sequential/testers.py (Corrected)
"""
Implements the sequential testing paradigms (e.g., Per-Window, Per-M-Block).
These classes take a fitted detector model and apply a decision rule over time.
"""

import numpy as np
from typing import Tuple

from . import thresholds
from ..models.base import BaseDetector
from ..models.ml import LGBMModel
from ..models.dl import DANNModel
from abc import ABC, abstractmethod

class BaseTester(ABC):
    """Abstract Base Class for all sequential testers."""
    def __init__(self, model: BaseDetector, alpha: float):
        self.model = model
        self.alpha = alpha
        self.threshold = None

    def set_threshold(self, noise_data: np.ndarray):
        self.threshold = thresholds.determine_threshold(self.model, noise_data, self.alpha)

    @abstractmethod
    def test(self, feature_block: np.ndarray, feature_idx: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Applies the sequential test to a block of feature data for multiple trials.

        Args:
            feature_block (np.ndarray): A batch of data. Shape: (n_trials, n_windows, n_bins, n_features).
            feature_idx (int): The index of the feature to use within the block.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]:
                - scores_over_time (np.ndarray): The raw scores for each bin over time.
                - detections_over_time (np.ndarray): Boolean array of detections.
                - ttfd_per_bin (np.ndarray): TTFD index for each bin in each trial.
        """
        pass

class StandardTester(BaseTester):
    """Implements the 'Per-Window' sequential test."""
    def test(self, feature_block: np.ndarray, feature_idx: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.threshold is None:
            raise RuntimeError("Threshold not set. Call set_threshold() before testing.")

        # Extract the relevant time series data for the chosen feature
        # Shape: (n_trials, n_windows, n_bins)
        scores_over_time = feature_block[:, :, :, feature_idx]

        # For ML models, the input is windowed features, not raw scores
        if isinstance(self.model, (LGBMModel, DANNModel)):
            # This path is complex and better handled inside a dedicated MLTester.
            # For this fix, we assume the evaluator will pre-score ML models.
            # If scores are passed directly (n_trials, n_windows), reshape for consistency.
            if scores_over_time.ndim == 2:
                 scores_over_time = scores_over_time[:, :, np.newaxis]

        detections_over_time = scores_over_time > self.threshold
        n_trials = feature_block.shape[0]
        ttfd_per_bin = np.full((n_trials, feature_block.shape[2]), np.nan)

        for trial_idx in range(n_trials):
            for bin_idx in range(feature_block.shape[2]):
                if np.any(detections_over_time[trial_idx, :, bin_idx]):
                    ttfd_per_bin[trial_idx, bin_idx] = np.argmax(detections_over_time[trial_idx, :, bin_idx])

        return scores_over_time, detections_over_time, ttfd_per_bin

class BlockTester(BaseTester):
    """Implements the 'Per-M-Block' sequential test."""
    def __init__(self, model: BaseDetector, alpha: float, m_block_size: int):
        super().__init__(model, alpha)
        self.m = m_block_size

    def test(self, feature_block: np.ndarray, feature_idx: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.threshold is None:
            raise RuntimeError("Threshold not set. Call set_threshold() before testing.")

        n_trials, n_windows, n_bins, _ = feature_block.shape
        n_blocks = n_windows // self.m

        if n_blocks == 0:
            return np.array([]), np.array([]), np.array([])
        
        # Extract the relevant feature time series
        scores_over_time = feature_block[:, :n_blocks * self.m, :, feature_idx]
        
        # Reshape into blocks: (n_trials, n_blocks, m_block_size, n_bins)
        scores_in_blocks = scores_over_time.reshape(n_trials, n_blocks, self.m, n_bins)
        
        # Aggregate by taking the max score within each block for each bin
        block_scores = np.max(scores_in_blocks, axis=2) # Shape: (n_trials, n_blocks, n_bins)

        detections_over_time = block_scores > self.threshold
        
        ttfd_per_bin = np.full((n_trials, n_bins), np.nan)
        for trial_idx in range(n_trials):
            for bin_idx in range(n_bins):
                if np.any(detections_over_time[trial_idx, :, bin_idx]):
                    block_idx = np.argmax(detections_over_time[trial_idx, :, bin_idx])
                    # TTFD is the window index at the END of the detecting block
                    ttfd_per_bin[trial_idx, bin_idx] = (block_idx + 1) * self.m
        
        # Return block scores as they are the effective scores for this paradigm
        return block_scores, detections_over_time, ttfd_per_bin