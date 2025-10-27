# seqtestlib/sequential/testers.py
"""
Implements the sequential testing paradigms (e.g., Per-Window, Per-M-Block).
These classes take a fitted detector model and apply a decision rule over time.
"""

from abc import ABC, abstractmethod
import numpy as np
from typing import Union, Tuple

from . import thresholds
from ..models.base import BaseDetector

class BaseTester(ABC):
    """
    Abstract Base Class for all sequential testers.
    """
    def __init__(self, model: BaseDetector, alpha: float):
        """
        Initializes the tester with a model and a significance level.

        Args:
            model (BaseDetector): The detector model instance (can be statistical, ML, or DL).
            alpha (float): The target false positive rate in percent (e.g., 5.0).
        """
        self.model = model
        self.alpha = alpha
        self.threshold = None

    def set_threshold(self, noise_data: np.ndarray):
        """
        Calculates and sets the decision threshold using the provided noise data.

        Args:
            noise_data (np.ndarray): Data from noise-only conditions, formatted as
                                     required by the specific model (raw values or windows).
        """
        self.threshold = thresholds.determine_threshold(self.model, noise_data, self.alpha)

    @abstractmethod
    def test(self, feature_windows: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Applies the sequential test to a series of feature windows for multiple trials.

        Args:
            feature_windows (np.ndarray): A batch of data to be tested.
                                          Shape: (n_trials, n_windows, feature_dim).

        Returns:
            Tuple[np.ndarray, np.ndarray]:
                - A boolean array indicating if a detection occurred in each trial.
                - An array with the time-to-first-detection (index) for each trial.
                  If no detection, value is np.nan.
        """
        pass

class StandardTester(BaseTester):
    """
    Implements the 'Per-Window' sequential test.

    A detection is declared for a trial if the score of *any* single window
    exceeds the threshold.
    """
    def test(self, feature_windows: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if self.threshold is None:
            raise RuntimeError("Threshold not set. Call set_threshold() before testing.")

        # For this tester, the model operates on single feature values, not windows.
        # We extract the last value of each window, which represents the current ORD value.
        # Shape becomes (n_trials, n_windows)
        scores_over_time = self.model.predict_score(feature_windows)

        # Find if any score in each trial exceeds the threshold
        detections_over_time = scores_over_time > self.threshold
        detected_trials = np.any(detections_over_time, axis=1)

        # Find the time-to-first-detection (TTFD)
        ttfd = np.full(feature_windows.shape[0], np.nan)
        if np.any(detected_trials):
            # argmax returns the index of the first 'True' value
            ttfd[detected_trials] = np.argmax(detections_over_time[detected_trials], axis=1)

        return detected_trials, ttfd

class BlockTester(BaseTester):
    """
    Implements the 'Per-M-Block' sequential test.

    The test data is divided into non-overlapping blocks of size M. A detection
    is declared if the maximum score within *any* block exceeds the threshold.
    """
    def __init__(self, model: BaseDetector, alpha: float, m_block_size: int):
        super().__init__(model, alpha)
        self.m = m_block_size

    def test(self, feature_windows: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if self.threshold is None:
            raise RuntimeError("Threshold not set. Call set_threshold() before testing.")

        n_trials, n_windows, _ = feature_windows.shape
        n_blocks = n_windows // self.m

        if n_blocks == 0:
            return np.zeros(n_trials, dtype=bool), np.full(n_trials, np.nan)

        # Truncate to fit full blocks and score all windows at once
        truncated_windows = feature_windows[:, :n_blocks * self.m, :]
        scores = self.model.predict_score(truncated_windows)

        # Reshape scores into blocks: (n_trials, n_blocks, m_block_size)
        scores_in_blocks = scores.reshape(n_trials, n_blocks, self.m)

        # Aggregate scores by taking the max within each block
        block_scores = np.max(scores_in_blocks, axis=2)

        # Find if any block score exceeds the threshold
        detections_over_time = block_scores > self.threshold
        detected_trials = np.any(detections_over_time, axis=1)

        # Find the time-to-first-detection (TTFD) in terms of block index
        ttfd = np.full(n_trials, np.nan)
        if np.any(detected_trials):
            block_idx = np.argmax(detections_over_time[detected_trials], axis=1)
            # Convert block index to window index (end of the block)
            ttfd[detected_trials] = (block_idx + 1) * self.m

        return detected_trials, ttfd