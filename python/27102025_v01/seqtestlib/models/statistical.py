# seqtestlib/models/statistical.py
"""
Implements a wrapper for standard statistical ORD methods (e.g., MSC, CSM).
This model acts as a pass-through, where the "score" is the raw feature value.
"""

import numpy as np
from .base import BaseDetector

class StatisticalModel(BaseDetector):
    """
    A simple pass-through detector for standard ORD metrics.

    This class conforms to the BaseDetector interface but does not perform any
    training. It is used to represent the traditional thresholding approach
    within the unified framework.
    """

    def fit(self, X: np.ndarray = None, y: np.ndarray = None, **kwargs):
        """
        Does nothing, as statistical models are threshold-based and do not require fitting.
        This method is included to satisfy the abstract base class requirements.
        """
        # No fitting is necessary for this model.
        # The threshold is determined externally from the null distribution.
        pass

    def predict_score(self, X: np.ndarray) -> np.ndarray:
        """
        Returns the input feature values directly as scores.

        For standard ORD, the "score" of a window is simply its calculated
        feature value (e.g., the MSC value).

        Args:
            X (np.ndarray): An array of ORD feature values (or windows of values).
                            If windows are passed, the score is taken from the last
                            value in the window, representing the current time step.

        Returns:
            np.ndarray: The raw feature values, which will be used as scores.
        """
        if X.ndim > 1:
            # If input is a set of windows, the relevant score is the most recent one.
            return X[:, -1]
        else:
            # If input is already a 1D array of scores.
            return X