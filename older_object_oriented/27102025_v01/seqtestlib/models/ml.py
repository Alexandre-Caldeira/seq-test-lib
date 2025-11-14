# seqtestlib/models/ml.py
"""
Implements the LightGBM (LGBM) classifier model for ORD detection.
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
from .base import BaseDetector
from .. import config

class LGBMModel(BaseDetector):
    """A LightGBM model for binary classification."""
    def __init__(self):
        self.model = lgb.LGBMClassifier(random_state=42)

    def fit(self, X, y, **kwargs):
        """
        Fits the LGBM model.
        Converts X to a NumPy array to avoid feature name warnings.
        """
        # If X is a pandas DataFrame, convert it to a NumPy array
        if isinstance(X, pd.DataFrame):
            X_train = X.values
        else:
            X_train = X
            
        self.model.fit(X_train, y)

    def predict_score(self, X):
        """
        Predicts the probability scores for the positive class.
        """
        # If X is a pandas DataFrame, convert it to a NumPy array
        if isinstance(X, pd.DataFrame):
            X_test = X.values
        else:
            X_test = X
            
        return self.model.predict_proba(X_test)[:, 1]

# class LGBMModel(BaseDetector):
#     """
#     A detector based on the LightGBM gradient boosting framework.

#     This model is trained as a binary classifier on windowed ORD feature vectors
#     to distinguish between signal-present and signal-absent patterns.
#     """

#     def __init__(self, **kwargs):
#         """
#         Initializes the LGBMClassifier with parameters from the global config.
#         """
#         super().__init__(**kwargs)
#         # Override default params with any kwargs passed during instantiation
#         params = {**config.LGBM_PARAMS, **self.params}
#         self.model = lgb.LGBMClassifier(**params)

#     def fit(self, X: np.ndarray, y: np.ndarray, **kwargs):
#         """
#         Trains the LightGBM model.

#         Args:
#             X (np.ndarray): Training data features (windowed ORD values).
#                             Shape: (n_samples, m_val).
#             y (np.ndarray): Training data labels (0 or 1).
#             **kwargs: Additional arguments for compatibility (not used by LGBM).
#         """
#         print(f"  Training LGBM model on {X.shape[0]} samples...")
#         self.model.fit(X, y)

#     def predict_score(self, X: np.ndarray) -> np.ndarray:
#         """
#         Predicts the probability of the positive class (signal present).

#         Args:
#             X (np.ndarray): An array of feature windows to score.
#                             Shape: (n_samples, m_val).

#         Returns:
#             np.ndarray: An array of probability scores, one for each input window.
#         """
#         if not hasattr(self.model, 'classes_'):
#             raise RuntimeError("Model has not been fitted yet. Call fit() before predicting.")
            
#         # predict_proba returns probabilities for [class_0, class_1]
#         # We want the probability of the positive class (1).
#         return self.model.predict_proba(X)[:, 1]

#     def get_params(self) -> dict:
#         """Returns the effective parameters of the LGBM model."""
#         return self.model.get_params()