# seqtestlib/sequential/thresholds.py
"""
Functions for determining decision thresholds for detector models.
"""

import numpy as np
from typing import Union

# Import base classes for type checking
from ..models.base import BaseDetector
from ..models.statistical import StatisticalModel
from ..models.ml import LGBMModel
from ..models.dl import DANNModel

def determine_threshold(
    detector: BaseDetector,
    noise_data: np.ndarray,
    alpha: float
) -> float:
    """
    Determines the decision threshold for a given detector and alpha level.

    The method differs based on the detector type:
    - For StatisticalModel: The threshold is the (100-alpha)th percentile of the
      raw noise feature values.
    - For ML/DL Models: The threshold is the (100-alpha)th percentile of the
      model's output scores when applied to the noise feature windows.

    Args:
        detector (BaseDetector): The fitted detector model instance.
        noise_data (np.ndarray): The data used to create the null distribution.
                                 - For StatisticalModel: A flattened array of ORD values from noise bins.
                                 - For ML/DL models: An array of feature windows from noise bins.
        alpha (float): The desired false positive rate in percent (e.g., 5.0 for 5%).

    Returns:
        float: The calculated threshold value.
    """
    if isinstance(detector, StatisticalModel):
        # For statistical models, the raw noise data *is* the null distribution.
        null_distribution = noise_data.flatten()
    else:
        # For ML/DL models, we first get the model's output scores on the noise data.
        # This array of scores becomes our null distribution.
        if noise_data.shape[0] == 0:
            # No noise windows to evaluate, return a safe high threshold
            return 1.0 if isinstance(detector, LGBMModel) else np.inf

        null_distribution = detector.predict_score(noise_data)

    if null_distribution.size == 0:
        # Handle edge case where no noise data is available for percentile calculation.
        # Return a high threshold to prevent false positives.
        return 1.0 if isinstance(detector, LGBMModel) else np.inf

    # Calculate the threshold at the (100 - alpha) percentile.
    threshold = np.percentile(null_distribution, 100 - alpha)
    return threshold