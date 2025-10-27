# seqtestlib/models/base.py
"""
Defines the abstract base class for all detector models.
Ensures a consistent API for fitting models and predicting scores.
"""

from abc import ABC, abstractmethod
import numpy as np

class BaseDetector(ABC):
    """
    Abstract Base Class for all orienting response detectors.

    This class defines the common interface that all detector models must implement,
    ensuring they can be used interchangeably within the evaluation framework.
    """

    def __init__(self, **kwargs):
        """
        Initializes the detector. Any model-specific parameters can be passed here.
        """
        self.model = None
        self.params = kwargs
        super().__init__()

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs):
        """
        Trains or fits the detector model.

        For statistical models, this might involve calculating a null distribution.
        For ML/DL models, this involves training the model on labeled data.

        Args:
            X (np.ndarray): Training data features. For ML models, this is typically
                            an array of windowed ORD values. For statistical models,
                            this could be a flattened array of noise values.
            y (np.ndarray): Training data labels (e.g., 0 for noise, 1 for signal).
                            May not be used by all models.
            **kwargs: Additional arguments for fitting, such as domain labels for DANN.
        """
        pass

    @abstractmethod
    def predict_score(self, X: np.ndarray) -> np.ndarray:
        """
        Generates a "likelihood of signal" score for each input window.

        For statistical models, this is often the raw feature value itself.
        For ML/DL models, this is typically the model's output probability.

        Args:
            X (np.ndarray): An array of feature windows to score.

        Returns:
            np.ndarray: An array of scores, one for each input window.
        """
        pass

    def get_params(self) -> dict:
        """
        Returns the parameters of the detector.
        """
        return self.params