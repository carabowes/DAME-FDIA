from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, Any

class BaseAnomalyDetector(ABC):
    """
    Abstract base class for window-level anomaly detectors.
    Detectors operated on fixed-sized windowed feature matrices and return continuous anomaly
    scores and binary alarm decisions
    """

    def __init__(self, **config: Any):
        # Store detector config
        self.config = config
        self._is_fitted = False

    @abstractmethod
    def fit(self, X: np.ndarray) -> None:
        # Fit detector on normal (or mixed) windowed data.
        pass

    @abstractmethod
    def score(self, X: np.ndarray) -> np.ndarray:
        # Compute continuous anomaly scores for each window.
        pass

    @abstractmethod
    def threshold(self, scores: np.ndarray) -> float:
        #Compute the decision threshold from anomaly scores.
        pass

    def predict(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """ 
        Compute anomaly scores and binary alarms
        Returns dict with keys:
        - 'scores': continuous anomaly scores
        - 'alarms': binary alarm signal (0/1)
        """
        if not self._is_fitted:
            raise RuntimeError("Detector must be fitted before prediction")
        scores = self.score(X)
        tau = self.threshold(scores)
        alarms = (scores >= tau).astype(int)

        return {
            "scores": scores,
            "alarms": alarms,
        }
