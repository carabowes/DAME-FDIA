from __future__ import annotations
from typing import Optional

import numpy as np
from sklearn.neighbors import LocalOutlierFactor
from src.ml.detectors.base import BaseAnomalyDetector

class LOFDetector(BaseAnomalyDetector):
    """
    Local Outlier Factor (LOF) based anomaly detector operating on windowed feature vectors.

    Trained on clean windows only and produces continuous anomaly scores via negative LOF,
    followed by quantile-based thresholding for alarm generation.
    """

    def __init__(
        self,
        n_neighbors: int = 20,
        contamination: float = 0.01,
        threshold_quantile: float = 95.0,
        **kwargs,
    ):
        super().__init__(
        n_neighbors=n_neighbors,
        threshold_quantile=threshold_quantile,
        **kwargs,
        )

        self.n_neighbors = n_neighbors
        self.threshold_quantile = threshold_quantile
        
        # novelty=True allows scoring on unseen data
        self.model = LocalOutlierFactor(
            n_neighbors=n_neighbors,
            contamination="auto",
            novelty=True,
        )
        self._tau: Optional[float] = None

    def fit(self, X: np.ndarray, clean_mask: Optional[np.ndarray] = None) -> None:
        
        if X.ndim != 2:
            raise ValueError("X must be a 2D array")
        
        if clean_mask is not None:
            X_fit = X[clean_mask==1]
        else:
            X_fit = X
        
        if X_fit.shape[0] ==0:
            raise ValueError("No clean samples available for training One-class SVM")
        
        self.model.fit(X_fit)

        # Compute anomaly scores on training data to set threshold
        scores = self.score(X_fit)
        self._tau = np.percentile(scores, self.threshold_quantile)
        self._is_fitted = True

    def score(self, X: np.ndarray) -> np.ndarray:
        # compute continuous anomaly scores, higher values - more anomalous
        if X.ndim != 2:
            raise ValueError("X must be a 2D array of shape (n_windows, n_features)")

        return -self.model.score_samples(X)
    
    def threshold(self, scores: np.ndarray) -> float:
        # Compute anomaly decision threshold t using a fixed quantile
        # if scores.ndim != 1:
        #     raise ValueError("scores must be a 1D array")
        
        if self._tau is None:
            raise RuntimeError("Detector must be fitted before thresholding")
        
        return self._tau
        