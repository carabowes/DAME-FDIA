from __future__ import annotations
from typing import Optional

import numpy as np
from sklearn.svm import OneClassSVM
from src.ml.detectors.base import BaseAnomalyDetector

class OneClassSVMDetector(BaseAnomalyDetector):
    """
    One-Class SVM based anomaly detector operating on windowed feature vectors. Trained on class windows only and produces
    continuous anomaly scores and binary alarm decisions via quantile-based thresholding
    """
    def __init__(
        self,
        kernel: str = "rbf",
        nu: float = 0.05,
        gamma: str | float = "scale",
        threshold_quantile: float = 95.0,
        **kwargs,
    ):
        super().__init__(
            kernel=kernel,
            nu=nu,
            gamma=gamma,
            threshold_quantile=threshold_quantile,
            **kwargs,
        )
        self.model= OneClassSVM(
            kernel=kernel,
            nu=nu,
            gamma=gamma,
        )
        self.threshold_quantile = threshold_quantile
        self._tau: Optional[float] = None

    def fit(self, X: np.ndarray, clean_mask: Optional[np.ndarray] = None) -> None:

        # Fit one-class SVM on windowed feature matrix. If a clean_mask is provided, only windows marked as clean (mask ==1) are used for training.
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

        # print(
        #     "DEBUG:",
        #     "clean_mask provided =", clean_mask is not None,
        #     "num_clean =", None if clean_mask is None else int(clean_mask.sum()),
        #     "num_windows =", X.shape[0],
        # )

    def score(self, X: np.ndarray) -> np.ndarray:
        # compute continuous anomaly scores, higher values - more anomalous
        if X.ndim != 2:
            raise ValueError("X must be a 2D array of shape (n_windows, n_features)")

        return -self.model.score_samples(X)
    
    def threshold(self, scores: np.ndarray) -> float:
        # Compute anomaly decision threshold t using a fixed quantile
        if scores.ndim != 1:
            raise ValueError("scores must be a 1D array")
        
        if self._tau is None:
            raise RuntimeError("Detector must be fitted before thresholding")
        
        return self._tau
        