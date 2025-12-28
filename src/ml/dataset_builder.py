import numpy as np
from src.ml.windowing import generate_sliding_windows

def build_dataset(
    Z: np.ndarray,
    window_size: int,
    convergence_mask: np.ndarray | None = None,
    stride: int = 1,
    representation: str = "flattened",
):
    # Build dataset of sliding windows from time-series measurements Z.
    # Z: time series matrix of shape (T, d)
    # window_size: number of time steps per window
    # convergence_mask: boolean array of length T indicating convergence at each timestep
    # stride: step size for sliding window
    # window representation: "flattened" for now - baseline
    windows , metadata = generate_sliding_windows(
        Z=Z,
        window_size=window_size,
        convergence_mask=convergence_mask,
        stride=stride,
    )

    if representation == "flattened":
        # Flatten each window to a 1D array
        num_windows, W, d = windows.shape
        X_windows = windows.reshape(num_windows, W * d)
    else:
        raise ValueError(f"Unsupported representation: {representation}")
    
    # Where X is the dataset of shape (num_windows, features)
    # Metadata including window indices and discarded counts
    return X_windows, metadata