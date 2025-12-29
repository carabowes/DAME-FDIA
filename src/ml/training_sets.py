import numpy as np
from src.ml.dataset_builder import build_dataset

def build_normal_training_set(
        Z: np.ndarray,
        window_size: int,
        attack_mask: np.ndarray,
        convergence_mask: np.ndarray | None = None,
        stride: int = 1,
):
    # Build a normal-operation training dataset for unsupervised models.
    # A window is considered normal if all timesteps within the window are non-attacked.
    # Convergence mask can be provided to filter out non-converged windows.

    # Validation
    T, d = Z.shape
    if attack_mask.shape[0] != T:
        raise ValueError("attack_mask length must match number of timesteps in Z")
    
    if len(train_indices) == 0:
        raise ValueError("No normal windows available for training.")


    # Build windowed feature matrix
    X_all, window_metadata = build_dataset(
        Z=Z,
        window_size=window_size,
        convergence_mask=convergence_mask,
        stride=stride,
        representation="flattened",
    )

    # Identify normal windows
    train_indices = []
    for i, start_t in enumerate(window_metadata["start_indices"]):
        end_t = start_t + window_size
        is_normal_window = not np.any(attack_mask[start_t:end_t])
        if is_normal_window:
            train_indices.append(i)
    
    # Extract training features for normal windows
    X_train = X_all[train_indices]
    train_metadata = {
        "num_total_windows": window_metadata["num_windows"],
        "num_normal_windows": len(train_indices),
        "window_size": window_size,
        "stride": stride,
        "train_indices": train_indices,
    }
    return X_train, train_metadata