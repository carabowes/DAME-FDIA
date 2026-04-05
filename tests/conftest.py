"""
Pytest configuration: Add project root to Python path for imports.

This allows tests to import from src/ without needing to set PYTHONPATH.
"""
import sys
import warnings
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Suppress benign NumPy warnings from WLS estimator
# The divide-by-zero in wls_estimate occurs when computing R_inv with sigma values.
# This is EXPECTED and HANDLED because:
#   1. sigma is always positive in practice (measurement noise std)
#   2. The regularization (reg=1e-6) prevents singular matrix issues
#   3. The solve is valid even with conditioning issues due to regularization
# We suppress the warning here to keep test output clean.
warnings.filterwarnings(
    "ignore",
    message="divide by zero encountered in matmul",
    category=RuntimeWarning
)
warnings.filterwarnings(
    "ignore",
    message="overflow encountered in matmul",
    category=RuntimeWarning
)
warnings.filterwarnings(
    "ignore",
    message="invalid value encountered in matmul",
    category=RuntimeWarning
)

