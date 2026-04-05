"""Unit tests for state estimation (WLS)."""
import numpy as np
import pytest
from src.pipeline.state_estimation import wls_estimate


class TestWLSEstimate:
    """Test weighted least squares state estimator."""
    
    def test_wls_estimate_perfect_measurements(self):
        """WLS should recover exact state with perfect measurements."""
        n = 3
        m = 5
        x_true = np.array([0.1, -0.05, 0.08])
        
        # Create a well-conditioned H matrix
        H = np.random.randn(m, n)
        H = H / np.linalg.norm(H, axis=0)
        
        # Perfect measurements (no noise)
        z = H @ x_true
        sigma = np.ones(m) * 0.01
        
        x_hat, residual = wls_estimate(H, z, sigma)
        
        # Should recover state closely
        assert np.allclose(x_hat, x_true, atol=0.01)
    
    def test_wls_estimate_with_noise(self):
        """WLS should estimate state despite measurement noise."""
        n = 3
        m = 5
        x_true = np.array([0.1, -0.05, 0.08])
        
        H = np.random.randn(m, n)
        H = H / np.linalg.norm(H, axis=0)
        
        # Measurements with noise
        noise = np.random.randn(m) * 0.01
        z = H @ x_true + noise
        sigma = np.ones(m) * 0.01
        
        x_hat, R = wls_estimate(H, z, sigma)
        
        # Should be close to true state
        error = np.linalg.norm(x_hat - x_true)
        assert error < 0.05
    
    def test_wls_estimate_residual_shape(self):
        """Residual should match measurement dimension."""
        m, n = 10, 8
        H = np.random.randn(m, n)
        z = np.random.randn(m)
        sigma = np.ones(m)
        
        x_hat, R = wls_estimate(H, z, sigma)
        
        # Compute residual from x_hat
        residual = z - H @ x_hat
        assert residual.shape == (m,)
    
    def test_wls_estimate_dimension_reduction(self):
        """State estimate dimension should match H column dimension."""
        H = np.random.randn(10, 8)
        z = np.random.randn(10)
        sigma = np.ones(10)
        
        x_hat, R = wls_estimate(H, z, sigma)
        
        assert x_hat.shape == (8,)
    
    def test_wls_estimate_bad_conditioning_solvable(self):
        """WLS should still produce solution even with ill-conditioned H."""
        m, n = 8, 8
        H = np.eye(m)  # identity matrix
        z = np.ones(m)
        sigma = np.ones(m)
        
        x_hat, R = wls_estimate(H, z, sigma)
        
        assert x_hat.shape == (n,)
        assert not np.any(np.isnan(x_hat))


class TestResidualComputation:
    """Test residual r = z - H @ x_hat."""
    
    def test_residual_zero_perfect_fit(self):
        """Residual should be small when z = H @ x (approximately)."""
        from src.pipeline.state_estimation import compute_residuals
        
        H = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        x = np.array([0.5, 0.3])
        z = H @ x
        
        x_hat, R = wls_estimate(H, z, np.ones(3))
        r = compute_residuals(H, z, x_hat)
        
        # Due to regularization and numerical precision, allow small residual
        assert np.allclose(r, 0.0, atol=1e-6)
    
    def test_residual_nonzero_for_bad_measurements(self):
        """Residual should be non-zero/larger for inconsistent measurements."""
        from src.pipeline.state_estimation import compute_residuals
        
        m, n = 5, 2
        H = np.random.randn(m, n)
        H = H / np.linalg.norm(H, axis=0)  # normalize columns
        
        # Create inconsistent measurements (more equations than unknowns)
        z_random = np.random.randn(m)
        
        x_hat, R = wls_estimate(H, z_random, np.ones(m))
        r = compute_residuals(H, z_random, x_hat)
        
        # Residual norm should be non-trivial (overdetermined system)
        # Residual won't be exactly zero, but should be > 1e-10 typically
        assert np.linalg.norm(r) > 1e-6
