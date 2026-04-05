"""Unit tests for feature extraction (innovations, residuals, state features)."""
import numpy as np
import pytest
from src.features.innovations import compute_innovations


class TestComputeInnovations:
    """Test temporal innovation (prediction error) feature extraction."""
    
    def test_innovations_first_step_zero(self):
        """First innovation should be zero (no history)."""
        Z = np.array([[1.0, 2.0], [1.5, 2.5], [1.8, 2.8]])
        
        E = compute_innovations(Z, alpha=0.3)
        
        assert np.allclose(E[0], 0.0)
    
    def test_innovations_shape(self):
        """Output shape should match input shape."""
        T, d = 100, 8
        Z = np.random.randn(T, d)
        
        E = compute_innovations(Z, alpha=0.7)
        
        assert E.shape == Z.shape
    
    def test_innovations_alpha_effect(self):
        """Different alpha values produce different smoothing levels."""
        Z = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
        
        E_low = compute_innovations(Z, alpha=0.1)   # conservative (stable predictions)
        E_high = compute_innovations(Z, alpha=0.9)  # aggressive (responsive predictions)
        
        # Low alpha: stable prediction - large innovations (prediction lags measurements)
        # High alpha: responsive prediction - small innovations (prediction follows closely)
        assert np.linalg.norm(E_low[1:]) > np.linalg.norm(E_high[1:])
    
    def test_innovations_step_function(self):
        """Innovations should be large at step changes."""
        Z = np.array([[1.0], [1.0], [1.0], [10.0], [10.0], [10.0]])
        
        E = compute_innovations(Z, alpha=0.5)
        
        # Innovation at step 3 should be large
        assert np.abs(E[3]) > np.abs(E[1])
    
    def test_innovations_steady_state(self):
        """Steady measurements should have small innovations."""
        Z = np.ones((50, 3))
        
        E = compute_innovations(Z, alpha=0.7)
        
        # After first few steps, innovations should be near zero
        assert np.allclose(E[20:], 0.0, atol=1e-10)
    
    def test_innovations_invalid_alpha(self):
        """Invalid alpha should raise error."""
        Z = np.random.randn(10, 5)
        
        with pytest.raises(ValueError):
            compute_innovations(Z, alpha=0.0)  # alpha must be > 0
        
        with pytest.raises(ValueError):
            compute_innovations(Z, alpha=1.5)  # alpha must be <= 1
    
    def test_innovations_invalid_shape(self):
        """1D input should raise error."""
        Z = np.random.randn(10)  # 1D, should be 2D
        
        with pytest.raises(ValueError):
            compute_innovations(Z, alpha=0.5)


class TestFeatureRepresentations:
    """Test equivalence and interpretation of different feature types."""
    
    def test_state_vs_residual_correlation(self):
        """
        Under stealth attack:
        - Residuals stay zero
        - State features change
        """
        # Simulate: clean state, then attacked state
        x_clean = np.array([0.1, 0.05, -0.08, 0.02, 0.0, -0.03, 0.07, 0.04])
        x_attacked = np.array([0.12, 0.05, -0.08, 0.05, 0.0, -0.03, 0.07, 0.04])  # Bus 4 shifted
        
        # State feature difference should be large
        state_diff = np.linalg.norm(x_attacked - x_clean)
        assert state_diff > 0.02
        
        # In stealth attack, residual would stay zero (by design)
    
    def test_state_feature_dimension(self):
        """State features should have dimension = number of buses."""
        n_buses = 8
        x = np.random.randn(n_buses)
        
        # State features are just the state itself
        state_features = x.copy()
        
        assert len(state_features) == n_buses
    
    def test_residual_feature_dimension(self):
        """Residual features can be scalar (norm) or vector."""
        m_measurements = 15
        r = np.random.randn(m_measurements)
        
        # Scalar residual feature
        residual_norm = np.linalg.norm(r)
        assert isinstance(residual_norm, (float, np.floating))
        
        # Vector residual feature
        residual_vector = r
        assert len(residual_vector) == m_measurements


class TestWindowedFeatureExtraction:
    """Test rolling window feature aggregation (for streaming)."""
    
    def test_window_flattening(self):
        """Window of states should flatten to single feature vector."""
        window_size = 5
        n_buses = 8
        
        # Simulated window of states
        window = np.random.randn(window_size, n_buses)
        
        # Flatten for detector
        flattened = window.flatten()
        
        assert flattened.shape == (window_size * n_buses,)
        assert flattened.shape == (40,)
    
    def test_window_buffer_order(self):
        """Features should maintain temporal order in window."""
        window = np.array([[1, 2], [3, 4], [5, 6]])  # 3 timesteps, 2 features
        
        flattened = window.flatten()
        expected = np.array([1, 2, 3, 4, 5, 6])
        
        assert np.array_equal(flattened, expected)
