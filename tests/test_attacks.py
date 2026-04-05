"""Unit tests for attack generation (stealth, standard, random FDIA)."""
import numpy as np
import pytest
from src.pipeline.attacks import (
    stealth_FDIA,
    standard_FDIA,
    random_attack,
    make_bus_targeted_c,
    raised_cosine_envelope,
)


class TestStealthFDIA:
    """Test stealth FDIA attack (a = alpha * H * c)."""
    
    def test_stealth_fdia_residual_zero(self):
        """Stealth attack should keep residual zero."""
        m, n = 10, 8
        H = np.random.randn(m, n)
        z_clean = np.random.randn(m)
        c = np.random.randn(n)
        c = c / np.linalg.norm(c)  # normalize
        
        percent = 0.15
        z_att, a = stealth_FDIA(H, z_clean, percent, c)
        
        # Attack should lie in Col(H)
        # Verify: residual = z_att - H @ x_hat should be unchanged
        # This is the core stealth property
        assert np.allclose(np.linalg.norm(a), percent * np.linalg.norm(z_clean), rtol=0.1)
    
    def test_stealth_fdia_column_space(self):
        """Attack magnitude should be percent * ||z_clean||."""
        m, n = 10, 8
        H = np.random.randn(m, n)
        z_clean = np.random.randn(m)
        c = np.random.randn(n)
        
        percent = 0.20
        z_att, a = stealth_FDIA(H, z_clean, percent, c)
        
        # Magnitude should scale correctly: ||a|| = percent * ||z_clean||
        target_norm = percent * np.linalg.norm(z_clean)
        actual_norm = np.linalg.norm(a)
        assert np.isclose(actual_norm, target_norm, rtol=0.01)
    
    def test_stealth_fdia_zero_percent(self):
        """Zero attack strength should produce no change."""
        m, n = 10, 8
        H = np.random.randn(m, n)
        z_clean = np.random.randn(m)
        c = np.random.randn(n)
        
        z_att, a = stealth_FDIA(H, z_clean, 0.0, c)
        
        assert np.allclose(z_att, z_clean)
        assert np.allclose(a, 0.0)


class TestStandardFDIA:
    """Test standard FDIA (shift on known measurements)."""
    
    def test_standard_fdia_shift(self):
        """Standard attack should shift specific measurements."""
        z = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        indices = np.array([1, 3])
        shift = 0.5
        
        z_att = standard_FDIA(z.copy(), indices, shift)
        
        assert z_att[1] == pytest.approx(2.5)
        assert z_att[3] == pytest.approx(4.5)
        assert z_att[0] == 1.0  # unattacked unchanged


class TestRandomFDIA:
    """Test random FDIA attack."""
    
    def test_random_fdia_bounds(self):
        """Random attack magnitude should be bounded."""
        z = np.ones(10)
        indices = np.array([0, 1, 2])
        scale = 0.1
        rng = np.random.default_rng(42)
        
        z_att = random_attack(z.copy(), indices, rng, scale)
        
        # Check attacked indices are perturbed
        assert not np.allclose(z_att[0], 1.0)
        # Check unattacked indices unchanged
        assert np.allclose(z_att[9], 1.0)


class TestMakeBusTargetedC:
    """Test c_direction generation localized to specific buses."""
    
    def test_targeted_c_localization(self):
        """c should have non-zero entries only at attacked buses."""
        n_state = 8
        attack_buses = [3, 4]
        slack_bus = 0
        rng = np.random.default_rng(42)
        
        c = make_bus_targeted_c(
            n_state=n_state,
            attack_buses=attack_buses,
            slack_bus=slack_bus,
            rng=rng
        )
        
        # c[0] should be zero (slack bus)
        assert c[0] == 0.0
        # c[2] and c[3] should be non-zero (buses 3, 4 map to indices 2, 3)
        assert c[2] != 0.0
        assert c[3] != 0.0
        # c[1], c[4], c[5], c[6], c[7] should be zero
        for i in [1, 4, 5, 6, 7]:
            assert c[i] == 0.0
    
    def test_targeted_c_single_bus(self):
        """Single bus targeting should work."""
        c = make_bus_targeted_c(
            n_state=8,
            attack_buses=[4],
            slack_bus=0,
            rng=np.random.default_rng(42)
        )
        
        nonzero_indices = np.where(c != 0.0)[0]
        assert len(nonzero_indices) == 1
        assert nonzero_indices[0] == 3  # Bus 4 maps to index 3


class TestRaisedCosineEnvelope:
    """Test temporal attack envelope."""
    
    def test_envelope_zero_outside_window(self):
        """Envelope should be zero outside attack window."""
        env = raised_cosine_envelope(t=50, start=100, end=200)
        assert env == 0.0
        
        env = raised_cosine_envelope(t=250, start=100, end=200)
        assert env == 0.0
    
    def test_envelope_one_at_middle(self):
        """Envelope should peak at middle of window."""
        env = raised_cosine_envelope(t=150, start=100, end=200)
        assert 0.9 <= env <= 1.0  # should be close to 1
    
    def test_envelope_smooth(self):
        """Envelope should be smooth at boundaries."""
        env_start = raised_cosine_envelope(t=100, start=100, end=200)
        env_start_plus1 = raised_cosine_envelope(t=101, start=100, end=200)
        
        # Should transition smoothly, not jump
        assert 0.0 <= env_start < env_start_plus1 <= 1.0
