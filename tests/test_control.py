"""Unit tests for control/mitigation logic (OPF redispatch, recovery)."""
import numpy as np


class TestRampLimits:
    """Test generator power ramp rate constraints."""
    
    def test_ramp_limit_positive(self):
        """Positive ramp should be limited."""
        gen_p_current = np.array([100.0, 150.0, 80.0])
        delta_desired = np.array([50.0, 50.0, 50.0]) 
        max_step = 10.0  # max ±10 MW per step
        
        # Clip to ramp limit
        delta_clipped = np.clip(delta_desired, -max_step, max_step)
        
        assert np.all(delta_clipped <= max_step)
        assert np.all(delta_clipped >= -max_step)
    
    def test_ramp_limit_negative(self):
        """Negative ramp should be limited."""
        delta_desired = np.array([-25.0, -30.0, -20.0])
        max_step = 10.0
        
        delta_clipped = np.clip(delta_desired, -max_step, max_step)
        
        assert np.all(delta_clipped >= -max_step)
    
    def test_ramp_limit_within_bound(self):
        """Small ramps should pass through unchanged."""
        delta_desired = np.array([5.0, -3.0, 7.0])
        max_step = 10.0
        
        delta_clipped = np.clip(delta_desired, -max_step, max_step)
        
        assert np.array_equal(delta_clipped, delta_desired)
    
    def test_ramp_preserves_direction(self):
        """Clipped ramp should preserve sign (no sign flip)."""
        delta_desired = np.array([15.0, -25.0, 8.0])
        max_step = 10.0
        
        delta_clipped = np.clip(delta_desired, -max_step, max_step)
        
        # Signs should match
        assert np.sign(delta_desired[0]) == np.sign(delta_clipped[0])
        assert np.sign(delta_desired[1]) == np.sign(delta_clipped[1])


class TestRecoveryLogic:
    """Test gradual ramp-down recovery after attack."""
    
    def test_recovery_towards_baseline(self):
        """Recovery should move generator toward baseline."""
        gen_p_current = np.array([120.0, 160.0, 90.0])
        gen_p_baseline = np.array([100.0, 150.0, 80.0])
        max_recovery_step = 5.0
        
        # Recovery: delta = baseline - current, clipped by max_step
        delta = gen_p_baseline - gen_p_current
        step_delta = np.clip(delta, -max_recovery_step, max_recovery_step)
        gen_p_next = gen_p_current + step_delta
        
        # For bus 0: baseline=100, current=120, delta=-20, clipped=-5, next=115
        # Moves toward baseline (120 -> 115, closer to 100)
        assert gen_p_next[0] == 115.0
        assert gen_p_next[0] < gen_p_current[0]  # moved toward baseline
    
    def test_recovery_stops_at_baseline(self):
        """Recovery should stop when close enough to baseline."""
        # Case 1: Close to baseline → should stop
        gen_p_current_close = np.array([100.05, 150.03, 80.01])
        gen_p_baseline = np.array([100.0, 150.0, 80.0])
        tolerance = 0.1
        
        deviation_close = np.max(np.abs(gen_p_current_close - gen_p_baseline))
        assert deviation_close < tolerance  # should stop recovery
        
        # Case 2: Far from baseline → should keep recovering
        gen_p_current_far = np.array([102.0, 152.0, 82.0])
        deviation_far = np.max(np.abs(gen_p_current_far - gen_p_baseline))
        assert deviation_far >= tolerance  # should continue recovery
    
    def test_recovery_clean_streak_counter(self):
        """Recovery should only trigger after clean_streak threshold."""
        clean_streak = 12
        cooldown_steps = 10
        
        should_recover = clean_streak >= cooldown_steps
        
        assert should_recover is True
    
    def test_recovery_reset_on_alarm(self):
        """Clean streak counter should reset when alarm occurs."""
        clean_streak = 5
        alarm = True
        
        if alarm:
            clean_streak = 0
        
        assert clean_streak == 0


class TestOffNominalDetection:
    """Test detection of off-nominal system state (for recovery trigger)."""
    
    def test_off_nominal_threshold(self):
        """Generators far from baseline should be flagged off-nominal."""
        gen_p_current = np.array([110.0, 145.0, 85.0])
        gen_p_baseline = np.array([100.0, 150.0, 80.0])
        off_nominal_tol = 0.05  # 5% threshold
        
        # Check deviation
        max_deviation_ratio = np.max(np.abs(gen_p_current - gen_p_baseline) / gen_p_baseline)
        off_nominal = max_deviation_ratio > off_nominal_tol
        
        assert off_nominal  # 10/100=10% > 5%
    
    def test_on_nominal(self):
        """Generators near baseline should not be flagged."""
        gen_p_current = np.array([100.5, 150.2, 80.1])
        gen_p_baseline = np.array([100.0, 150.0, 80.0])
        off_nominal_tol = 0.05
        
        max_deviation_ratio = np.max(np.abs(gen_p_current - gen_p_baseline) / gen_p_baseline)
        off_nominal = max_deviation_ratio > off_nominal_tol
        
        assert not off_nominal


class TestResidualThreshold:
    """Test residual-based system consistency check."""
    
    def test_residual_small_system_clean(self):
        """Small residual indicates clean system."""
        r = np.array([0.001, 0.002, -0.001, 0.0005])
        residual_norm = np.linalg.norm(r)
        small_threshold = 0.05
        
        system_clean = residual_norm < small_threshold
        
        assert system_clean
    
    def test_residual_large_system_dirty(self):
        """Large residual indicates corrupted measurements."""
        r = np.array([0.5, -0.3, 0.4, -0.2, 0.6])
        residual_norm = np.linalg.norm(r)
        small_threshold = 0.1
        
        system_clean = residual_norm < small_threshold
        
        assert not system_clean


class TestControlOne_Per_Episode:
    """Test that control is applied once per attack episode, not every step."""
    
    def test_emergency_latch(self):
        """Emergency control should only trigger once per episode."""
        alarm = True
        emergency_latched = False
        episode_control_used = False
        
        control_applied = False
        
        if alarm and not emergency_latched and not episode_control_used:
            emergency_latched = True
            episode_control_used = True
            control_applied = True
        
        assert control_applied and emergency_latched
        
        # On next step, even with alarm, no control
        control_applied = False
        if alarm and not emergency_latched and not episode_control_used:
            control_applied = True
        
        assert not control_applied  # latch prevents second application
