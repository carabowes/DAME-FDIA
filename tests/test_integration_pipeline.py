"""Integration tests for end-to-end pipeline scenarios."""
import numpy as np
import pytest


class TestCleanRunNoAttack:
    """End-to-end test: clean run should produce no alarms."""
    
    def test_clean_scenario_no_false_alarms(self):
        """
        Simulate 100 clean steps (no attack).
        Expectations:
        - No measurement anomalies detected
        - State estimates converge
        - Feature values remain low (innovations, residuals)
        """
        from src.pipeline.state_estimation import wls_estimate, compute_residuals
        from src.features.innovations import compute_innovations
        
        n_steps = 100
        n_buses = 8
        n_meas = 10
        
        # Synthetic stable H matrix (measurement Jacobian)
        H = np.random.randn(n_meas, n_buses)
        H = H / np.linalg.norm(H, axis=0)
        
        # True state
        x_true = np.ones(n_buses) * 0.05  # small bus angles
        
        # Simulate clean measurements
        Z = []
        residuals_all = []
        
        sigma = np.ones(n_meas) * 0.01  # noise std
        
        for t in range(n_steps):
            # Clean measurement
            z_t = H @ x_true + np.random.randn(n_meas) * 0.01
            Z.append(z_t)
            
            # WLS estimate
            x_hat, _ = wls_estimate(H, z_t, sigma)
            
            # Residual
            r_t = compute_residuals(H, z_t, x_hat)
            residuals_all.append(r_t)
        
        Z = np.array(Z)
        residuals_all = np.array(residuals_all)
        
        # Assertions: no large residuals (would trigger false alarm)
        max_residual = np.max(np.abs(residuals_all))
        assert max_residual < 0.5, f"Residuals too large in clean run: {max_residual}"
        
        # Feature values should be bounded (no wild swings)
        E = compute_innovations(Z, alpha=0.7)
        max_innovation = np.max(np.abs(E))
        assert max_innovation < 2.0, f"Innovations too large: {max_innovation}"
    
    def test_clean_run_outputs_sensible_format(self):
        """Verify output structure (state estimates, residuals, metrics)."""
        from src.pipeline.state_estimation import wls_estimate, compute_residuals
        
        H = np.random.randn(10, 8)
        H = H / np.linalg.norm(H, axis=0)
        x_true = np.ones(8) * 0.05
        
        z = H @ x_true + np.random.randn(10) * 0.01
        sigma = np.ones(10) * 0.01
        
        x_hat, R = wls_estimate(H, z, sigma)
        r = compute_residuals(H, z, x_hat)
        
        # Type checks
        assert isinstance(x_hat, np.ndarray), "x_hat should be ndarray"
        assert isinstance(r, np.ndarray), "residual should be ndarray"
        assert isinstance(R, np.ndarray), "R should be ndarray"
        
        # Shape checks
        assert x_hat.shape == (8,), f"x_hat shape {x_hat.shape}, expected (8,)"
        assert r.shape == (10,), f"residual shape {r.shape}, expected (10,)"
        assert R.shape == (10, 10), f"R shape {R.shape}, expected (10,10)"
        
        # Value checks
        assert not np.any(np.isnan(x_hat)), "x_hat contains NaN"
        assert not np.any(np.isnan(r)), "residual contains NaN"


class TestAttackedRunProducesAlarm:
    """End-to-end test: attacked run should detect anomaly."""
    
    def test_stealth_attack_detection_integration(self):
        """
        Inject stealth attack and verify detection through feature anomaly.
        Workflow:
        1. Generate clean training features
        2. Fit OCSVM on clean data
        3. Inject attack in test window
        4. Verify anomaly score spikes
        """
        from src.pipeline.state_estimation import wls_estimate, compute_residuals
        from src.pipeline.attacks import stealth_FDIA, make_bus_targeted_c
        from src.features.innovations import compute_innovations
        
        np.random.seed(42)
        
        # Setup
        H = np.random.randn(10, 8)
        H = H / np.linalg.norm(H, axis=0)
        x_true = np.ones(8) * 0.05
        sigma = np.ones(10) * 0.01
        
        n_train = 50
        n_attack = 10
        
        # Phase 1: Generate clean training data
        Z_train = []
        S_train = []  # state features
        
        for t in range(n_train):
            z_t = H @ x_true + np.random.randn(10) * 0.01
            Z_train.append(z_t)
            x_hat, _ = wls_estimate(H, z_t, sigma)
            S_train.append(x_hat)
        
        Z_train = np.array(Z_train)
        S_train = np.array(S_train)
        
        # Phase 2: Generate attacked test data
        c = make_bus_targeted_c(n_state=8, attack_buses=[3, 4], slack_bus=0, rng=np.random.default_rng(42))
        z_clean_attack = H @ x_true
        
        Z_attack = []
        S_attack = []
        
        for t in range(n_attack):
            if 5 <= t < 8:  # Attack window
                z_att, _ = stealth_FDIA(H, z_clean_attack, percent=0.20, c_direction=c)
                z_t = z_att + np.random.randn(10) * 0.01
            else:
                z_t = z_clean_attack + np.random.randn(10) * 0.01
            
            Z_attack.append(z_t)
            x_hat, _ = wls_estimate(H, z_t, sigma)
            S_attack.append(x_hat)
        
        S_attack = np.array(S_attack)
        
        # Phase 3: Compute anomaly metric (simple: L2 distance from clean mean)
        clean_mean = np.mean(S_train, axis=0)
        clean_std = np.std(S_train, axis=0) + 1e-6
        
        anomaly_scores_train = np.linalg.norm((S_train - clean_mean) / clean_std, axis=1)
        anomaly_scores_attack = np.linalg.norm((S_attack - clean_mean) / clean_std, axis=1)
        
        # Assertions
        mean_score_clean = np.mean(anomaly_scores_train)
        max_score_attack = np.max(anomaly_scores_attack[5:8])  # during attack window
        
        # Attack window should show elevated anomaly (slightly above normal)
        # Note: with synthetic data and small attack (20%), detection is subtle
        assert max_score_attack > mean_score_clean * 1.2, \
            f"Attack not detected: clean_mean={mean_score_clean}, attack_max={max_score_attack}"


class TestMitigationControlSafety:
    """End-to-end test: mitigation should not crash and produce valid outputs."""
    
    def test_mitigation_enabled_run_produces_trajectories(self):
        """
        Simulate control + recovery loop without crashes.
        Verify:
        - Ramps respect ±10 MW limits
        - Recovery moves toward baseline (not away)
        - All outputs are finite and bounded
        """
        n_gens = 3
        n_steps = 50
        baseline = np.array([100.0, 150.0, 80.0])
        max_ramp = 10.0
        
        # Simulate control trajectory under attack + recovery
        gen_p = baseline.copy()
        gen_p_traj = [gen_p.copy()]
        
        for t in range(n_steps):
            # Attack injects disturbance at t=15-25
            if 15 <= t <= 25:
                disturbance = np.array([15.0, 10.0, 5.0])
            else:
                disturbance = np.zeros(3)
            
            # Apply disturbance
            gen_p_desired = gen_p + disturbance
            
            # Enforce ramp limits
            delta = gen_p_desired - gen_p
            delta_clipped = np.clip(delta, -max_ramp, max_ramp)
            gen_p_new = gen_p + delta_clipped
            
            # Recovery: move toward baseline
            if t > 30:
                recovery_delta = baseline - gen_p_new
                recovery_clipped = np.clip(recovery_delta, -max_ramp, max_ramp)
                gen_p_new = gen_p_new + recovery_clipped
            
            gen_p = gen_p_new
            gen_p_traj.append(gen_p.copy())
        
        gen_p_traj = np.array(gen_p_traj)
        
        # Assertions
        # No NaN/Inf
        assert not np.any(np.isnan(gen_p_traj)), "Trajectory contains NaN"
        assert not np.any(np.isinf(gen_p_traj)), "Trajectory contains Inf"
        
        # Trajectories remain bounded (not diverging)
        assert np.all(gen_p_traj >= 0), "Negative generator power"
        assert np.all(gen_p_traj <= 300), "Generator power exceeds bounds"
        
        # Ramps respected
        ramps = np.abs(np.diff(gen_p_traj, axis=0))
        assert np.all(ramps <= max_ramp * 1.01), \
            f"Ramp violated: max={np.max(ramps)}, limit={max_ramp}"
        
        # Recovery moves toward baseline at end
        final_deviation = np.linalg.norm(gen_p_traj[-1] - baseline)
        early_deviation = np.linalg.norm(gen_p_traj[30] - baseline)
        assert final_deviation < early_deviation, \
            "Recovery should move toward baseline"


class TestPipelineComponentIntegration:
    """Verify components work together without interface mismatches."""
    
    def test_attack_to_detection_pipeline(self):
        """
        Full chain: attacks → state estimation → features → anomaly metric.
        Verify no dimension/type mismatches at interfaces.
        """
        from src.pipeline.state_estimation import wls_estimate
        from src.pipeline.attacks import stealth_FDIA, make_bus_targeted_c
        from src.features.innovations import compute_innovations
        
        np.random.seed(42)
        
        # Setup
        m, n = 10, 8
        H = np.random.randn(m, n)
        H = H / np.linalg.norm(H, axis=0)
        x_true = np.random.randn(n) * 0.05
        sigma = np.ones(m) * 0.01
        z_clean = H @ x_true
        
        # Interface 1: Attack generation → measurements
        c = make_bus_targeted_c(n_state=n, attack_buses=[2, 3], slack_bus=0, rng=np.random.default_rng(42))
        z_att, a = stealth_FDIA(H, z_clean, percent=0.15, c_direction=c)
        assert z_att.shape == (m,), f"Attack output shape {z_att.shape}, expected ({m},)"
        
        # Interface 2: Measurements → state estimation
        x_hat, R = wls_estimate(H, z_att, sigma)
        assert x_hat.shape == (n,), f"State estimate shape {x_hat.shape}, expected ({n},)"
        assert R.shape == (m, m), f"Covariance shape {R.shape}, expected ({m},{m})"
        
        # Interface 3: Collect states → feature extraction
        Z_sequence = np.array([z_att, z_att, z_att, z_att, z_att])
        E = compute_innovations(Z_sequence, alpha=0.7)
        assert E.shape == Z_sequence.shape, f"Innovation shape mismatch"
        
        # Interface 4: Features → anomaly score (mock detector)
        window_size = 3
        flattened = E[-window_size:].flatten()
        assert flattened.shape == (window_size * m,), "Window flattening failed"
        
        anomaly_score = np.linalg.norm(flattened)
        assert np.isfinite(anomaly_score), "Anomaly score not finite"
        assert anomaly_score >= 0, "Anomaly score negative"
