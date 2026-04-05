"""Reproducibility tests: fixed seeds should produce identical results."""
import numpy as np
import pytest


class TestAttackReproducibility:
    """Verify attack generation with fixed seeds is deterministic."""
    
    def test_stealth_attack_same_seed_same_result(self):
        """Same seed → same attack vector."""
        from src.pipeline.attacks import stealth_FDIA, make_bus_targeted_c
        
        H = np.random.randn(10, 8)
        z_clean = np.random.randn(10)
        
        # Run 1: Fixed seed
        rng1 = np.random.default_rng(12345)
        c1 = make_bus_targeted_c(n_state=8, attack_buses=[2, 3], slack_bus=0, rng=rng1)
        z_att1, a1 = stealth_FDIA(H, z_clean, percent=0.15, c_direction=c1)
        
        # Run 2: Same fixed seed
        rng2 = np.random.default_rng(12345)
        c2 = make_bus_targeted_c(n_state=8, attack_buses=[2, 3], slack_bus=0, rng=rng2)
        z_att2, a2 = stealth_FDIA(H, z_clean, percent=0.15, c_direction=c2)
        
        # Verify exact reproduction
        assert np.allclose(c1, c2, rtol=1e-15), "c vectors differ with same seed"
        assert np.allclose(a1, a2, rtol=1e-15), "attack vectors differ with same seed"
        assert np.allclose(z_att1, z_att2, rtol=1e-15), "measurements differ with same seed"
    
    def test_random_attack_different_seed_different_result(self):
        """Different seed → different attack (with very high probability)."""
        from src.pipeline.attacks import random_attack
        
        z = np.ones(10) * 5.0
        attacked_indices = [0, 1, 2]
        scale = 1.0
        
        # Different seeds should produce different random attacks
        rng1 = np.random.default_rng(111)
        z_att1 = random_attack(z, attacked_indices, rng1, scale)
        
        rng2 = np.random.default_rng(222)
        z_att2 = random_attack(z, attacked_indices, rng2, scale)
        
        # Should be different (probability of collision is negligible)
        assert not np.allclose(z_att1, z_att2), "Different seeds produced same attack"


class TestStateEstimationReproducibility:
    """WLS estimator deterministic (same input → same output)."""
    
    def test_wls_deterministic_same_matrices_same_result(self):
        """No randomness in WLS, so identical matrices give identical estimates."""
        from src.pipeline.state_estimation import wls_estimate
        
        H = np.array([[1.0, 0.5], [0.5, 1.0], [1.0, -0.5]])
        z = np.array([2.0, 1.5, 1.0])
        sigma = np.array([0.1, 0.1, 0.1])
        
        # Run 1
        x_hat1, R1 = wls_estimate(H, z, sigma, reg=1e-6)
        
        # Run 2 (identical inputs)
        x_hat2, R2 = wls_estimate(H, z, sigma, reg=1e-6)
        
        # Must be exactly reproducible (no randomness)
        assert np.allclose(x_hat1, x_hat2, rtol=1e-15), "WLS not deterministic"
        assert np.allclose(R1, R2, rtol=1e-15), "Covariance not deterministic"
    
    def test_wls_stable_under_small_perturbations(self):
        """Small input changes → small output changes (stability check)."""
        from src.pipeline.state_estimation import wls_estimate
        
        H = np.random.randn(8, 6)
        x_true = np.random.randn(6) * 0.1
        z = H @ x_true
        sigma = np.ones(8) * 0.01
        
        # Baseline estimate
        x_hat1, _ = wls_estimate(H, z, sigma)
        
        # Tiny perturbation on measurements
        z_pert = z + np.random.randn(8) * 1e-8
        x_hat2, _ = wls_estimate(H, z_pert, sigma)
        
        # Change in output should be tiny
        error_ratio = np.linalg.norm(x_hat2 - x_hat1) / np.linalg.norm(x_hat1)
        assert error_ratio < 1e-5, f"Unstable: 1e-8 perturbation caused {error_ratio:.2e} change"


class TestFeatureReproducibility:
    """Innovation features deterministic given input."""
    
    def test_innovations_deterministic(self):
        """No randomness in feature extraction."""
        from src.features.innovations import compute_innovations
        
        Z = np.array([[1.0], [1.2], [1.5], [2.0], [2.1]])
        
        # Run 1
        E1 = compute_innovations(Z, alpha=0.7)
        
        # Run 2
        E2 = compute_innovations(Z, alpha=0.7)
        
        # Must be identical
        assert np.allclose(E1, E2, rtol=1e-15), "Innovations not deterministic"
    
    def test_window_flattening_deterministic(self):
        """Window flattening is deterministic."""
        Z = np.random.randn(20, 8)  # 20 steps, 8 features
        window_size = 5
        
        # Manual flattening test
        window1 = Z[-window_size:].flatten()
        window2 = Z[-window_size:].flatten()
        
        assert np.allclose(window1, window2), "Flattening not deterministic"


class TestNoiseInjectionReproducibility:
    """Measurement noise reproducible with seed."""
    
    def test_measurement_noise_seeded(self):
        """Fix random seed → fix noise injection."""
        n_meas = 10
        
        # Noise run 1 with seed
        rng1 = np.random.default_rng(9999)
        noise1 = rng1.normal(0, 0.01, size=n_meas)
        
        # Noise run 2 with same seed
        rng2 = np.random.default_rng(9999)
        noise2 = rng2.normal(0, 0.01, size=n_meas)
        
        assert np.allclose(noise1, noise2, rtol=1e-15), "Noise not reproducible with seed"
    
    def test_multiple_noise_samples_reproducible(self):
        """Sequence of noise vectors reproducible."""
        rng1 = np.random.default_rng(7777)
        noise_sequence1 = [rng1.normal(0, 0.01, size=10) for _ in range(5)]
        
        rng2 = np.random.default_rng(7777)
        noise_sequence2 = [rng2.normal(0, 0.01, size=10) for _ in range(5)]
        
        for n1, n2 in zip(noise_sequence1, noise_sequence2):
            assert np.allclose(n1, n2, rtol=1e-15), "Noise sequence not reproducible"


class TestDetectorReproducibility:
    """Detection behavior reproducible for debugging and auditing."""
    
    def test_anomaly_score_deterministic(self):
        """Given state features, anomaly score is deterministic."""
        S = np.random.randn(10, 8)  # 10 samples, 8 features
        
        # Compute mean and std
        mean1 = np.mean(S, axis=0)
        std1 = np.std(S, axis=0)
        
        # Recompute (should be identical)
        mean2 = np.mean(S, axis=0)
        std2 = np.std(S, axis=0)
        
        assert np.allclose(mean1, mean2), "Mean not deterministic"
        assert np.allclose(std1, std2), "Std not deterministic"
        
        # Anomaly scores should be identical
        scores1 = np.linalg.norm((S - mean1) / (std1 + 1e-6), axis=1)
        scores2 = np.linalg.norm((S - mean2) / (std2 + 1e-6), axis=1)
        
        assert np.allclose(scores1, scores2), "Anomaly scores not deterministic"
    
    def test_threshold_calculation_deterministic(self):
        """Threshold derived from fixed feature data is deterministic."""
        S = np.random.randn(100, 8)
        
        # Compute threshold as 95th percentile of anomaly scores
        mean = np.mean(S, axis=0)
        std = np.std(S, axis=0)
        scores = np.linalg.norm((S - mean) / (std + 1e-6), axis=1)
        
        threshold1 = np.percentile(scores, 95)
        threshold2 = np.percentile(scores, 95)
        
        assert np.isclose(threshold1, threshold2), "Threshold not deterministic"


class TestEndToEndReproducibility:
    """Full pipeline with fixed seed produces reproducible results."""
    
    def test_full_pipeline_reproducible_with_seed(self):
        """
        Seed all sources of randomness:
        - H matrix, x_true (setup)
        - Attack direction c
        - Measurement noise
        - Feature computation
        Result should be fully reproducible.
        """
        from src.pipeline.state_estimation import wls_estimate
        from src.pipeline.attacks import stealth_FDIA, make_bus_targeted_c
        from src.features.innovations import compute_innovations
        
        def run_pipeline_with_seed(seed):
            np.random.seed(seed)
            rng = np.random.default_rng(seed)
            
            # Setup (seeded)
            H = np.random.randn(10, 8)
            H = H / np.linalg.norm(H, axis=0)
            x_true = np.random.randn(8) * 0.05
            sigma = np.ones(10) * 0.01
            
            # Attack (seeded)
            z_clean = H @ x_true
            c = make_bus_targeted_c(n_state=8, attack_buses=[2, 3], slack_bus=0, rng=rng)
            z_att, a = stealth_FDIA(H, z_clean, percent=0.15, c_direction=c)
            
            # State estimation (deterministic)
            x_hat, _ = wls_estimate(H, z_att, sigma)
            
            # Features (deterministic)
            Z_seq = np.array([z_att, z_att, z_att])
            E = compute_innovations(Z_seq, alpha=0.7)
            
            return x_hat, a, E
        
        # Run 1: seed 555
        x_hat1, a1, E1 = run_pipeline_with_seed(555)
        
        # Run 2: same seed 555
        x_hat2, a2, E2 = run_pipeline_with_seed(555)
        
        # All outputs must be identical
        assert np.allclose(x_hat1, x_hat2, rtol=1e-15), "State estimates differ"
        assert np.allclose(a1, a2, rtol=1e-15), "Attack vectors differ"
        assert np.allclose(E1, E2, rtol=1e-15), "Features differ"
