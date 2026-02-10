"""
Unit Tests for Credal-TTA Components
"""

import unittest
import numpy as np
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from credal_tta.core.credal_set import (
    GaussianDistribution, 
    wasserstein_2_gaussian, 
    CredalSet,
    initialize_credal_set
)
from credal_tta.core.hca import HausdorffContextAdapter
from credal_tta.core.context_manager import ContextManager
from credal_tta.credal_tta import CredalTTA
from credal_tta.models.wrappers import NaiveBaseline


class TestCredalSet(unittest.TestCase):
    """Test credal set operations"""
    
    def test_gaussian_bayesian_update(self):
        """Test Gaussian conjugate update"""
        p = GaussianDistribution(mu=0.0, sigma=1.0)
        p_updated = p.bayesian_update(x_obs=2.0, sigma_noise=1.0)
        
        # After observing 2.0, mean should shift toward 2.0
        self.assertGreater(p_updated.mu, p.mu)
        self.assertLess(p_updated.sigma, p.sigma)  # Variance decreases
    
    def test_wasserstein_distance(self):
        """Test Wasserstein-2 distance computation"""
        p = GaussianDistribution(0, 1)
        q = GaussianDistribution(3, 1)
        
        dist = wasserstein_2_gaussian(p, q)
        
        # For equal variance, W2 ≈ |μ_p - μ_q|
        self.assertAlmostEqual(dist, 3.0, places=1)
    
    def test_credal_set_diameter(self):
        """Test diameter computation"""
        extremes = [
            GaussianDistribution(0, 1),
            GaussianDistribution(5, 1)
        ]
        credal_set = CredalSet(extremes)
        
        diam = credal_set.diameter()
        
        # Diameter should be positive
        self.assertGreater(diam, 0)
    
    def test_credal_set_contraction(self):
        """Test geometric contraction in stable regime"""
        burn_in = np.random.normal(0, 1, 100)
        credal_set = initialize_credal_set(burn_in, K=3)
        
        # Simulate stable observations
        diameters = [credal_set.diameter()]
        for _ in range(50):
            x = np.random.normal(0, 1)
            credal_set = credal_set.update(x, sigma_noise=0.5)
            diameters.append(credal_set.diameter())
        
        # Diameter should decrease on average
        self.assertLess(np.mean(diameters[-10:]), np.mean(diameters[:10]))


class TestHCA(unittest.TestCase):
    """Test Hausdorff Context Adapter"""
    
    def test_initialization(self):
        """Test HCA initialization"""
        hca = HausdorffContextAdapter(K=3, lambda_reset=1.2)
        
        burn_in = np.random.normal(0, 1, 50)
        hca.initialize(burn_in)
        
        self.assertTrue(hca.is_initialized)
        self.assertEqual(hca.credal_set.K, 3)
    
    def test_regime_detection(self):
        """Test regime shift detection"""
        hca = HausdorffContextAdapter(K=3, lambda_reset=1.2)
        
        # Stable regime
        stable_data = np.random.normal(0, 1, 100)
        hca.initialize(stable_data[:50])
        
        for x in stable_data[50:]:
            output = hca.update(x)
            # Should not detect shift in stable data
        
        # Check that most ratios are < 1.2
        stable_ratios = hca.ratio_history[-40:]
        self.assertLess(np.mean(stable_ratios), 1.1)
        
        # Regime shift
        shift_data = np.random.normal(5, 2, 20)
        for x in shift_data:
            output = hca.update(x)
        
        # Should detect expansion
        shift_ratios = hca.ratio_history[-10:]
        self.assertGreater(max(shift_ratios), 1.2)


class TestContextManager(unittest.TestCase):
    """Test Context Manager"""
    
    def test_reset_and_grow(self):
        """Test reset-and-grow mechanism"""
        manager = ContextManager(W_max=512, L_min=10)
        
        # Add data
        for t in range(100):
            context, info = manager.update(
                x_new=float(t),
                regime_shift=(t == 50),
                t=t
            )
        
        # Origin should be at reset point
        self.assertEqual(manager.S, 50)
        
        # Context length should be growing from reset
        self.assertLessEqual(len(context), 50)


class TestCredalTTA(unittest.TestCase):
    """Test full Credal-TTA framework"""
    
    def test_predict_sequence(self):
        """Test sequence prediction"""
        model = NaiveBaseline()
        adapter = CredalTTA(model=model, K=3, lambda_reset=1.2)
        
        # Generate data with shift
        data = np.concatenate([
            np.random.normal(0, 1, 100),
            np.random.normal(5, 1, 100)
        ])
        
        predictions = adapter.predict_sequence(data)
        
        # Should produce predictions
        self.assertEqual(len(predictions), len(data))
        
        # Should detect shift
        summary = adapter.get_uncertainty_summary()
        self.assertGreater(summary['num_resets'], 0)


def run_tests():
    """Run all tests"""
    unittest.main(argv=[''], exit=False, verbosity=2)


if __name__ == "__main__":
    run_tests()
