#!/usr/bin/env python
"""
Quick Validation Script
Verifies that core components work correctly
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import numpy as np


def test_imports():
    """Test that all modules can be imported"""
    print("Testing imports...", end=" ")
    try:
        from credal_tta import CredalTTA
        from credal_tta.core.hca import HausdorffContextAdapter
        from credal_tta.core.context_manager import ContextManager
        from credal_tta.core.credal_set import CredalSet, GaussianDistribution
        from credal_tta.models.wrappers import NaiveBaseline
        from credal_tta.utils.metrics import mae, recovery_time
        from credal_tta.utils.data_loader import generate_sin_freq_shift
        print("✓ PASS")
        return True
    except Exception as e:
        print(f"✗ FAIL: {e}")
        return False


def test_credal_set():
    """Test credal set operations"""
    print("Testing credal set...", end=" ")
    try:
        from credal_tta.core.credal_set import GaussianDistribution, wasserstein_2_gaussian, initialize_credal_set
        
        # Test Gaussian
        p = GaussianDistribution(0, 1)
        q = GaussianDistribution(3, 1)
        
        # Test Wasserstein distance
        dist = wasserstein_2_gaussian(p, q)
        assert dist > 0, "Distance should be positive"
        
        # Test initialization
        burn_in = np.random.normal(0, 1, 100)
        credal_set = initialize_credal_set(burn_in, K=3)
        assert credal_set.K == 3, "Should have 3 extremes"
        
        # Test update
        credal_set = credal_set.update(1.0, 0.1)
        assert credal_set.diameter() >= 0, "Diameter should be non-negative"
        
        print("✓ PASS")
        return True
    except Exception as e:
        print(f"✗ FAIL: {e}")
        return False


def test_hca():
    """Test Hausdorff Context Adapter"""
    print("Testing HCA...", end=" ")
    try:
        from credal_tta.core.hca import HausdorffContextAdapter
        
        hca = HausdorffContextAdapter(K=3, lambda_reset=1.2)
        
        # Initialize
        burn_in = np.random.normal(0, 1, 50)
        hca.initialize(burn_in)
        assert hca.is_initialized, "Should be initialized"
        
        # Update
        for _ in range(10):
            output = hca.update(np.random.normal(0, 1))
            assert 'diameter' in output, "Should return diameter"
            assert 'ratio' in output, "Should return ratio"
        
        print("✓ PASS")
        return True
    except Exception as e:
        print(f"✗ FAIL: {e}")
        return False


def test_context_manager():
    """Test Context Manager"""
    print("Testing context manager...", end=" ")
    try:
        from credal_tta.core.context_manager import ContextManager
        
        manager = ContextManager(W_max=512, L_min=10)
        
        # Add data without reset
        for t in range(50):
            context, info = manager.update(float(t), regime_shift=False, t=t)
            assert len(context) <= 512, "Context should not exceed W_max"
        
        # Trigger reset
        context, info = manager.update(100.0, regime_shift=True, t=50)
        assert info['reset_occurred'], "Should record reset"
        assert manager.S == 50, "Origin should be updated"
        
        print("✓ PASS")
        return True
    except Exception as e:
        print(f"✗ FAIL: {e}")
        return False


def test_credal_tta():
    """Test full Credal-TTA framework"""
    print("Testing Credal-TTA framework...", end=" ")
    try:
        from credal_tta import CredalTTA
        from credal_tta.models.wrappers import NaiveBaseline
        from credal_tta.utils.data_loader import generate_sin_freq_shift
        
        # Generate data
        data, _ = generate_sin_freq_shift(T=200, shift_point=100)
        
        # Initialize
        model = NaiveBaseline()
        adapter = CredalTTA(model=model, K=3, lambda_reset=1.2)
        
        # Predict
        predictions = adapter.predict_sequence(data)
        assert len(predictions) == len(data), "Should produce prediction for each point"
        
        # Check diagnostics
        summary = adapter.get_uncertainty_summary()
        assert 'mean_diameter' in summary, "Should have uncertainty metrics"
        
        print("✓ PASS")
        return True
    except Exception as e:
        print(f"✗ FAIL: {e}")
        return False


def test_metrics():
    """Test evaluation metrics"""
    print("Testing metrics...", end=" ")
    try:
        from credal_tta.utils.metrics import mae, rmse, recovery_time, compute_all_metrics
        
        # Generate dummy data
        y_true = np.random.randn(100)
        y_pred = y_true + np.random.randn(100) * 0.1
        
        # Test basic metrics
        error = mae(y_true, y_pred)
        assert error >= 0, "MAE should be non-negative"
        
        error = rmse(y_true, y_pred)
        assert error >= 0, "RMSE should be non-negative"
        
        # Test recovery time
        rt = recovery_time(y_true, y_pred, shift_point=50)
        assert rt >= 0, "Recovery time should be non-negative"
        
        # Test all metrics
        metrics = compute_all_metrics(y_true, y_pred, shift_points=[50])
        assert 'MAE' in metrics, "Should have MAE"
        assert 'RMSE' in metrics, "Should have RMSE"
        
        print("✓ PASS")
        return True
    except Exception as e:
        print(f"✗ FAIL: {e}")
        return False


def main():
    print("=" * 60)
    print("Credal-TTA Validation Suite")
    print("=" * 60)
    print()
    
    tests = [
        test_imports,
        test_credal_set,
        test_hca,
        test_context_manager,
        test_credal_tta,
        test_metrics,
    ]
    
    results = [test() for test in tests]
    
    print()
    print("=" * 60)
    print(f"Results: {sum(results)}/{len(results)} tests passed")
    print("=" * 60)
    
    if all(results):
        print("\n✓ All tests passed! The installation is working correctly.")
        return 0
    else:
        print("\n✗ Some tests failed. Please check the error messages above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
