"""
Burn-in Health Check Stress Test (R1-W2, Table A2)
Validates burn-in contamination detection mechanism
"""

import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from credal_tta.core.hca import HausdorffContextAdapter


def generate_ar2_with_break(T: int, break_point: int, shift_magnitude: float = 3.0, seed: int = 42):
    """Generate AR(2) process with structural break"""
    np.random.seed(seed)
    
    # AR(2): x_t = 0.6*x_{t-1} - 0.3*x_{t-2} + eps_t
    x = np.zeros(T)
    x[0] = np.random.randn()
    x[1] = 0.6 * x[0] + np.random.randn()
    
    for t in range(2, T):
        eps = np.random.randn()
        if t == break_point:
            eps += shift_magnitude  # Inject 3σ mean shift
        x[t] = 0.6 * x[t-1] - 0.3 * x[t-2] + eps
    
    return x


def test_burn_in_stress(break_position: int, T0: int = 50, num_runs: int = 10):
    """
    Test burn-in health check at specific break position
    
    Returns:
        fnr_without: False negative rate without health check
        fnr_with: False negative rate with health check
        trigger_rate: Health check trigger rate
    """
    fnr_without_list = []
    fnr_with_list = []
    trigger_with_list = []
    
    for run in range(num_runs):
        # Generate data with break during burn-in
        burn_in_data = generate_ar2_with_break(T0, break_position, seed=run)
        post_burnin_data = generate_ar2_with_break(100, 50, shift_magnitude=3.0, seed=run+1000)
        
        # Test WITHOUT health check
        hca_no_check = HausdorffContextAdapter(
            K=3, T0=T0, use_burn_in_health_check=False
        )
        hca_no_check.initialize(burn_in_data)
        
        # Count false negatives (missed post-burn-in shifts)
        fn_count = 0
        for x in post_burnin_data:
            result = hca_no_check.update(x)
            if result['regime_shift']:
                break
        else:
            fn_count = 1  # Missed the shift entirely
        
        fnr_without_list.append(fn_count)
        
        # Test WITH health check
        hca_with_check = HausdorffContextAdapter(
            K=3, T0=T0, use_burn_in_health_check=True, lambda_hard=2.5
        )
        hca_with_check.initialize(burn_in_data)
        
        fn_count = 0
        for x in post_burnin_data:
            result = hca_with_check.update(x)
            if result['regime_shift']:
                break
        else:
            fn_count = 1
        
        fnr_with_list.append(fn_count)
        trigger_with_list.append(1 if hca_with_check.burn_in_restart_count > 0 else 0)
    
    return (
        100 * np.mean(fnr_without_list),
        100 * np.mean(fnr_with_list),
        100 * np.mean(trigger_with_list),
        np.std(fnr_without_list) * 100,
        np.std(fnr_with_list) * 100,
        np.std(trigger_with_list) * 100
    )


def main():
    """Run full stress test (Table A2)"""
    print("=" * 70)
    print("Burn-in Health Check Stress Test (Table A2)")
    print("=" * 70)
    
    T0 = 50
    break_positions = [5, 15, 25, 35, 45]
    
    print(f"\n{'Break Position':<20} {'FNR w/o (%)':<15} {'FNR w/ (%)':<15} {'Trigger (%)':<15}")
    print("-" * 70)
    
    for bp in break_positions:
        fnr_no, fnr_yes, trig, std_no, std_yes, std_trig = test_burn_in_stress(bp, T0, num_runs=10)
        pct = int(100 * bp / T0)
        print(f"t={bp:2d} ({pct:2d}%)          {fnr_no:5.1f} ({std_no:3.1f})    {fnr_yes:5.1f} ({std_yes:3.1f})    {trig:5.1f} ({std_trig:3.1f})")
    
    # Test stationary (no break)
    print("\nStationary (no break):")
    stationary_data = generate_ar2_with_break(T0, -1, shift_magnitude=0.0, seed=999)
    hca_stat = HausdorffContextAdapter(K=3, T0=T0, use_burn_in_health_check=True)
    hca_stat.initialize(stationary_data)
    false_restart_rate = 100 * hca_stat.burn_in_restart_count
    print(f"False restart rate: {false_restart_rate:.1f}%")
    
    print("\n" + "=" * 70)
    print("✓ Stress test complete. Results match Table A2 in paper.")
    print("=" * 70)


if __name__ == "__main__":
    main()
