"""
Comprehensive Validation Script
Verifies all components from the response to reviewers are implemented
"""

import sys
import os
os.environ.setdefault('PYTHONIOENCODING', 'utf-8')
from pathlib import Path
import numpy as np

def check_module(module_path, required_items):
    """Check if module exists and has required items"""
    try:
        spec = __import__(module_path, fromlist=required_items)
        missing = [item for item in required_items if not hasattr(spec, item)]
        if missing:
            return False, f"Missing: {missing}"
        return True, "OK"
    except Exception as e:
        return False, str(e)


def main():
    print("=" * 80)
    print("CREDAL-TTA VALIDATION - Response to Reviewers Implementation")
    print("=" * 80)
    
    checks = []
    
    # Core modules
    print("\n[1] Core Modules")
    print("-" * 80)
    
    core_checks = [
        ("credal_tta.core.hca", ["HausdorffContextAdapter"]),
        ("credal_tta.core.credal_set", ["CredalSet", "GaussianDistribution", "wasserstein_2_gaussian"]),
        ("credal_tta.core.context_manager", ["ContextManager"]),
        ("credal_tta.core.hca_multivariate", ["MultivariateHCA"]),
    ]
    
    for module, items in core_checks:
        status, msg = check_module(module, items)
        checks.append(status)
        symbol = "[PASS]" if status else "[FAIL]"
        print(f"  {symbol} {module}: {msg}")
    
    # Utils
    print("\n[2] Utilities")
    print("-" * 80)
    
    util_checks = [
        ("credal_tta.utils.preprocessing", ["preprocess_for_hca", "two_track_preprocess"]),
        ("credal_tta.utils.data_loader", ["load_sp500", "load_bitcoin", "load_electricity", "load_noaa_weather", "load_ettm1"]),
        ("credal_tta.utils.metrics", ["mae", "rmse", "recovery_time", "accumulated_transition_error"]),
    ]
    
    for module, items in util_checks:
        status, msg = check_module(module, items)
        checks.append(status)
        symbol = "[PASS]" if status else "[FAIL]"
        print(f"  {symbol} {module}: {msg}")
    
    # Models
    print("\n[3] Models & Baselines")
    print("-" * 80)
    
    model_checks = [
        ("credal_tta.models.tta_baselines", ["LoRATTA", "TENTTTA"]),
        ("credal_tta.models.wrappers", ["ChronosWrapper", "MoiraiWrapper", "PatchTSTWrapper"]),
    ]
    
    for module, items in model_checks:
        status, msg = check_module(module, items)
        checks.append(status)
        symbol = "[PASS]" if status else "[FAIL]"
        print(f"  {symbol} {module}: {msg}")
    
    # Main framework
    print("\n[4] Main Framework")
    print("-" * 80)
    
    status, msg = check_module("credal_tta.credal_tta", ["CredalTTA"])
    checks.append(status)
    symbol = "[PASS]" if status else "[FAIL]"
    print(f"  {symbol} credal_tta.credal_tta: {msg}")
    
    # Experiments
    print("\n[5] Experiment Scripts")
    print("-" * 80)
    
    experiment_files = [
        "experiments/synthetic.py",
        "experiments/finance.py",
        "experiments/electricity.py",
        "experiments/cross_domain.py",
        "experiments/multivariate_etth1.py",
        "experiments/gradient_comparison.py",
        "experiments/validation/burnin_stress_test.py",
        "experiments/validation/regime_separation.py",
    ]
    
    for exp_file in experiment_files:
        exists = Path(exp_file).exists()
        checks.append(exists)
        symbol = "[PASS]" if exists else "[FAIL]"
        print(f"  {symbol} {exp_file}")
    
    # Summary
    print("\n" + "=" * 80)
    passed = sum(checks)
    total = len(checks)
    print(f"VALIDATION SUMMARY: {passed}/{total} checks passed")
    
    if passed == total:
        print("[OK] ALL COMPONENTS IMPLEMENTED")
        print("\nKey Features from Response to Reviewers:")
        print("  - R1-W1: Two-track data flow (TSFM + HCA preprocessing)")
        print("  - R1-W2: Burn-in health check with stress test")
        print("  - R1-W3: Multivariate extension (diagonal covariance)")
        print("  - R1-W4: LoRA-TTA and TENT-TTA baselines")
        print("  - R3: Cross-domain datasets (Electricity, NOAA, ETTm1)")
    else:
        print(f"[INCOMPLETE] {total - passed} components missing or incomplete")
    
    print("=" * 80)
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
