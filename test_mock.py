#!/usr/bin/env python
"""
Quick test to verify mock model improvements work
"""

import numpy as np
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from credal_tta.credal_tta import CredalTTA
from credal_tta.models.wrappers import ChronosWrapper
from credal_tta.utils.data_loader import generate_sin_freq_shift
from credal_tta.utils.metrics import compute_all_metrics


def test_mock_model():
    print("=" * 70)
    print("Testing Mock Model with Credal-TTA")
    print("=" * 70)
    print()
    
    # Generate data
    data, shift_point = generate_sin_freq_shift(T=1000, shift_point=500, seed=42)
    print(f"Generated {len(data)} points, shift at t={shift_point}")
    print()
    
    # Initialize mock model (chronos not installed → uses moving average)
    model = ChronosWrapper()
    print("Initialized model (should show WARNING about mock)")
    print()
    
    # Test 1: Standard approach
    print("[Test 1] Standard fixed-window approach...")
    standard_preds = []
    context = []
    
    for x_t in data:
        context.append(x_t)
        if len(context) > 512:
            context = context[-512:]
        pred = model.predict(np.array(context))
        standard_preds.append(pred)
    
    standard_preds = np.array(standard_preds)
    print(f"  Generated {len(standard_preds)} predictions")
    
    # Check if predictions actually change with context
    print(f"  Predictions at t=400: {standard_preds[400]:.4f}")
    print(f"  Predictions at t=500: {standard_preds[500]:.4f}")
    print(f"  Predictions at t=600: {standard_preds[600]:.4f}")
    
    if abs(standard_preds[400] - standard_preds[600]) < 0.1:
        print("  ⚠ WARNING: Predictions barely change! Mock model may be broken.")
    else:
        print(f"  ✓ Predictions adapt (change: {abs(standard_preds[400] - standard_preds[600]):.4f})")
    print()
    
    # Test 2: Credal-TTA
    print("[Test 2] Credal-TTA adaptive approach...")
    adapter = CredalTTA(
        model=model,
        K=3,
        lambda_reset=1.15,  # Lower threshold
        W_max=512,
        L_min=10,
        smoothing_alpha=0.1
    )
    
    credal_preds, diagnostics = adapter.predict_sequence(data, return_diagnostics=True)
    
    num_resets = sum(1 for d in diagnostics if d.get('reset_occurred', False))
    reset_times = [d['t'] for d in diagnostics if d.get('reset_occurred', False)]
    
    print(f"  Generated {len(credal_preds)} predictions")
    print(f"  Number of resets: {num_resets}")
    if num_resets > 0:
        print(f"  Reset times: {reset_times[:5]}...")  # Show first 5
    print()
    
    # Test 3: Compare metrics
    print("[Test 3] Comparing performance...")
    ground_truth = data[1:]
    
    standard_metrics = compute_all_metrics(
        ground_truth,
        standard_preds[:-1],
        shift_points=[shift_point]
    )
    
    credal_metrics = compute_all_metrics(
        ground_truth,
        credal_preds[:-1],
        shift_points=[shift_point]
    )
    
    print(f"  Standard MAE: {standard_metrics['MAE']:.4f}")
    print(f"  Credal MAE:   {credal_metrics['MAE']:.4f}")
    print(f"  Standard RT:  {standard_metrics.get('Avg_RT', 0):.1f}")
    print(f"  Credal RT:    {credal_metrics.get('Avg_RT', 0):.1f}")
    print()
    
    # Diagnosis
    print("=" * 70)
    print("DIAGNOSIS")
    print("=" * 70)
    
    if abs(standard_metrics['MAE'] - credal_metrics['MAE']) < 0.001:
        print("❌ PROBLEM: No difference between Standard and Credal-TTA")
        print("   This means Credal-TTA is not working properly.")
        print()
        print("   Possible reasons:")
        print("   1. HCA not detecting shifts (check diagnostics)")
        print("   2. Mock model not context-dependent")
        print("   3. Need to monitor prediction errors instead of raw values")
        print()
        
        # Additional diagnostics
        diameters = [d['diameter'] for d in diagnostics if 'diameter' in d]
        ratios = [d['ratio'] for d in diagnostics if 'ratio' in d]
        
        if len(diameters) > 0:
            print(f"   Max diameter: {max(diameters):.6f}")
            print(f"   Max ratio: {max(ratios):.4f}")
            print(f"   Diameter at shift: {diameters[shift_point] if shift_point < len(diameters) else 'N/A'}")
            
            if max(diameters) < 0.01:
                print("   → Diameter too small! HCA not working properly.")
            if max(ratios) < 1.15:
                print("   → Ratio never exceeds threshold. No shifts detected.")
    
    elif credal_metrics['MAE'] < standard_metrics['MAE']:
        print("✓ SUCCESS: Credal-TTA shows improvement!")
        improvement = 100 * (1 - credal_metrics['MAE'] / standard_metrics['MAE'])
        print(f"   MAE improvement: {improvement:.1f}%")
        
        if num_resets > 0:
            print(f"   Detected {num_resets} shifts")
            print(f"   First detection at t={reset_times[0]} (true shift at t={shift_point})")
        else:
            print("   ⚠ Improvement without explicit detections")
            print("     (This can happen due to gradual adaptation)")
    
    else:
        print("⚠ UNEXPECTED: Credal-TTA worse than Standard")
        print("   This suggests a bug in the implementation")
    
    print()
    return standard_metrics, credal_metrics, num_resets


if __name__ == "__main__":
    test_mock_model()
