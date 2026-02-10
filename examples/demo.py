#!/usr/bin/env python
"""
Credal-TTA Demo Script
Quick demonstration of core functionality
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from credal_tta import CredalTTA
from credal_tta.models.wrappers import NaiveBaseline
from credal_tta.utils.data_loader import generate_sin_freq_shift
from credal_tta.utils.metrics import compute_all_metrics


def main():
    print("=" * 70)
    print("Credal-TTA Demo: Context Inertia Elimination")
    print("=" * 70)
    print()
    
    # Step 1: Generate synthetic data
    print("[1/5] Generating synthetic time series with regime shift...")
    data, shift_point = generate_sin_freq_shift(
        T=1000,
        shift_point=500,
        freq_before=10,
        freq_after=30,
        noise_std=0.1,
        seed=42
    )
    print(f"  ✓ Generated {len(data)} points, shift at t={shift_point}")
    print()
    
    # Step 2: Initialize model
    print("[2/5] Initializing forecasting model...")
    model = NaiveBaseline(method="moving_average")
    print(f"  ✓ Using moving average baseline")
    print()
    
    # Step 3: Standard approach
    print("[3/5] Running standard fixed-window forecasting...")
    standard_preds = []
    context = []
    W_max = 512
    
    for x_t in data:
        context.append(x_t)
        if len(context) > W_max:
            context = context[-W_max:]
        pred = model.predict(np.array(context)) if len(context) >= 10 else x_t
        standard_preds.append(pred)
    
    standard_preds = np.array(standard_preds)
    print(f"  ✓ Generated {len(standard_preds)} predictions")
    print()
    
    # Step 4: Credal-TTA approach
    print("[4/5] Running Credal-TTA adaptive forecasting...")
    adapter = CredalTTA(
        model=model,
        K=3,
        lambda_reset=1.15,  # Lower threshold for more sensitive detection
        lambda_caution=0.95,
        W_max=512,
        L_min=10,
        smoothing_alpha=0.1  # Less smoothing for faster response
    )
    
    credal_preds, diagnostics = adapter.predict_sequence(
        data,
        return_diagnostics=True
    )
    
    # Count resets
    num_resets = sum(1 for d in diagnostics if d.get('reset_occurred', False))
    reset_times = [d['t'] for d in diagnostics if d.get('reset_occurred', False)]
    
    print(f"  ✓ Generated {len(credal_preds)} predictions")
    print(f"  ✓ Detected {num_resets} regime shifts at times: {reset_times}")
    print()
    
    # Step 5: Compare performance
    print("[5/5] Comparing performance...")
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
    
    print()
    print("-" * 70)
    print("RESULTS SUMMARY")
    print("-" * 70)
    print()
    print(f"{'Metric':<20} {'Standard':<15} {'Credal-TTA':<15} {'Improvement':<15}")
    print("-" * 70)
    
    for metric in ['MAE', 'RMSE', 'Avg_RT', 'Avg_ATE']:
        standard_val = standard_metrics.get(metric, 0)
        credal_val = credal_metrics.get(metric, 0)
        improvement = 100 * (1 - credal_val / standard_val) if standard_val > 0 else 0
        
        print(f"{metric:<20} {standard_val:<15.4f} {credal_val:<15.4f} {improvement:<15.1f}%")
    
    print("-" * 70)
    print()
    
    # Diagnostic information
    print("DIAGNOSTIC INFORMATION")
    print("-" * 70)
    print(f"Number of resets detected: {num_resets}")
    if num_resets > 0:
        print(f"Reset times: {reset_times}")
    else:
        print("No resets detected. Possible reasons:")
        print("  - Detection threshold too high")
        print("  - Smoothing too aggressive")
        print("  - Regime shift not strong enough for this model")
        print("\nCredal set diameter evolution:")
        diameters = [d['diameter'] for d in diagnostics if 'diameter' in d]
        if len(diameters) > 0:
            print(f"  Initial diameter: {diameters[0]:.4f}")
            print(f"  Max diameter: {max(diameters):.4f}")
            print(f"  At shift point (t={shift_point}): {diameters[shift_point] if shift_point < len(diameters) else 'N/A'}")
            max_ratio = max([d.get('ratio', 0) for d in diagnostics])
            print(f"  Max contraction ratio: {max_ratio:.4f} (threshold: 1.15)")
    print("-" * 70)
    print()
    
    # Visualization
    print("Generating visualization...")
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    
    # Panel 1: Predictions
    t = np.arange(len(data))
    axes[0].plot(t, data, 'k-', alpha=0.4, linewidth=1, label='Ground Truth')
    axes[0].plot(t[1:], standard_preds[:-1], '-', color='orange', 
                alpha=0.6, linewidth=1.5, label='Standard')
    axes[0].plot(t[1:], credal_preds[:-1], '-', color='blue', 
                alpha=0.7, linewidth=1.5, label='Credal-TTA')
    
    axes[0].axvline(shift_point, color='red', linestyle='--', 
                   linewidth=2, alpha=0.5, label='True Shift')
    
    for rt in reset_times:
        if rt < len(t):
            axes[0].axvline(rt, color='green', linestyle=':', 
                          linewidth=1.5, alpha=0.5)
    
    axes[0].set_ylabel('Value', fontsize=11)
    axes[0].set_title('Prediction Comparison', fontsize=12, fontweight='bold')
    axes[0].legend(loc='upper right')
    axes[0].grid(True, alpha=0.3)
    
    # Panel 2: Credal set diameter
    diameters = [d['diameter'] for d in diagnostics]
    axes[1].plot(diameters, color='purple', linewidth=2, label='Diameter')
    axes[1].axvline(shift_point, color='red', linestyle='--', 
                   linewidth=2, alpha=0.5, label='True Shift')
    axes[1].axhline(0, color='gray', linestyle=':', alpha=0.5)
    
    axes[1].set_ylabel('Diameter', fontsize=11)
    axes[1].set_xlabel('Time Step', fontsize=11)
    axes[1].set_title('Epistemic Uncertainty (Credal Set Diameter)', 
                     fontsize=12, fontweight='bold')
    axes[1].legend(loc='upper right')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    output_path = Path("demo_output.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  ✓ Saved visualization to {output_path}")
    print()
    
    print("=" * 70)
    print("Demo complete!")
    print("=" * 70)
    print()
    print("Key Takeaways:")
    if num_resets > 0 and len(reset_times) > 0:
        print(f"  • Credal-TTA detected regime shift within {reset_times[0] - shift_point} steps")
        print(f"  • Recovery time reduced by {100*(1 - credal_metrics['Avg_RT']/standard_metrics['Avg_RT']):.1f}%")
        print(f"  • Prediction error (MAE) reduced by {100*(1 - credal_metrics['MAE']/standard_metrics['MAE']):.1f}%")
    else:
        print(f"  • Regime shift occurred at t={shift_point}")
        print(f"  • For this simple baseline, regime shift impact is minimal")
        print(f"  • Try with a more sophisticated model (Chronos) for better demonstration")
        print(f"  • Or adjust detection threshold: lambda_reset < 1.15")
    print()
    print("For more examples, see:")
    print("  - examples/quickstart.ipynb")
    print("  - experiments/synthetic.py (uses actual TSFM models)")
    print("  - docs/API.md")


if __name__ == "__main__":
    main()
