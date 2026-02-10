#!/usr/bin/env python
"""
Enhanced Credal-TTA Demo Script
Demonstrates context inertia elimination with strong regime shift
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from credal_tta import CredalTTA
from credal_tta.models.wrappers import NaiveBaseline
from credal_tta.utils.metrics import compute_all_metrics


def generate_strong_regime_shift(T=1000, shift_point=500, seed=42):
    """Generate data with very obvious regime shift"""
    np.random.seed(seed)
    
    # Before shift: stable sinusoidal
    t1 = np.arange(shift_point)
    regime1 = 2 * np.sin(2 * np.pi * t1 / 50) + np.random.normal(0, 0.2, shift_point)
    
    # After shift: different frequency + higher mean
    t2 = np.arange(T - shift_point)
    regime2 = 5 + 3 * np.sin(2 * np.pi * t2 / 20) + np.random.normal(0, 0.3, T - shift_point)
    
    data = np.concatenate([regime1, regime2])
    return data, shift_point


def main():
    print("=" * 70)
    print("Enhanced Credal-TTA Demo: Strong Regime Shift Detection")
    print("=" * 70)
    print()
    
    # Step 1: Generate data with obvious shift
    print("[1/5] Generating time series with strong regime shift...")
    data, shift_point = generate_strong_regime_shift(T=1000, shift_point=500)
    print(f"  ✓ Generated {len(data)} points, shift at t={shift_point}")
    print(f"  ✓ Before shift: mean={np.mean(data[:shift_point]):.2f}, std={np.std(data[:shift_point]):.2f}")
    print(f"  ✓ After shift:  mean={np.mean(data[shift_point:]):.2f}, std={np.std(data[shift_point:]):.2f}")
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
    
    # Step 4: Credal-TTA
    print("[4/5] Running Credal-TTA adaptive forecasting...")
    adapter = CredalTTA(
        model=model,
        K=3,
        lambda_reset=1.15,      # Sensitive detection
        lambda_caution=0.95,
        W_max=512,
        L_min=10,
        smoothing_alpha=0.1     # Fast response
    )
    
    credal_preds, diagnostics = adapter.predict_sequence(data, return_diagnostics=True)
    
    # Extract diagnostics
    num_resets = sum(1 for d in diagnostics if d.get('reset_occurred', False))
    reset_times = [d['t'] for d in diagnostics if d.get('reset_occurred', False)]
    diameters = [d['diameter'] for d in diagnostics]
    ratios = [d['ratio'] for d in diagnostics]
    
    print(f"  ✓ Generated {len(credal_preds)} predictions")
    print(f"  ✓ Detected {num_resets} regime shifts at times: {reset_times}")
    print()
    
    # Step 5: Compare
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
        
        print(f"{metric:<20} {standard_val:<15.4f} {credal_val:<15.4f} {improvement:>14.1f}%")
    
    print("-" * 70)
    print()
    
    # Diagnostic information
    print("CREDAL-TTA DIAGNOSTICS")
    print("-" * 70)
    print(f"Number of resets: {num_resets}")
    if num_resets > 0:
        print(f"First reset at: t={reset_times[0]} (true shift at t={shift_point}, lag={reset_times[0]-shift_point})")
    print(f"Max diameter: {max(diameters):.4f}")
    print(f"Max contraction ratio: {max(ratios):.4f} (threshold: 1.15)")
    print(f"Diameter at shift point: {diameters[shift_point]:.4f}")
    print(f"Ratio at shift point: {ratios[shift_point]:.4f}")
    print("-" * 70)
    print()
    
    # Visualization
    print("Generating visualization...")
    
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    # Panel 1: Ground truth
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(data, 'k-', alpha=0.5, linewidth=1, label='Ground Truth')
    ax1.axvline(shift_point, color='red', linestyle='--', linewidth=2, label='True Shift')
    ax1.set_ylabel('Value', fontsize=11)
    ax1.set_title('Time Series with Strong Regime Shift', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # Panel 2: Predictions comparison
    ax2 = fig.add_subplot(gs[1, :])
    window = slice(shift_point - 50, shift_point + 150)
    ax2.plot(range(shift_point - 50, shift_point + 150), data[window], 
            'k-', alpha=0.4, linewidth=2, label='Ground Truth')
    ax2.plot(range(shift_point - 50, shift_point + 150), standard_preds[window], 
            'o-', color='orange', alpha=0.6, markersize=2, linewidth=1.5, label='Standard (Fixed Window)')
    ax2.plot(range(shift_point - 50, shift_point + 150), credal_preds[window], 
            's-', color='blue', alpha=0.7, markersize=2, linewidth=1.5, label='Credal-TTA (Adaptive)')
    ax2.axvline(shift_point, color='red', linestyle='--', linewidth=2, alpha=0.5)
    
    for rt in reset_times:
        if shift_point - 50 < rt < shift_point + 150:
            ax2.axvline(rt, color='green', linestyle=':', linewidth=2, alpha=0.7, label='Reset' if rt == reset_times[0] else '')
    
    ax2.set_ylabel('Value', fontsize=11)
    ax2.set_xlabel('Time Step', fontsize=11)
    ax2.set_title('Prediction Comparison (Zoomed Around Shift)', fontsize=12, fontweight='bold')
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    # Panel 3: Credal set diameter
    ax3 = fig.add_subplot(gs[2, 0])
    ax3.plot(diameters, color='purple', linewidth=2, label='Diameter')
    ax3.axvline(shift_point, color='red', linestyle='--', linewidth=2, alpha=0.5)
    for rt in reset_times:
        ax3.axvline(rt, color='green', linestyle=':', linewidth=1.5, alpha=0.7)
    ax3.set_ylabel('Diameter', fontsize=11)
    ax3.set_xlabel('Time Step', fontsize=11)
    ax3.set_title('Epistemic Uncertainty (Credal Set Diameter)', fontsize=11, fontweight='bold')
    ax3.legend(loc='upper right')
    ax3.grid(True, alpha=0.3)
    
    # Panel 4: Contraction ratio
    ax4 = fig.add_subplot(gs[2, 1])
    ax4.plot(ratios, color='orange', linewidth=2, label='Contraction Ratio')
    ax4.axhline(1.15, color='red', linestyle='--', linewidth=2, alpha=0.5, label='Threshold λ=1.15')
    ax4.axvline(shift_point, color='red', linestyle='--', linewidth=2, alpha=0.5)
    ax4.set_ylabel('Ratio ρ_t', fontsize=11)
    ax4.set_xlabel('Time Step', fontsize=11)
    ax4.set_title('Detection Signal (Contraction Ratio)', fontsize=11, fontweight='bold')
    ax4.legend(loc='upper right')
    ax4.grid(True, alpha=0.3)
    
    plt.savefig('demo_output_enhanced.png', dpi=150, bbox_inches='tight')
    print(f"  ✓ Saved visualization to demo_output_enhanced.png")
    print()
    
    print("=" * 70)
    print("Demo complete!")
    print("=" * 70)
    print()
    print("Key Takeaways:")
    if num_resets > 0 and len(reset_times) > 0:
        detection_lag = reset_times[0] - shift_point
        print(f"  ✓ Strong regime shift successfully detected!")
        print(f"  • Detection lag: {detection_lag} steps after true shift")
        print(f"  • Recovery time improvement: {100*(1 - credal_metrics['Avg_RT']/standard_metrics['Avg_RT']):.1f}%")
        print(f"  • Error reduction (MAE): {100*(1 - credal_metrics['MAE']/standard_metrics['MAE']):.1f}%")
        print(f"  • ATE reduction: {100*(1 - credal_metrics['Avg_ATE']/standard_metrics['Avg_ATE']):.1f}%")
    else:
        print(f"  ⚠ No regime shift detected (try lowering lambda_reset)")
    print()
    print("For better results with real TSFMs:")
    print("  - python experiments/synthetic.py --model chronos")
    print("  - jupyter lab examples/quickstart.ipynb")


if __name__ == "__main__":
    main()
