#!/usr/bin/env python
"""
Working Demo: Credal-TTA with Prediction Error Monitoring
This version monitors prediction errors instead of raw values for better detection
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from credal_tta.core.hca import HausdorffContextAdapter
from credal_tta.core.context_manager import ContextManager
from credal_tta.models.wrappers import NaiveBaseline
from credal_tta.utils.metrics import compute_all_metrics


def generate_strong_shift(T=1000, shift_point=500):
    """Generate obvious regime shift"""
    np.random.seed(42)
    
    # Regime 1: Low mean, low variance
    r1 = np.random.normal(0, 1, shift_point)
    
    # Regime 2: High mean, high variance
    r2 = np.random.normal(5, 2, T - shift_point)
    
    return np.concatenate([r1, r2]), shift_point


class ErrorBasedCredalTTA:
    """Credal-TTA that monitors prediction errors for better detection"""
    
    def __init__(self, model, K=3, lambda_reset=1.15, W_max=512, L_min=10):
        self.model = model
        self.hca = HausdorffContextAdapter(K=K, lambda_reset=lambda_reset, smoothing_alpha=0.1)
        self.context_manager = ContextManager(W_max=W_max, L_min=L_min)
        self.t = 0
        self.predictions = []
        self.diagnostics = []
        
    def predict_step(self, x_t):
        # Get context
        context, context_info = self.context_manager.update(
            x_new=x_t,
            regime_shift=False,  # Will update after prediction
            t=self.t
        )
        
        # Predict
        if len(context) >= 10:
            pred = self.model.predict(np.array(context))
        else:
            pred = x_t
        
        # Compute prediction error (this is what HCA monitors!)
        if self.t > 0 and len(self.predictions) > 0:
            prev_pred = self.predictions[-1]
            pred_error = abs(x_t - prev_pred)
            
            # Update HCA with prediction error
            hca_output = self.hca.update(pred_error)
            regime_shift = hca_output['regime_shift']
            
            # If shift detected, trigger context reset
            if regime_shift:
                self.context_manager.S = self.t
                context, context_info = self.context_manager.update(
                    x_new=x_t,
                    regime_shift=True,
                    t=self.t
                )
                # Re-predict with fresh context
                if len(context) >= 10:
                    pred = self.model.predict(np.array(context))
        else:
            hca_output = {
                'regime_shift': False,
                'diameter': 0.0,
                'ratio': 1.0,
                'smoothed_ratio': 1.0
            }
        
        self.predictions.append(pred)
        self.diagnostics.append({
            't': self.t,
            **hca_output,
            **context_info,
            'prediction': pred
        })
        
        self.t += 1
        return pred
    
    def predict_sequence(self, time_series):
        predictions = []
        for x_t in time_series:
            pred = self.predict_step(x_t)
            predictions.append(pred)
        return np.array(predictions), self.diagnostics


def main():
    print("=" * 70)
    print("Working Demo: Error-Based Credal-TTA")
    print("=" * 70)
    print()
    
    # Generate data
    print("[1/4] Generating time series...")
    data, shift_point = generate_strong_shift(T=1000, shift_point=500)
    print(f"  ✓ Generated {len(data)} points, shift at t={shift_point}")
    print(f"  ✓ Before shift: mean={np.mean(data[:shift_point]):.2f}")
    print(f"  ✓ After shift:  mean={np.mean(data[shift_point:]):.2f}")
    print()
    
    # Initialize
    print("[2/4] Initializing models...")
    model = NaiveBaseline(method="moving_average")
    print()
    
    # Standard approach
    print("[3/4] Running standard forecasting...")
    standard_preds = []
    context = []
    for x_t in data:
        context.append(x_t)
        if len(context) > 512:
            context = context[-512:]
        pred = model.predict(np.array(context)) if len(context) >= 10 else x_t
        standard_preds.append(pred)
    standard_preds = np.array(standard_preds)
    print(f"  ✓ Done")
    print()
    
    # Error-based Credal-TTA
    print("[4/4] Running Error-Based Credal-TTA...")
    adapter = ErrorBasedCredalTTA(
        model=model,
        K=3,
        lambda_reset=1.2,  # Can use higher threshold now
        W_max=512,
        L_min=10
    )
    
    credal_preds, diagnostics = adapter.predict_sequence(data)
    
    num_resets = sum(1 for d in diagnostics if d.get('reset_occurred', False))
    reset_times = [d['t'] for d in diagnostics if d.get('reset_occurred', False)]
    
    print(f"  ✓ Generated {len(credal_preds)} predictions")
    print(f"  ✓ Detected {num_resets} regime shifts at: {reset_times}")
    print()
    
    # Compare
    print("-" * 70)
    print("RESULTS")
    print("-" * 70)
    
    ground_truth = data[1:]
    
    standard_metrics = compute_all_metrics(
        ground_truth, standard_preds[:-1], shift_points=[shift_point]
    )
    credal_metrics = compute_all_metrics(
        ground_truth, credal_preds[:-1], shift_points=[shift_point]
    )
    
    for metric in ['MAE', 'RMSE', 'Avg_RT', 'Avg_ATE']:
        std_val = standard_metrics.get(metric, 0)
        cre_val = credal_metrics.get(metric, 0)
        imp = 100 * (1 - cre_val / std_val) if std_val > 0 else 0
        print(f"{metric:<15} Standard: {std_val:.4f}  Credal: {cre_val:.4f}  ({imp:+.1f}%)")
    
    print("-" * 70)
    print()
    
    if num_resets > 0:
        print("✓ SUCCESS: Regime shift detected!")
        print(f"  Detection lag: {reset_times[0] - shift_point} steps")
        print(f"  Recovery time reduced by {100*(1-credal_metrics['Avg_RT']/standard_metrics['Avg_RT']):.1f}%")
    else:
        print("⚠ No detection. This can happen if:")
        print("  - Moving average is too smooth")
        print("  - Try lowering lambda_reset to 1.05")
    
    # Visualize
    diameters = [d['diameter'] for d in diagnostics]
    ratios = [d['ratio'] for d in diagnostics]
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    
    # Predictions
    axes[0].plot(data, 'k-', alpha=0.4, label='True')
    axes[0].plot(standard_preds, 'o-', color='orange', markersize=1, alpha=0.6, label='Standard')
    axes[0].plot(credal_preds, 's-', color='blue', markersize=1, alpha=0.7, label='Credal-TTA')
    axes[0].axvline(shift_point, color='red', linestyle='--', label='Shift')
    for rt in reset_times:
        axes[0].axvline(rt, color='green', linestyle=':', alpha=0.7)
    axes[0].legend()
    axes[0].set_ylabel('Value')
    axes[0].set_title('Predictions')
    axes[0].grid(True, alpha=0.3)
    
    # Diameter
    axes[1].plot(diameters, 'purple', linewidth=2)
    axes[1].axvline(shift_point, color='red', linestyle='--')
    axes[1].set_ylabel('Diameter')
    axes[1].set_title('Credal Set Diameter (Prediction Error Based)')
    axes[1].grid(True, alpha=0.3)
    
    # Ratio
    axes[2].plot(ratios, 'orange', linewidth=2)
    axes[2].axhline(1.2, color='red', linestyle='--', label='Threshold')
    axes[2].axvline(shift_point, color='red', linestyle='--')
    axes[2].legend()
    axes[2].set_ylabel('Ratio')
    axes[2].set_xlabel('Time')
    axes[2].set_title('Contraction Ratio')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('demo_working.png', dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved to demo_working.png")


if __name__ == "__main__":
    main()
