"""
UCI Electricity Experiments - Reproducing Table 3
"""

import numpy as np
import argparse
import json
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from credal_tta.credal_tta import CredalTTA
from credal_tta.models.wrappers import ChronosWrapper, MoiraiWrapper, NaiveBaseline
from credal_tta.utils.data_loader import load_uci_electricity
from credal_tta.utils.metrics import compute_all_metrics, recovery_time
from credal_tta.core.hca import HausdorffContextAdapter


def adwin_baseline(time_series, model, delta=0.002):
    """ADWIN drift detection baseline (simplified implementation)"""
    predictions = []
    context = []
    window_size = 100
    
    # Track running statistics
    running_mean = []
    running_std = []
    
    for t, x_t in enumerate(time_series):
        context.append(x_t)
        
        # Compute statistics
        if len(context) >= window_size:
            recent_mean = np.mean(context[-window_size//2:])
            old_mean = np.mean(context[-window_size:-window_size//2])
            
            # Simple drift test: mean difference
            if abs(recent_mean - old_mean) > 2 * np.std(context[-window_size:]):
                # Reset context
                context = context[-10:]
        
        # Predict
        if len(context) >= 10:
            pred = model.predict(np.array(context))
        else:
            pred = x_t
        predictions.append(pred)
    
    return np.array(predictions)


def kswin_baseline(time_series, model, window_size=100, p_threshold=0.01):
    """KSWIN baseline (simplified KS test)"""
    from scipy.stats import ks_2samp
    
    predictions = []
    context = []
    
    for t, x_t in enumerate(time_series):
        context.append(x_t)
        
        # KS test between recent windows
        if len(context) >= window_size:
            w1 = context[-window_size:-window_size//2]
            w2 = context[-window_size//2:]
            
            _, p_value = ks_2samp(w1, w2)
            
            if p_value < p_threshold:
                # Detected drift
                context = context[-10:]
        
        # Predict
        if len(context) >= 10:
            pred = model.predict(np.array(context))
        else:
            pred = x_t
        predictions.append(pred)
    
    return np.array(predictions)


def run_electricity_experiment(
    model_name="chronos",
    num_series=20,
    save_dir="results/electricity"
):
    """
    Run UCI Electricity experiments
    
    Args:
        model_name: chronos, moirai, or naive
        num_series: Number of customer series to test
        save_dir: Output directory
    """
    print(f"\n{'='*70}")
    print(f"UCI Electricity Experiment: {model_name.upper()}")
    print(f"{'='*70}\n")
    
    results = {
        'Standard': {'MAE': [], 'RMSE': [], 'MAPE': [], 'RT': []},
        'Variance-Trigger': {'MAE': [], 'RMSE': [], 'MAPE': [], 'RT': []},
        'ADWIN': {'MAE': [], 'RMSE': [], 'MAPE': [], 'RT': []},
        'KSWIN': {'MAE': [], 'RMSE': [], 'MAPE': [], 'RT': []},
        'Credal-TTA': {'MAE': [], 'RMSE': [], 'MAPE': [], 'RT': []},
    }
    
    for series_id in range(num_series):
        print(f"\nProcessing customer {series_id+1}/{num_series}...")
        
        # Load data
        data = load_uci_electricity(customer_id=series_id)
        
        # Use subset for faster experiments
        data = data[:5000]  # ~200 days
        
        # Initialize model
        if model_name == "chronos":
            base_model = ChronosWrapper()
        elif model_name == "moirai":
            base_model = MoiraiWrapper()
        else:
            base_model = NaiveBaseline(method="moving_average")
        
        # Ground truth (one-step-ahead)
        ground_truth = data[1:]
        
        # --- Standard Fixed Window ---
        print("  Running Standard...")
        standard_preds = []
        context = []
        W_max = 512
        
        for x_t in data:
            context.append(x_t)
            if len(context) > W_max:
                context = context[-W_max:]
            pred = base_model.predict(np.array(context)) if len(context) >= 10 else x_t
            standard_preds.append(pred)
        
        standard_preds = np.array(standard_preds)[:-1]
        
        # --- Variance-Trigger ---
        print("  Running Variance-Trigger...")
        variance_preds = []
        context = []
        baseline_var = np.var(data[:200])
        
        for x_t in data:
            context.append(x_t)
            if len(context) >= 50:
                if np.var(context[-50:]) > 2.5 * baseline_var:
                    context = context[-10:]
            pred = base_model.predict(np.array(context)) if len(context) >= 10 else x_t
            variance_preds.append(pred)
        
        variance_preds = np.array(variance_preds)[:-1]
        
        # --- ADWIN ---
        print("  Running ADWIN...")
        adwin_preds = adwin_baseline(data, base_model)[:-1]
        
        # --- KSWIN ---
        print("  Running KSWIN...")
        kswin_preds = kswin_baseline(data, base_model)[:-1]
        
        # --- Credal-TTA ---
        print("  Running Credal-TTA...")
        adapter = CredalTTA(
            model=base_model,
            K=3,
            lambda_reset=3,
            W_max=512,
            L_min=192,
            smoothing_alpha=0.1,
            sigma_noise=0.5
        )
        
        credal_preds, diagnostics = adapter.predict_sequence(data, return_diagnostics=True)

        # 添加这几行诊断
        num_resets = sum(1 for d in diagnostics if d.get('reset_occurred', False))
        reset_times = [d['t'] for d in diagnostics if d.get('reset_occurred', False)]
        max_diameter = max([d.get('diameter', 0) for d in diagnostics])
        max_ratio = max([d.get('ratio', 0) for d in diagnostics])
        print(f"  DEBUG: Resets={num_resets}, Times={reset_times[:10]}, MaxDiam={max_diameter:.4f}, MaxRatio={max_ratio:.4f}")

        credal_preds = credal_preds[:-1]
        
        # Detect shift points from diagnostics
        shift_points = [d['t'] for d in diagnostics if d.get('reset_occurred', False)]
        
        # Compute metrics for each method
        for method_name, preds in [
            ('Standard', standard_preds),
            ('Variance-Trigger', variance_preds),
            ('ADWIN', adwin_preds),
            ('KSWIN', kswin_preds),
            ('Credal-TTA', credal_preds)
        ]:
            metrics = compute_all_metrics(ground_truth, preds, shift_points=shift_points[:3])
            
            results[method_name]['MAE'].append(metrics['MAE'])
            results[method_name]['RMSE'].append(metrics['RMSE'])
            results[method_name]['MAPE'].append(metrics['MAPE'])
            results[method_name]['RT'].append(metrics.get('Avg_RT', 0))
    
    # Aggregate statistics
    summary = {}
    for method in results:
        summary[method] = {
            metric: {
                'mean': np.mean(values),
                'std': np.std(values)
            }
            for metric, values in results[method].items()
        }
    
    # Print results
    print(f"\n{'='*70}")
    print(f"FINAL RESULTS ({num_series} series)")
    print(f"{'='*70}\n")
    
    for method in ['Standard', 'Variance-Trigger', 'ADWIN', 'KSWIN', 'Credal-TTA']:
        print(f"\n{method}:")
        for metric in ['MAE', 'RMSE', 'MAPE', 'RT']:
            mean = summary[method][metric]['mean']
            std = summary[method][metric]['std']
            print(f"  {metric}: {mean:.4f} ({std:.4f})")
    
    # Save results
    output_dir = Path(save_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / f"{model_name}_results.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nResults saved to {output_dir / f'{model_name}_results.json'}")
    
    return summary


def main():
    parser = argparse.ArgumentParser(description="UCI Electricity Experiments")
    parser.add_argument('--model', type=str, default='chronos',
                        choices=['chronos', 'moirai', 'naive'],
                        help='Base TSFM model')
    parser.add_argument('--num_series', type=int, default=20,
                        help='Number of customer series to test')
    parser.add_argument('--save_dir', type=str, default='results/electricity',
                        help='Output directory')
    
    args = parser.parse_args()
    
    run_electricity_experiment(
        model_name=args.model,
        num_series=args.num_series,
        save_dir=args.save_dir
    )


if __name__ == "__main__":
    main()
