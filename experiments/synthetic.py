"""
Synthetic Benchmark Experiments - Reproducing Table 2 in paper
"""

import numpy as np
import argparse
import json
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from credal_tta.credal_tta import CredalTTA
from credal_tta.models.wrappers import ChronosWrapper, NaiveBaseline
from credal_tta.utils.data_loader import get_dataset
from credal_tta.utils.metrics import compute_all_metrics
from credal_tta.core.hca import HausdorffContextAdapter


def variance_baseline(time_series, model, threshold_multiplier=2.0):
    """Variance-Trigger baseline (Section 5.5)"""
    predictions = []
    context = []
    window_size = 50
    
    # Compute baseline variance
    baseline_var = np.var(time_series[:100])
    threshold = threshold_multiplier * baseline_var
    
    for t, x_t in enumerate(time_series):
        context.append(x_t)
        
        # Check variance trigger
        if len(context) >= window_size:
            recent_var = np.var(context[-window_size:])
            if recent_var > threshold:
                # Reset context
                context = context[-10:]
        
        # Predict
        if len(context) >= 10:
            pred = model.predict(np.array(context))
        else:
            pred = x_t
        predictions.append(pred)
    
    return np.array(predictions)


def run_single_experiment(dataset_name, model_name, num_runs=10, save_dir="results"):
    """
    Run single experiment configuration
    
    Args:
        dataset_name: SinFreq or StepMean
        model_name: chronos or naive
        num_runs: Number of random seeds
        save_dir: Output directory
    """
    results = {
        'Standard': {'MAE': [], 'RMSE': [], 'RT': [], 'ATE': []},
        'Variance-Trigger': {'MAE': [], 'RMSE': [], 'RT': [], 'ATE': []},
        'Credal-TTA': {'MAE': [], 'RMSE': [], 'RT': [], 'ATE': []},
    }
    
    for run in range(num_runs):
        print(f"\n=== Run {run+1}/{num_runs} ===")
        
        # Generate data
        if dataset_name == 'StepMean':
            time_series, shift_point = get_dataset(dataset_name, seed=42 + run, std=0.1)
        else:
            time_series, shift_point = get_dataset(dataset_name, seed=42 + run)
        
        # Initialize model
        if model_name == "chronos":
            base_model = ChronosWrapper()
        else:
            base_model = NaiveBaseline(method="moving_average")
        
        # --- Standard (Fixed Window) ---
        print("Running Standard...")
        standard_preds = []
        context_standard = []
        W_max = 512
        
        for x_t in time_series:
            context_standard.append(x_t)
            if len(context_standard) > W_max:
                context_standard = context_standard[-W_max:]
            
            if len(context_standard) >= 10:
                pred = base_model.predict(np.array(context_standard))
            else:
                pred = x_t
            standard_preds.append(pred)
        
        standard_preds = np.array(standard_preds)
        
        # --- Variance-Trigger ---
        print("Running Variance-Trigger...")
        variance_preds = variance_baseline(time_series, base_model)
        
        # --- Credal-TTA ---
        print("Running Credal-TTA...")
        adapter = CredalTTA(
            model=base_model,
            K=3,
            lambda_reset=3.5,
            W_max=512,
            L_min=64,
            smoothing_alpha=1.0
        )
        
        credal_preds = adapter.predict_sequence(time_series)
        
        # Compute metrics
        ground_truth = time_series[1:]  # One-step-ahead targets
        
        for method, preds in [
            ('Standard', standard_preds[:-1]),
            ('Variance-Trigger', variance_preds[:-1]),
            ('Credal-TTA', credal_preds[:-1])
        ]:
            metrics = compute_all_metrics(ground_truth, preds, shift_points=[shift_point])
            results[method]['MAE'].append(metrics['MAE'])
            results[method]['RMSE'].append(metrics['RMSE'])
            results[method]['RT'].append(metrics.get('Avg_RT', 0))
            results[method]['ATE'].append(metrics.get('Avg_ATE', 0))
    
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
    print(f"\n{'='*60}")
    print(f"Results for {dataset_name} + {model_name}")
    print(f"{'='*60}")
    for method in ['Standard', 'Variance-Trigger', 'Credal-TTA']:
        print(f"\n{method}:")
        for metric in ['MAE', 'RMSE', 'RT', 'ATE']:
            mean = summary[method][metric]['mean']
            std = summary[method][metric]['std']
            print(f"  {metric}: {mean:.3f} ({std:.3f})")
    
    # Save results
    output_dir = Path(save_dir) / dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / f"{model_name}_results.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    return summary


def main():
    parser = argparse.ArgumentParser(description="Synthetic Benchmark Experiments")
    parser.add_argument('--dataset', type=str, default='SinFreq',
                        choices=['SinFreq', 'StepMean'],
                        help='Synthetic dataset')
    parser.add_argument('--model', type=str, default='chronos',
                        choices=['chronos', 'naive'],
                        help='Base TSFM model')
    parser.add_argument('--num_runs', type=int, default=10,
                        help='Number of random seeds')
    parser.add_argument('--save_dir', type=str, default='results/synthetic',
                        help='Output directory')
    
    args = parser.parse_args()
    
    run_single_experiment(
        dataset_name=args.dataset,
        model_name=args.model,
        num_runs=args.num_runs,
        save_dir=args.save_dir
    )


if __name__ == "__main__":
    main()
