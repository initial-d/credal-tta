"""
Financial Crisis Experiments - Reproducing Table 4
S&P 500 COVID crash and Bitcoin 2021 crash
"""

import numpy as np
import argparse
import json
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from credal_tta.credal_tta import CredalTTA
from credal_tta.models.wrappers import ChronosWrapper, PatchTSTWrapper, NaiveBaseline
from credal_tta.utils.data_loader import load_sp500_crisis, load_bitcoin_crash
from credal_tta.utils.metrics import compute_all_metrics


def run_finance_experiment(dataset="sp500", model_name="chronos", num_runs=5):
    """
    Run financial crisis experiments
    
    Args:
        dataset: sp500 or bitcoin
        model_name: chronos or patchtst
        num_runs: Number of random seeds for robustness
    """
    print(f"\n{'='*70}")
    print(f"Financial Crisis Experiment: {dataset.upper()} + {model_name.upper()}")
    print(f"{'='*70}\n")
    
    results = {
        'Standard': {'MAE': [], 'MAPE': [], 'RT': [], 'ATE': []},
        'Variance-Trigger': {'MAE': [], 'MAPE': [], 'RT': [], 'ATE': []},
        'ADWIN': {'MAE': [], 'MAPE': [], 'RT': [], 'ATE': []},
        'Credal-TTA': {'MAE': [], 'MAPE': [], 'RT': [], 'ATE': []},
    }
    
    for run in range(num_runs):
        print(f"\nRun {run+1}/{num_runs}")
        
        # Load data
        if dataset == "sp500":
            data, crash_point = load_sp500_crisis()
        else:  # bitcoin
            data, crash_point = load_bitcoin_crash()
        
        # Initialize model
        if model_name == "chronos":
            base_model = ChronosWrapper()
        elif model_name == "patchtst":
            base_model = PatchTSTWrapper()
        else:
            base_model = NaiveBaseline(method="moving_average")
        
        ground_truth = data[1:]
        
        # --- Standard Fixed Window ---
        print("  Running Standard...")
        standard_preds = []
        context = []
        
        for x_t in data:
            context.append(x_t)
            if len(context) > 512:
                context = context[-512:]
            pred = base_model.predict(np.array(context)) if len(context) >= 10 else x_t
            standard_preds.append(pred)
        
        standard_preds = np.array(standard_preds)[:-1]
        
        # --- Variance-Trigger ---
        print("  Running Variance-Trigger...")
        variance_preds = []
        context = []
        baseline_var = np.var(data[:crash_point-20])
        
        for x_t in data:
            context.append(x_t)
            if len(context) >= 30:
                if np.var(context[-30:]) > 3 * baseline_var:
                    context = context[-10:]
            pred = base_model.predict(np.array(context)) if len(context) >= 10 else x_t
            variance_preds.append(pred)
        
        variance_preds = np.array(variance_preds)[:-1]
        
        # --- ADWIN (simplified) ---
        print("  Running ADWIN...")
        adwin_preds = []
        context = []
        
        for t, x_t in enumerate(data):
            context.append(x_t)
            if len(context) >= 100:
                w1_mean = np.mean(context[-100:-50])
                w2_mean = np.mean(context[-50:])
                if abs(w1_mean - w2_mean) > 2 * np.std(context[-100:]):
                    context = context[-10:]
            pred = base_model.predict(np.array(context)) if len(context) >= 10 else x_t
            adwin_preds.append(pred)
        
        adwin_preds = np.array(adwin_preds)[:-1]
        
        # --- Credal-TTA ---
        print("  Running Credal-TTA...")
        adapter = CredalTTA(
            model=base_model,
            K=3,
            lambda_reset=1.2,
            W_max=512,
            L_min=10
        )
        
        credal_preds = adapter.predict_sequence(data)
        credal_preds = credal_preds[:-1]
        
        # Compute metrics
        for method_name, preds in [
            ('Standard', standard_preds),
            ('Variance-Trigger', variance_preds),
            ('ADWIN', adwin_preds),
            ('Credal-TTA', credal_preds)
        ]:
            metrics = compute_all_metrics(ground_truth, preds, shift_points=[crash_point])
            
            results[method_name]['MAE'].append(metrics['MAE'])
            results[method_name]['MAPE'].append(metrics['MAPE'])
            results[method_name]['RT'].append(metrics.get('Avg_RT', 0))
            results[method_name]['ATE'].append(metrics.get('Avg_ATE', 0))
    
    # Aggregate
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
    print(f"FINAL RESULTS ({num_runs} runs)")
    print(f"{'='*70}\n")
    
    for method in ['Standard', 'Variance-Trigger', 'ADWIN', 'Credal-TTA']:
        print(f"\n{method}:")
        for metric in ['MAE', 'MAPE', 'RT', 'ATE']:
            mean = summary[method][metric]['mean']
            std = summary[method][metric]['std']
            print(f"  {metric}: {mean:.2f} ({std:.2f})")
    
    # Compute improvements
    print(f"\n{'='*70}")
    print("IMPROVEMENTS vs. Standard:")
    print(f"{'='*70}")
    for metric in ['MAE', 'MAPE', 'RT', 'ATE']:
        standard_val = summary['Standard'][metric]['mean']
        credal_val = summary['Credal-TTA'][metric]['mean']
        improvement = 100 * (1 - credal_val / standard_val) if standard_val > 0 else 0
        print(f"  {metric}: {improvement:.1f}%")
    
    # Save results
    save_dir = Path(f"results/finance/{dataset}")
    save_dir.mkdir(parents=True, exist_ok=True)
    
    with open(save_dir / f"{model_name}_results.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nResults saved to {save_dir / f'{model_name}_results.json'}")
    
    return summary


def main():
    parser = argparse.ArgumentParser(description="Financial Crisis Experiments")
    parser.add_argument('--dataset', type=str, default='sp500',
                        choices=['sp500', 'bitcoin'],
                        help='Financial dataset')
    parser.add_argument('--model', type=str, default='chronos',
                        choices=['chronos', 'patchtst', 'naive'],
                        help='Base TSFM model')
    parser.add_argument('--num_runs', type=int, default=1,
                        help='Number of runs for robustness')
    
    args = parser.parse_args()
    
    run_finance_experiment(
        dataset=args.dataset,
        model_name=args.model,
        num_runs=args.num_runs
    )


if __name__ == "__main__":
    main()
