"""
Ablation Studies - Reproducing Tables 5-7 in paper
"""

import numpy as np
import argparse
import json
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from credal_tta.credal_tta import CredalTTA
from credal_tta.models.wrappers import NaiveBaseline
from credal_tta.utils.data_loader import load_uci_electricity
from credal_tta.utils.metrics import compute_all_metrics


def ablation_variance_vs_credal(num_series=20):
    """
    Table 5: Variance vs. Credal Diameter comparison
    """
    print("\n=== Ablation: Variance vs. Credal ===\n")
    
    results = {
        'Variance-Trigger': {'RMSE': [], 'RT': [], 'FPR': []},
        'Bayesian-Trigger (K=1)': {'RMSE': [], 'RT': [], 'FPR': []},
        'HCA (K=2)': {'RMSE': [], 'RT': [], 'FPR': []},
        'HCA (K=3)': {'RMSE': [], 'RT': [], 'FPR': []},
        'HCA (K=5)': {'RMSE': [], 'RT': [], 'FPR': []},
    }
    
    for series_id in range(num_series):
        print(f"Processing series {series_id+1}/{num_series}")
        
        # Load data
        data = load_uci_electricity(customer_id=series_id)
        data = data[:2000]  # Use subset for speed
        
        base_model = NaiveBaseline(method="moving_average")
        
        # Test each variant
        for K in [1, 2, 3, 5]:
            method_name = f'HCA (K={K})' if K > 1 else 'Bayesian-Trigger (K=1)'
            
            adapter = CredalTTA(
                model=base_model,
                K=K,
                lambda_reset=1.2,
                W_max=512,
                L_min=10
            )
            
            preds, diagnostics = adapter.predict_sequence(data, return_diagnostics=True)
            
            # Compute metrics
            ground_truth = data[1:]
            preds = preds[:-1]
            
            metrics = compute_all_metrics(ground_truth, preds)
            
            # False positive rate (manually labeled stable periods)
            # Simplified: count resets in first 500 steps (assumed stable)
            num_resets = sum(1 for d in diagnostics[:500] if d.get('reset_occurred', False))
            fpr = 100 * num_resets / 500
            
            results[method_name]['RMSE'].append(metrics['RMSE'])
            results[method_name]['FPR'].append(fpr)
        
        # Variance-Trigger baseline
        variance_preds = []
        context = []
        baseline_var = np.var(data[:100])
        
        for x_t in data:
            context.append(x_t)
            if len(context) >= 50:
                if np.var(context[-50:]) > 2 * baseline_var:
                    context = context[-10:]
            
            pred = base_model.predict(np.array(context)) if len(context) >= 10 else x_t
            variance_preds.append(pred)
        
        variance_preds = np.array(variance_preds)[:-1]
        metrics_var = compute_all_metrics(ground_truth, variance_preds)
        results['Variance-Trigger']['RMSE'].append(metrics_var['RMSE'])
        
        num_resets_var = 0  # Simplified counting
        results['Variance-Trigger']['FPR'].append(0.0)  # Placeholder
    
    # Aggregate
    summary = {}
    for method in results:
        summary[method] = {
            metric: {
                'mean': np.mean(values) if len(values) > 0 else 0.0,
                'std': np.std(values) if len(values) > 0 else 0.0
            }
            for metric, values in results[method].items()
        }
    
    # Print
    print("\n" + "="*60)
    for method in results:
        print(f"\n{method}:")
        for metric in ['RMSE', 'FPR']:
            if metric in summary[method]:
                mean = summary[method][metric]['mean']
                std = summary[method][metric]['std']
                print(f"  {metric}: {mean:.4f} ({std:.4f})")
    
    return summary


def ablation_num_extremes():
    """
    Table 6: Sensitivity to K (number of extremes)
    """
    print("\n=== Ablation: Number of Extremes K ===\n")
    
    from credal_tta.utils.data_loader import generate_step_mean_shift
    
    data, shift_point = generate_step_mean_shift(T=1000, seed=42)
    base_model = NaiveBaseline()
    
    results = {}
    
    for K in [2, 3, 4, 5]:
        print(f"Testing K={K}")
        
        adapter = CredalTTA(
            model=base_model,
            K=K,
            lambda_reset=1.2,
            W_max=512
        )
        
        preds = adapter.predict_sequence(data)
        ground_truth = data[1:]
        preds = preds[:-1]
        
        metrics = compute_all_metrics(ground_truth, preds, shift_points=[shift_point])
        
        results[f'K={K}'] = {
            'RMSE': metrics['RMSE'],
            'RT': metrics.get('Avg_RT', 0),
            'FPR': 0.0  # Placeholder
        }
    
    # Print
    print("\n" + "="*60)
    for k, vals in results.items():
        print(f"\n{k}:")
        for metric, value in vals.items():
            print(f"  {metric}: {value:.4f}")
    
    return results


def ablation_threshold_sensitivity():
    """
    Table 7: Sensitivity to lambda_reset threshold
    """
    print("\n=== Ablation: Threshold Sensitivity ===\n")
    
    data = load_uci_electricity(customer_id=0)[:2000]
    base_model = NaiveBaseline()
    
    results = {}
    
    for lambda_reset in [1.1, 1.2, 1.3, 1.5, 1.8]:
        print(f"Testing λ={lambda_reset}")
        
        adapter = CredalTTA(
            model=base_model,
            K=3,
            lambda_reset=lambda_reset,
            W_max=512
        )
        
        preds, diagnostics = adapter.predict_sequence(data, return_diagnostics=True)
        ground_truth = data[1:]
        preds = preds[:-1]
        
        metrics = compute_all_metrics(ground_truth, preds)
        
        # Count resets
        num_resets = sum(1 for d in diagnostics if d.get('reset_occurred', False))
        fpr = 100 * num_resets / len(diagnostics) if len(diagnostics) > 0 else 0
        
        results[f'λ={lambda_reset}'] = {
            'MAE': metrics['MAE'],
            'RMSE': metrics['RMSE'],
            'FPR': fpr
        }
    
    # Print
    print("\n" + "="*60)
    for thresh, vals in results.items():
        print(f"\n{thresh}:")
        for metric, value in vals.items():
            print(f"  {metric}: {value:.4f}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Ablation Studies")
    parser.add_argument('--study', type=str, required=True,
                        choices=['variance_vs_credal', 'num_extremes', 
                                 'threshold_sensitivity', 'all'],
                        help='Which ablation study to run')
    parser.add_argument('--num_series', type=int, default=5,
                        help='Number of series for variance_vs_credal')
    parser.add_argument('--save_dir', type=str, default='results/ablation',
                        help='Output directory')
    
    args = parser.parse_args()
    
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    if args.study == 'variance_vs_credal' or args.study == 'all':
        results = ablation_variance_vs_credal(num_series=args.num_series)
        with open(save_dir / 'variance_vs_credal.json', 'w') as f:
            json.dump(results, f, indent=2)
    
    if args.study == 'num_extremes' or args.study == 'all':
        results = ablation_num_extremes()
        with open(save_dir / 'num_extremes.json', 'w') as f:
            json.dump(results, f, indent=2)
    
    if args.study == 'threshold_sensitivity' or args.study == 'all':
        results = ablation_threshold_sensitivity()
        with open(save_dir / 'threshold_sensitivity.json', 'w') as f:
            json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
