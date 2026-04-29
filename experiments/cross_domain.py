"""
Cross-Domain Generalization Experiment (R3, Table 3)
Tests Credal-TTA across Finance, Demand, Sensor, Energy domains
"""

import numpy as np
import json
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from credal_tta.credal_tta import CredalTTA
from credal_tta.models.wrappers import ChronosWrapper, MoiraiWrapper, PatchTSTWrapper
from credal_tta.utils.data_loader import (
    load_sp500_crisis, load_bitcoin_crash,
    load_uci_electricity, load_noaa_weather, load_ettm1
)
from credal_tta.utils.metrics import compute_all_metrics


def run_cross_domain():
    """Run all 5 datasets across 4 domains"""
    
    datasets = [
        ('Finance', 'S&P500', load_sp500_crisis, ChronosWrapper, {}),
        ('Finance', 'Bitcoin', load_bitcoin_crash, PatchTSTWrapper, {}),
        ('Demand', 'Electricity', load_uci_electricity, MoiraiWrapper, {}),
        ('Sensor', 'NOAA', load_noaa_weather, ChronosWrapper, {}),
        ('Energy', 'ETTm1', load_ettm1, ChronosWrapper, {}),
    ]
    
    results = []
    
    for domain, name, loader, model_cls, kwargs in datasets:
        print(f"\n{'='*60}")
        print(f"{domain} - {name}")
        print(f"{'='*60}")
        
        data, shift_point = loader(**kwargs)
        model = model_cls()
        
        # Standard
        std_preds = []
        ctx = []
        for x in data:
            ctx.append(x)
            if len(ctx) > 512:
                ctx = ctx[-512:]
            std_preds.append(model.predict(np.array(ctx)) if len(ctx) >= 10 else x)
        
        # Credal-TTA
        adapter = CredalTTA(model=model, K=3, lambda_reset=1.3, W_max=512, L_min=10)
        credal_preds = adapter.predict_sequence(data)
        
        # Metrics
        gt = data[1:]
        std_metrics = compute_all_metrics(gt, np.array(std_preds)[:-1], [shift_point])
        credal_metrics = compute_all_metrics(gt, credal_preds[:-1], [shift_point])
        
        rt_reduction = 100 * (1 - credal_metrics.get('Avg_RT', 0) / std_metrics.get('Avg_RT', 1))
        
        results.append({
            'Domain': domain,
            'Dataset': name,
            'MAE_Std': std_metrics['MAE'],
            'MAE_Credal': credal_metrics['MAE'],
            'RT_Std': std_metrics.get('Avg_RT', 0),
            'RT_Credal': credal_metrics.get('Avg_RT', 0),
            'RT_Reduction': rt_reduction
        })
        
        print(f"MAE: {std_metrics['MAE']:.3f} → {credal_metrics['MAE']:.3f}")
        print(f"RT: {std_metrics.get('Avg_RT', 0):.0f} → {credal_metrics.get('Avg_RT', 0):.0f} ({rt_reduction:.1f}%)")
    
    # Save
    Path("results/cross_domain").mkdir(parents=True, exist_ok=True)
    with open("results/cross_domain/table3.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*60}")
    print("Cross-Domain Results Saved")
    print(f"{'='*60}")


if __name__ == "__main__":
    run_cross_domain()
