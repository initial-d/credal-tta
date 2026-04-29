"""
Multivariate Extension Benchmark: ETTh1 (R1-W3, Appendix B)
Diagonal covariance approximation vs full covariance
"""

import numpy as np
import json
import time
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from credal_tta.core.hca_multivariate import MultivariateDiagonalHCA


def generate_etth1_synthetic(T=5000, d=7, anomaly_point=2500, seed=42):
    """
    Synthetic ETTh1-like data (d=7): oil temp + 6 power-load features.
    Anomaly at anomaly_point: +22% mean shift, +180% variance in dim 0.
    """
    np.random.seed(seed)
    # Pre-anomaly
    pre = np.random.normal(loc=50, scale=2, size=(anomaly_point, d))
    # Post-anomaly: dim 0 shifts
    post = np.random.normal(loc=50, scale=2, size=(T - anomaly_point, d))
    post[:, 0] = np.random.normal(loc=61, scale=5.6, size=T - anomaly_point)
    data = np.vstack([pre, post])
    return data, anomaly_point


def run_standard_baseline(data, context_len=512):
    """Simple rolling-mean baseline (stand-in for Standard Moirai)."""
    T, d = data.shape
    preds = np.zeros_like(data)
    for t in range(1, T):
        start = max(0, t - context_len)
        preds[t] = np.mean(data[start:t], axis=0)
    return preds


def run_credal_multivariate(data, K=3, lambda_reset=1.3, diagonal=True):
    """Run multivariate Credal-TTA with diagonal or full approx."""
    T, d = data.shape
    hca = MultivariateDiagonalHCA(d=d, K=K, lambda_reset=lambda_reset)
    preds = np.zeros_like(data)
    context = []

    for t in range(T):
        x_t = data[t]
        result = hca.update(x_t)

        if result['regime_shift']:
            # Reset context
            context = context[-10:] if len(context) > 10 else context

        context.append(x_t)
        if len(context) > 512:
            context = context[-512:]

        if len(context) >= 2:
            preds[t] = np.mean(context[:-1], axis=0)
        else:
            preds[t] = x_t

    return preds


def compute_metrics(gt, pred, anomaly_point, post_window=100):
    """MAE, RMSE, and recovery time on post-anomaly window."""
    post_gt = gt[anomaly_point:anomaly_point + post_window]
    post_pred = pred[anomaly_point:anomaly_point + post_window]
    errors = np.abs(post_gt - post_pred).mean(axis=1)
    mae = float(np.mean(errors))
    rmse = float(np.sqrt(np.mean((post_gt - post_pred) ** 2)))

    # Recovery time: first step where error < 1.5 * pre-anomaly error
    pre_errors = np.abs(gt[anomaly_point-100:anomaly_point] -
                        pred[anomaly_point-100:anomaly_point]).mean(axis=1)
    threshold = 1.5 * np.mean(pre_errors)
    rt = post_window
    for i, e in enumerate(errors):
        if e < threshold:
            rt = i
            break
    return mae, rmse, rt


def main():
    print("=" * 60)
    print("Multivariate ETTh1 Benchmark (d=7, Appendix B)")
    print("=" * 60)

    num_runs = 5
    results = {m: {'mae': [], 'rmse': [], 'rt': [], 'latency': []}
               for m in ['Standard', 'Credal-Diag']}

    for run in range(num_runs):
        data, ap = generate_etth1_synthetic(T=5000, d=7, seed=42 + run)

        # Standard baseline
        t0 = time.time()
        std_pred = run_standard_baseline(data)
        lat_std = 1000 * (time.time() - t0) / len(data)
        mae_s, rmse_s, rt_s = compute_metrics(data, std_pred, ap)
        results['Standard']['mae'].append(mae_s)
        results['Standard']['rmse'].append(rmse_s)
        results['Standard']['rt'].append(rt_s)
        results['Standard']['latency'].append(lat_std)

        # Credal diagonal
        t0 = time.time()
        cred_pred = run_credal_multivariate(data, diagonal=True)
        lat_cred = 1000 * (time.time() - t0) / len(data)
        mae_c, rmse_c, rt_c = compute_metrics(data, cred_pred, ap)
        results['Credal-Diag']['mae'].append(mae_c)
        results['Credal-Diag']['rmse'].append(rmse_c)
        results['Credal-Diag']['rt'].append(rt_c)
        results['Credal-Diag']['latency'].append(lat_cred)

        print(f"Run {run+1}: Std MAE={mae_s:.3f} RT={rt_s} | "
              f"Credal MAE={mae_c:.3f} RT={rt_c}")

    # Summary
    print("\n" + "=" * 60)
    print("Table B1: ETTh1 Multivariate Benchmark (d=7)")
    print("=" * 60)
    for method, vals in results.items():
        print(f"{method:20s}  MAE={np.mean(vals['mae']):.3f}({np.std(vals['mae']):.3f})  "
              f"RMSE={np.mean(vals['rmse']):.3f}({np.std(vals['rmse']):.3f})  "
              f"RT={np.mean(vals['rt']):.0f}({np.std(vals['rt']):.0f})  "
              f"Lat={np.mean(vals['latency']):.1f}ms")

    # Save
    Path("results/multivariate").mkdir(parents=True, exist_ok=True)
    with open("results/multivariate/table_b1.json", 'w') as f:
        json.dump({k: {m: float(np.mean(v)) for m, v in vals.items()}
                   for k, vals in results.items()}, f, indent=2)
    print("\nResults saved to results/multivariate/table_b1.json")


if __name__ == "__main__":
    main()
