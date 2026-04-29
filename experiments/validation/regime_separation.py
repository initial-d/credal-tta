"""
Regime Separation Validation (R1-W1, Table A1)
Confirms Wasserstein separation preserved after preprocessing
"""

import numpy as np
import json
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from credal_tta.utils.data_loader import load_sp500_crisis, load_bitcoin_crash
from scipy.stats import wasserstein_distance


def compute_regime_separation(data, crisis_point, window=60):
    """Compute W2 distance between pre/crisis regimes"""
    pre_data = data[max(0, crisis_point - window):crisis_point]
    crisis_data = data[crisis_point:crisis_point + 20]
    
    # Raw W2
    w2_raw = wasserstein_distance(pre_data, crisis_data)
    
    # Preprocessed W2
    pre_log = np.sign(pre_data) * np.log1p(np.abs(pre_data))
    crisis_log = np.sign(crisis_data) * np.log1p(np.abs(crisis_data))
    w2_preprocessed = wasserstein_distance(pre_log, crisis_log)
    
    # Normalize by std
    std_raw = np.std(pre_data)
    std_log = np.std(pre_log)
    
    return {
        'w2_raw': w2_raw,
        'w2_raw_sigma': w2_raw / std_raw if std_raw > 0 else 0,
        'w2_preprocessed': w2_preprocessed,
        'w2_preprocessed_sigma': w2_preprocessed / std_log if std_log > 0 else 0
    }


def validate_regime_separation():
    """Table A1: Regime separation after preprocessing"""
    
    results = []
    
    # S&P 500
    sp500_data, sp500_crisis = load_sp500_crisis()
    sp500_sep = compute_regime_separation(sp500_data, sp500_crisis)
    results.append({
        'dataset': 'S&P 500 COVID',
        **sp500_sep
    })
    print(f"S&P 500: {sp500_sep['w2_raw_sigma']:.1f}σ → {sp500_sep['w2_preprocessed_sigma']:.1f}σ")
    
    # Bitcoin
    btc_data, btc_crisis = load_bitcoin_crash()
    btc_sep = compute_regime_separation(btc_data, btc_crisis)
    results.append({
        'dataset': 'Bitcoin May 2021',
        **btc_sep
    })
    print(f"Bitcoin: {btc_sep['w2_raw_sigma']:.1f}σ → {btc_sep['w2_preprocessed_sigma']:.1f}σ")
    
    # Save
    Path("results/validation").mkdir(parents=True, exist_ok=True)
    with open("results/validation/table_a1_regime_sep.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\nRegime separation validation complete.")


if __name__ == "__main__":
    validate_regime_separation()
