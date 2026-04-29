# Credal-TTA: Breaking Context Inertia in Time Series Foundation Models

Official implementation of "Breaking Context Inertia: Adaptive Context Management for Time Series Foundation Models in Non-Stationary Environments"

## Overview

Credal-TTA is a training-free neuro-symbolic framework that enables Time Series Foundation Models (TSFMs) to rapidly adapt to regime shifts without retraining. By monitoring epistemic uncertainty through credal set geometry, it eliminates **context inertia** — the tendency of fixed-window models to persist with obsolete distributional assumptions.

## Key Features

- **62–83% faster regime adaptation** across finance, demand, sensor, and energy domains
- **Training-free**: No gradient updates or parameter modifications required
- **Architecture-agnostic**: Works with Chronos, Moirai, PatchTST, and other TSFMs
- **Burn-in health check**: Robust initialization even when structural breaks occur during burn-in
- **Two-track data flow**: TSFM receives raw data; HCA uses preprocessed data for detection
- **Multivariate support**: Diagonal-covariance approximation scales linearly in dimension
- **Interpretable**: Credal set diameter provides human-understandable confidence metrics
- **Lightweight**: <2ms CPU overhead per time step

## Installation

```bash
# Clone repository
git clone https://github.com/initial-d/credal-tta.git
cd credal-tta

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

```python
from credal_tta import CredalTTA
from credal_tta.models.wrappers import ChronosWrapper
import numpy as np

# Load your TSFM
model = ChronosWrapper(model_name="chronos-t5-base")

# Initialize Credal-TTA
credal_adapter = CredalTTA(
    model=model,
    K=3,              # Number of extreme distributions
    lambda_reset=1.2, # Detection threshold
    W_max=512,         # Maximum context length
    L_min=10           # Minimum context buffer
)

# Adaptive forecasting
predictions = []
for t in range(len(time_series)):
    x_t = time_series[t]
    pred = credal_adapter.predict_step(x_t)
    predictions.append(pred)
```

## Project Structure

```
credal-tta/
├── credal_tta/
│   ├── core/
│   │   ├── hca.py                # Hausdorff Context Adapter (with burn-in health check)
│   │   ├── hca_multivariate.py   # Multivariate HCA (diagonal covariance approx.)
│   │   ├── context_manager.py    # Reset-and-Grow logic
│   │   └── credal_set.py         # Credal set operations
│   ├── models/
│   │   ├── wrappers.py           # Unified TSFM wrappers (Chronos, Moirai, PatchTST)
│   │   └── tta_baselines.py      # LoRA-TTA & TENT-TTA gradient baselines
│   ├── utils/
│   │   ├── metrics.py            # Evaluation metrics (MAE, RMSE, RT, ATE)
│   │   ├── data_loader.py        # All dataset loaders (5 datasets)
│   │   └── preprocessing.py      # Two-track preprocessing pipeline
│   └── credal_tta.py             # Main framework entry point
├── experiments/
│   ├── synthetic.py              # Synthetic benchmarks (SinFreq, StepMean)
│   ├── finance.py                # Financial crisis experiments (S&P 500, Bitcoin)
│   ├── electricity.py            # UCI Electricity demand experiments
│   ├── cross_domain.py           # Cross-domain evaluation (Table 3)
│   ├── gradient_comparison.py    # LoRA/TENT vs Credal-TTA (Section 5.3)
│   ├── multivariate_etth1.py     # ETTh1 multivariate benchmark (Appendix B)
│   ├── ablation.py               # Ablation studies
│   └── validation/
│       ├── burnin_stress_test.py     # Burn-in health check stress test (Table A2)
│       └── regime_separation.py      # Wasserstein regime separation (Table A1)
├── data/                         # Dataset storage
├── validate_all.py               # Validate all modules are properly installed
└── requirements.txt
```

## Reproducing Paper Results

### 0. Validate Environment

```bash
python validate_all.py
```

### 1. Synthetic Benchmarks (Tables 1–2)

```bash
python experiments/synthetic.py --dataset SinFreq --model chronos
python experiments/synthetic.py --dataset StepMean --model chronos
```

### 2. Financial Crisis (Tables 1–2)

```bash
python experiments/finance.py --dataset sp500 --model chronos
python experiments/finance.py --dataset bitcoin --model patchtst
```

### 3. Gradient-Based Baseline Comparison — Section 5.3 (R1-W4)

Compares Credal-TTA against LoRA-TTA (1-step & 5-step) and TENT-TTA:

```bash
python experiments/gradient_comparison.py --dataset SinFreq --model chronos
python experiments/gradient_comparison.py --dataset sp500 --model chronos
```

### 4. Cross-Domain Generalization — Table 3, Section 5.5 (R3)

Evaluates across all 5 real-world datasets (S&P 500, Bitcoin, UCI Electricity, NOAA Weather, ETTm1):

```bash
python experiments/cross_domain.py
```

### 5. Multivariate Extension — Appendix B (R1-W3)

ETTh1 benchmark (d=7) with diagonal-covariance HCA:

```bash
python experiments/multivariate_etth1.py
```

### 6. Validation Experiments — Appendix A.4 & A.5 (R1-W1, R1-W2)

```bash
# Burn-in health check stress test (Table A2)
python experiments/validation/burnin_stress_test.py

# Wasserstein regime separation analysis (Table A1)
python experiments/validation/regime_separation.py
```

### 7. Ablation Studies (Tables 4–6)

```bash
python experiments/ablation.py --study variance_vs_credal
python experiments/ablation.py --study num_extremes
python experiments/ablation.py --study distance_metric
python experiments/ablation.py --study threshold_sensitivity
```

## Revision Summary

| Concern | Reviewer | New Code | Paper Section |
|---------|----------|----------|---------------|
| Winsorization vs. Black Swan tension | R1-W1 | `preprocessing.py`, `regime_separation.py` | Remark 2, App. A.4 |
| Burn-in contamination risk | R1-W2 | `hca.py` (health check), `burnin_stress_test.py` | Alg. 1, Remark 3, App. A.5 |
| No multivariate benchmark | R1-W3 | `hca_multivariate.py`, `multivariate_etth1.py` | App. B |
| Missing LoRA/TENT baselines | R1-W4 | `tta_baselines.py`, `gradient_comparison.py` | Sec. 5.3, Tables 1–2 |
| Only financial real-world data | R3/AE | `data_loader.py`, `cross_domain.py` | Table 3, Sec. 5.5 |

## Citation

```bibtex

```

## License

MIT License

## Contact

For questions or issues, please open a GitHub issue or contact authors.
