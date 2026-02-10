# Credal-TTA: Breaking Context Inertia in Time Series Foundation Models

Official implementation of "Breaking Context Inertia: A Neuro-Symbolic Framework for Test-Time Adaptation of Time Series Foundation Models via Credal Set Theory"

## Overview

Credal-TTA is a training-free neuro-symbolic framework that enables Time Series Foundation Models (TSFMs) to rapidly adapt to regime shifts without retraining. By monitoring epistemic uncertainty through credal set geometry, it eliminates **context inertia** - the tendency of fixed-window models to persist with obsolete distributional assumptions.

## Key Features

- **40-60% faster regime adaptation** compared to standard fixed-window approaches
- **Training-free**: No gradient updates or parameter modifications required
- **Architecture-agnostic**: Works with Chronos, Moirai, PatchTST, and other TSFMs
- **Interpretable**: Credal set diameter provides human-understandable confidence metrics
- **Lightweight**: <2ms CPU overhead per time step

## Installation

```bash
# Clone repository
git clone https://github.com/anonymous-repo/credal-tta.git
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
from credal_tta.models import ChronosWrapper
import numpy as np

# Load your TSFM
model = ChronosWrapper(model_name="chronos-t5-base")

# Initialize Credal-TTA
credal_adapter = CredalTTA(
    model=model,
    K=3,  # Number of extreme distributions
    lambda_reset=1.2,  # Detection threshold
    W_max=512,  # Maximum context length
    L_min=10  # Minimum context buffer
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
│   │   ├── hca.py              # Hausdorff Context Adapter
│   │   ├── context_manager.py  # Reset-and-Grow logic
│   │   └── credal_set.py       # Credal set operations
│   ├── models/
│   │   ├── chronos_wrapper.py  # Chronos interface
│   │   ├── moirai_wrapper.py   # Moirai interface
│   │   └── patchtst_wrapper.py # PatchTST interface
│   ├── utils/
│   │   ├── metrics.py          # Evaluation metrics
│   │   └── data_loader.py      # Dataset utilities
│   └── credal_tta.py           # Main framework
├── experiments/
│   ├── synthetic.py            # Synthetic benchmarks
│   ├── electricity.py          # UCI Electricity experiments
│   ├── finance.py              # Financial crisis experiments
│   └── ablation.py             # Ablation studies
├── data/                       # Dataset storage
├── results/                    # Experiment outputs
└── requirements.txt
```

## Reproducing Paper Results

### Synthetic Benchmarks (Table 2)
```bash
python experiments/synthetic.py --dataset SinFreq --model chronos
python experiments/synthetic.py --dataset StepMean --model chronos
```

### UCI Electricity (Table 3)
```bash
python experiments/electricity.py --model chronos --num_series 20
python experiments/electricity.py --model moirai --num_series 20
```

### Financial Crisis (Table 4)
```bash
python experiments/finance.py --dataset sp500 --model chronos
python experiments/finance.py --dataset bitcoin --model patchtst
```

### Ablation Studies (Tables 5-7)
```bash
# Variance vs. Credal comparison
python experiments/ablation.py --study variance_vs_credal

# Number of extremes K
python experiments/ablation.py --study num_extremes

# Distance metric comparison
python experiments/ablation.py --study distance_metric

# Threshold sensitivity
python experiments/ablation.py --study threshold_sensitivity
```

## Citation

```bibtex

```

## License

MIT License

## Contact

For questions or issues, please open a GitHub issue or contact authors
