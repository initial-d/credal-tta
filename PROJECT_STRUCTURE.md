# Credal-TTA Project Structure

```
credal-tta/
│
├── README.md                    # Project overview and quick start
├── LICENSE                      # MIT License
├── CHANGELOG.md                 # Version history
├── CONTRIBUTING.md              # Contribution guidelines
├── setup.py                     # Package installation script
├── requirements.txt             # Python dependencies
├── run_all.sh                   # Batch script to run all experiments
│
├── credal_tta/                  # Main package
│   ├── __init__.py             # Package initialization
│   ├── credal_tta.py           # Main CredalTTA framework
│   │
│   ├── core/                    # Core algorithms
│   │   ├── __init__.py
│   │   ├── credal_set.py       # Credal set operations & Wasserstein distance
│   │   ├── hca.py              # Hausdorff Context Adapter
│   │   └── context_manager.py  # Reset-and-Grow mechanism
│   │
│   ├── models/                  # TSFM wrappers
│   │   ├── __init__.py
│   │   └── wrappers.py         # Chronos/Moirai/PatchTST interfaces
│   │
│   └── utils/                   # Utilities
│       ├── __init__.py
│       ├── metrics.py          # Evaluation metrics (MAE, RT, ATE)
│       ├── data_loader.py      # Dataset loaders
│       └── visualization.py    # Plotting functions
│
├── experiments/                 # Experiment scripts
│   ├── synthetic.py            # Table 2: Synthetic benchmarks
│   ├── electricity.py          # Table 3: UCI Electricity
│   ├── finance.py              # Table 4: Financial crises
│   └── ablation.py             # Tables 5-7: Ablation studies
│
├── examples/                    # Tutorials and demos
│   ├── quickstart.ipynb        # Interactive Jupyter tutorial
│   └── demo.py                 # Standalone demo script
│
├── tests/                       # Unit tests
│   └── test_credal_tta.py      # Test suite (pytest)
│
├── docs/                        # Documentation
│   └── API.md                  # API reference
│
├── data/                        # Data directory (user-provided)
│   ├── electricity.npy         # UCI Electricity dataset
│   ├── sp500.csv               # S&P 500 prices
│   └── bitcoin.csv             # Bitcoin prices
│
└── results/                     # Experiment outputs (auto-generated)
    ├── synthetic/
    ├── electricity/
    ├── finance/
    └── ablation/
```

## Key Files Explained

### Core Implementation

**`credal_tta/credal_tta.py`** (150 lines)
- Main framework integrating HCA, Context Manager, and TSFM
- `CredalTTA` class with `predict_step()` and `predict_sequence()` methods
- Wraps arbitrary TSFMs for adaptive forecasting

**`credal_tta/core/credal_set.py`** (200 lines)
- `GaussianDistribution`: Single extreme distribution
- `CredalSet`: Finitely-generated credal set with diameter computation
- `wasserstein_2_gaussian()`: Closed-form W2 distance for Gaussians
- `initialize_credal_set()`: Initialization from burn-in data

**`credal_tta/core/hca.py`** (150 lines)
- `HausdorffContextAdapter`: Online Bayesian updating of credal set
- Contraction ratio computation and regime detection
- Epistemic vs. aleatoric uncertainty separation

**`credal_tta/core/context_manager.py`** (80 lines)
- `ContextManager`: Dynamic context window with reset-and-grow
- Origin tracking and surgical pruning logic

### Experiments

**`experiments/synthetic.py`** (200 lines)
- Reproduces Table 2 (SinFreq, StepMean benchmarks)
- Compares Standard, Variance-Trigger, and Credal-TTA
- Configurable number of runs and random seeds

**`experiments/electricity.py`** (250 lines)
- Reproduces Table 3 (UCI Electricity experiments)
- Tests Chronos and Moirai models
- Includes ADWIN and KSWIN baselines

**`experiments/finance.py`** (200 lines)
- Reproduces Table 4 (S&P 500, Bitcoin crashes)
- Stress tests with extreme volatility
- Multiple runs for statistical robustness

**`experiments/ablation.py`** (300 lines)
- Reproduces Tables 5-7 (ablation studies)
- Variance vs. credal comparison
- Sensitivity to K, λ_reset, distance metrics

### Utilities

**`credal_tta/utils/metrics.py`** (150 lines)
- Standard metrics: MAE, RMSE, MAPE
- Adaptation metrics: Recovery Time (RT), Accumulated Transition Error (ATE)
- `compute_all_metrics()`: Unified evaluation

**`credal_tta/utils/data_loader.py`** (200 lines)
- Synthetic data generators (SinFreq, StepMean, GradualDrift)
- Real-world loaders (Electricity, S&P 500, Bitcoin)
- Unified `get_dataset()` interface

**`credal_tta/utils/visualization.py`** (150 lines)
- `plot_synthetic_comparison()`: Figure 3 in paper
- `plot_recovery_time_comparison()`: Figure 4
- `plot_ablation_heatmap()`: Ablation results

## Running the Code

### Quick Demo
```bash
python examples/demo.py
```

### Single Experiment
```bash
python experiments/synthetic.py --dataset SinFreq --model chronos --num_runs 10
```

### All Experiments (Reproduce Paper)
```bash
bash run_all.sh
```

### Tests
```bash
python -m pytest tests/
```

## Estimated Runtimes

- **Synthetic benchmarks**: ~5 minutes (10 runs, CPU)
- **Electricity experiments**: ~30 minutes (20 series, Chronos on GPU)
- **Finance experiments**: ~10 minutes (5 runs)
- **Ablation studies**: ~15 minutes
- **Total (all experiments)**: ~1 hour

## Hardware Requirements

- **Minimum**: CPU with 4GB RAM (for synthetic + ablation)
- **Recommended**: GPU with 8GB VRAM (for Chronos/Moirai experiments)
- **Storage**: ~500MB (including datasets and results)

## Key Design Principles

1. **Modularity**: Core algorithms (HCA, Context Manager) decoupled from TSFM wrappers
2. **Extensibility**: Easy to add new models, baselines, or datasets
3. **Reproducibility**: Fixed random seeds, comprehensive logging
4. **Efficiency**: Lightweight CPU operations (~2ms overhead per step)
5. **Clarity**: Well-documented code with type hints and docstrings
