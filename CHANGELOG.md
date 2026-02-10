# Changelog

All notable changes to Credal-TTA will be documented in this file.

## [1.0.0] - 2025-02-01

### Initial Release

#### Core Features
- **Hausdorff Context Adapter (HCA)**: Lightweight epistemic uncertainty detector based on credal set theory
- **Reset-and-Grow Context Manager**: Dynamic context pruning to eliminate context inertia
- **Model-Agnostic Interface**: Support for Chronos, Moirai, and PatchTST foundation models

#### Experiments
- Synthetic benchmarks (SinFreq, StepMean)
- UCI Electricity load forecasting
- Financial crisis datasets (S&P 500 COVID crash, Bitcoin 2021 crash)
- Comprehensive ablation studies

#### Baselines
- Standard fixed-window approach
- Variance-Trigger baseline
- ADWIN drift detection
- KSWIN (Kolmogorov-Smirnov window)

#### Documentation
- Complete API reference
- Quick start tutorial (Jupyter notebook)
- Example scripts for all experiments
- Unit tests with >85% coverage

#### Key Results
- 40-60% faster regime adaptation vs. standard approaches
- Significant reduction in accumulated transition error
- Architecture-agnostic performance across multiple TSFMs

### Known Limitations
- Ultra-high-frequency data (>1000Hz) may experience update latency
- Very short context windows (W<100) show minimal benefit
- Stationary environments may see small overhead from unnecessary monitoring

### Future Work
- Multivariate time series support
- Integration with prompt engineering techniques
- Adversarial robustness improvements
- GPU acceleration for credal set operations
