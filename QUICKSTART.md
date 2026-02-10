# Quick Start Guide

## Installation (5 minutes)

### Option 1: Install from Source (Recommended)

```bash
# Clone the repository
git clone https://github.com/anonymous-repo/credal-tta.git
cd credal-tta

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install package
pip install -e .

# Verify installation
python validate.py
```

### Option 2: Minimal Installation (Core Only)

```bash
# Install only core dependencies
pip install numpy scipy matplotlib scikit-learn

# Clone repository
git clone https://github.com/anonymous-repo/credal-tta.git
cd credal-tta

# Verify
python validate.py
```

### Option 3: Full Installation (With TSFMs)

```bash
# Install with all TSFM models
pip install -e ".[full]"

# Note: This requires PyTorch and transformer models (~5GB)
```

## Quick Demo (2 minutes)

Run the standalone demo to see Credal-TTA in action:

```bash
python examples/demo.py
```

This will:
1. Generate synthetic time series with regime shift
2. Compare standard vs. Credal-TTA forecasting
3. Display performance improvements
4. Save visualization to `demo_output.png`

Expected output:
```
Metric              Standard        Credal-TTA      Improvement    
---------------------------------------------------------------------
MAE                 0.4200          0.2800          33.3%
RMSE                0.5500          0.3700          32.7%
Avg_RT              78.0000         37.0000         52.6%
Avg_ATE             32.1000         12.5000         61.1%
```

## Interactive Tutorial (10 minutes)

Open the Jupyter notebook for step-by-step walkthrough:

```bash
jupyter lab examples/quickstart.ipynb
```

The notebook covers:
- Data generation with regime shifts
- Standard fixed-window forecasting
- Credal-TTA adaptive forecasting
- Visualization of adaptation dynamics
- Performance comparison

## Running Experiments

### Reproduce Paper Results

Run all experiments from the paper (~1 hour):

```bash
bash run_all.sh
```

### Individual Experiments

**Synthetic Benchmarks (Table 2):**
```bash
python experiments/synthetic.py --dataset SinFreq --model chronos --num_runs 10
python experiments/synthetic.py --dataset StepMean --model chronos --num_runs 10
```

**UCI Electricity (Table 3):**
```bash
python experiments/electricity.py --model chronos --num_series 20
```

**Financial Crisis (Table 4):**
```bash
python experiments/finance.py --dataset sp500 --model chronos --num_runs 5
```

**Ablation Studies (Tables 5-7):**
```bash
python experiments/ablation.py --study variance_vs_credal
python experiments/ablation.py --study num_extremes
python experiments/ablation.py --study threshold_sensitivity
```

## Basic Usage

### Example 1: Simple Forecasting

```python
import numpy as np
from credal_tta import CredalTTA
from credal_tta.models.wrappers import NaiveBaseline

# Your time series data
time_series = np.random.randn(1000)

# Initialize model and adapter
model = NaiveBaseline(method="moving_average")
adapter = CredalTTA(model=model, K=3, lambda_reset=1.2)

# Generate predictions
predictions = adapter.predict_sequence(time_series)
```

### Example 2: With Chronos Foundation Model

```python
from credal_tta import CredalTTA
from credal_tta.models.wrappers import ChronosWrapper

# Load Chronos model (requires transformers)
model = ChronosWrapper(model_name="amazon/chronos-t5-base")

# Create adaptive wrapper
adapter = CredalTTA(model=model, K=3, lambda_reset=1.2, W_max=512)

# Predict with diagnostics
predictions, diagnostics = adapter.predict_sequence(
    time_series,
    return_diagnostics=True
)

# Analyze regime shifts
shifts = [d['t'] for d in diagnostics if d['reset_occurred']]
print(f"Detected {len(shifts)} regime shifts at times: {shifts}")
```

### Example 3: Custom Configuration

```python
adapter = CredalTTA(
    model=model,
    K=5,                    # More extreme distributions
    lambda_reset=1.3,       # Higher detection threshold
    lambda_caution=0.9,     # Lower stable threshold
    W_max=256,              # Shorter max context
    L_min=20,               # Larger minimum buffer
    smoothing_alpha=0.15    # Less smoothing
)
```

## Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'credal_tta'"

**Solution:** Install the package first:
```bash
pip install -e .
```

### Issue: "chronos-forecasting not found"

**Solution:** Chronos is optional. Either:
- Install it: `pip install chronos-forecasting`
- Use NaiveBaseline instead for testing

### Issue: "CUDA out of memory"

**Solution:** 
- Use CPU: `ChronosWrapper(device="cpu")`
- Reduce batch size or context length
- Use smaller model: `chronos-t5-small`

### Issue: Tests fail with numerical errors

**Solution:** 
- Check NumPy/SciPy versions: `pip install --upgrade numpy scipy`
- Run: `python validate.py` for diagnostic info

## Performance Tips

### Speed Optimization

1. **Use CPU for HCA**: The credal set operations are lightweight and run fast on CPU
2. **GPU for TSFM**: Use GPU only for the foundation model inference
3. **Reduce K**: Use K=2 or K=3 for real-time applications

### Memory Optimization

1. **Limit W_max**: Use smaller context windows (256 instead of 512)
2. **Clear history**: Call `adapter.reset()` between episodes
3. **Process in batches**: For very long series, split into chunks

## Next Steps

1. **Read the paper**: Understand the theoretical foundations
2. **Explore API docs**: See `docs/API.md` for detailed reference
3. **Try your data**: Replace synthetic data with your own time series
4. **Customize detection**: Tune λ_reset and K for your domain
5. **Contribute**: See `CONTRIBUTING.md` for guidelines

## Getting Help

- **GitHub Issues**: Report bugs or request features
- **Documentation**: Check `docs/API.md` and `PROJECT_STRUCTURE.md`
- **Email**: Contact maintainers at sa613403@mail.ustc.edu.cn

## Citation

If you use Credal-TTA in your research, please cite:

```bibtex
@article{du2025credal,
  title={Breaking Context Inertia: A Neuro-Symbolic Framework for Test-Time Adaptation of Time Series Foundation Models via Credal Set Theory},
  author={Du, Yimin and Wu, Guixing},
  journal={Knowledge-Based Systems},
  year={2025}
}
```
