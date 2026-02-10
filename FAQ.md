# Frequently Asked Questions (FAQ)

## Installation & Setup

### Q: How do I install Credal-TTA?

**A:** Three options:

```bash
# Option 1: Basic installation (recommended for testing)
pip install -e .

# Option 2: Full installation with TSFM models
pip install -e ".[full]"

# Option 3: Development installation
pip install -e ".[dev]"
```

### Q: Do I need a GPU?

**A:** Not required, but recommended:
- **CPU only**: Works fine for core algorithms and synthetic experiments (~5 min)
- **GPU**: Needed for Chronos/Moirai models in real experiments (~1 hour)

---

## Demo Issues

### Q: Why doesn't `demo.py` or `demo_enhanced.py` detect any regime shifts?

**A:** There are **two fundamental issues**:

1. **Long burn-in period** (50 samples): HCA doesn't start monitoring until 50 observations are collected, so regime shifts before t=50 are invisible.

2. **Wrong monitoring target**: The original demos monitor **raw values**, but moving average predictions are too smooth. We should monitor **prediction errors** instead!

**Diagnosis from your output:**
```
Max diameter: 0.0051          ← Nearly zero!
Max contraction ratio: 1.0000 ← Never expands
```

This means credal set never saw significant changes because prediction errors were too small.

**Solutions:**

1. **Use the working demo** (monitors prediction errors):
   ```bash
   python examples/demo_working.py
   ```
   This version tracks when predictions fail, not when raw values change.

2. **Use actual TSFM models** (Chronos, Moirai):
   ```bash
   python experiments/synthetic.py --model chronos
   ```
   Real TSFMs make larger prediction errors during regime shifts.

3. **Manually lower the threshold** (not recommended):
   ```python
   # In demo.py, change:
   adapter = CredalTTA(model=model, lambda_reset=0.95)  # Too low, many false positives
   ```

### Q: What's the difference between the demo scripts?

**A:** 

| Script | What it monitors | Detection success | When to use |
|--------|------------------|-------------------|-------------|
| `demo.py` | Raw values | ❌ Fails | Understanding code structure |
| `demo_enhanced.py` | Raw values | ❌ Fails | See stronger data shifts |
| **`demo_working.py`** | **Prediction errors** | ✅ **Works!** | **See actual detection** |
| `experiments/synthetic.py` | Uses real TSFM | ✅ Works | Paper reproduction |

**Key insight:** With simple baseline models, you must monitor **prediction errors**, not raw time series values, because:
- Moving average predictions are very smooth
- Raw value changes don't translate to large prediction failures
- Prediction errors spike dramatically when models fail to adapt

---

## Understanding the Results

### Q: What does "Detected 0 regime shifts" mean?

**A:** The HCA module monitors the **contraction ratio** ρ_t. Detection occurs when:

```
ρ_t = diameter(t) / diameter(t-1) > λ_reset
```

If ρ_t never exceeds λ_reset (default: 1.2), no shift is detected. This can happen when:

1. **Model predictions are too smooth** (e.g., moving average)
2. **Regime shift is too subtle** for the model to notice
3. **Detection threshold is too high**
4. **Smoothing is too aggressive** (high `smoothing_alpha`)

### Q: How do I know if detection is working?

**A:** Check the diagnostics:

```python
preds, diagnostics = adapter.predict_sequence(data, return_diagnostics=True)

# Extract key metrics
diameters = [d['diameter'] for d in diagnostics]
ratios = [d['ratio'] for d in diagnostics]

print(f"Max diameter: {max(diameters)}")
print(f"Max ratio: {max(ratios)}")  # Should spike near shift
print(f"Threshold: {adapter.hca.lambda_reset}")
```

If `max(ratios) > λ_reset`, detection should occur.

### Q: What's a good detection threshold?

**A:** Depends on your domain:

| Domain | Recommended λ_reset | Reasoning |
|--------|---------------------|-----------|
| Synthetic data | 1.15 - 1.3 | Controlled, clean shifts |
| Financial data | 1.1 - 1.2 | Need fast response to crises |
| Sensor data | 1.2 - 1.4 | Noisy, many false positives |
| Electricity | 1.2 - 1.3 | Seasonal vs. structural changes |

**Tuning tip:** Start at 1.2, then:
- **Too many false positives?** → Increase to 1.3-1.5
- **Missing real shifts?** → Decrease to 1.1-1.15

---

## Performance Questions

### Q: How much overhead does Credal-TTA add?

**A:** Very little:
- **HCA update**: ~1.8ms per step (CPU)
- **Context management**: ~0.4ms per step
- **Total overhead**: <5% compared to TSFM inference (45-120ms)

### Q: Can I use Credal-TTA in real-time systems?

**A:** Yes, with caveats:
- **Streaming data (<100Hz)**: ✅ Excellent, negligible latency
- **High-frequency (100-1000Hz)**: ✅ Good, ~2ms per sample
- **Ultra-high-frequency (>1000Hz)**: ⚠️ May need optimizations

For real-time systems, consider:
```python
adapter = CredalTTA(
    model=model,
    K=2,  # Fewer extremes = faster
    W_max=256,  # Shorter context
    smoothing_alpha=0.3  # More smoothing = fewer triggers
)
```

### Q: Does it work with multivariate time series?

**A:** Not in the current version (v1.0.0). Univariate only.

**Workaround:** Apply Credal-TTA independently to each dimension, then aggregate detection signals (e.g., trigger if ≥2 dimensions detect shifts).

**Future work:** Multivariate credal sets are theoretically straightforward and planned for v2.0.

---

## Experiment Questions

### Q: How long do experiments take?

**A:** Approximate runtimes (Intel Xeon + NVIDIA A100):

| Experiment | Runtime | GPU Required? |
|------------|---------|---------------|
| Synthetic (CPU only) | ~5 min | No |
| Synthetic (with Chronos) | ~15 min | Yes |
| Electricity | ~30 min | Recommended |
| Finance | ~10 min | Recommended |
| Ablation | ~15 min | No |
| **Total (run_all.sh)** | **~1 hour** | Yes |

### Q: Can I run experiments without GPU?

**A:** Yes, but:
- Set `model_name="naive"` in experiment scripts
- Results won't match paper (which uses Chronos/Moirai)
- Useful for testing the framework logic

```bash
# CPU-only test
python experiments/synthetic.py --model naive --num_runs 3
```

### Q: Where are experiment results saved?

**A:** `results/` directory:

```
results/
├── synthetic/
│   ├── SinFreq_chronos_results.json
│   └── StepMean_chronos_results.json
├── electricity/
│   └── chronos_results.json
├── finance/
│   ├── sp500/chronos_results.json
│   └── bitcoin/patchtst_results.json
└── ablation/
    ├── variance_vs_credal.json
    ├── num_extremes.json
    └── threshold_sensitivity.json
```

Each JSON contains mean ± std over multiple runs.

---

## Theoretical Questions

### Q: Why credal sets instead of Bayesian ensembles?

**A:** Key advantages:

1. **Theoretical guarantees**: Geometric contraction (Theorem 3.1) provides principled detection
2. **Computational efficiency**: O(K) updates vs. O(M) ensembles (typically M=5-10)
3. **Epistemic/aleatoric separation**: Diameter captures *conflicting beliefs*, not just variance
4. **Interpretability**: Hausdorff diameter has clear geometric meaning

### Q: What happens during stable regimes?

**A:** Credal set contracts geometrically:

```
E[ρ_t] = γ < 1  (Theorem 3.1)
diameter(t) → 0  at rate O(γ^t)
```

This means:
- Diameter shrinks exponentially
- Ratio ρ_t < 1 consistently
- No false detections (in theory)

### Q: What happens during regime shifts?

**A:** Credal set expands:

```
ρ_t ≥ 1 + c·Δ/ε  (Theorem 3.2)
```

Where:
- Δ = distance between regimes
- ε = pre-shift diameter (small)
- c > 0 is learning rate

This causes ρ_t >> 1, triggering detection.

---

## Troubleshooting

### Q: Error: "ModuleNotFoundError: No module named 'credal_tta'"

**A:** Install the package:
```bash
cd credal-tta/
pip install -e .
```

### Q: Error: "chronos-forecasting not found"

**A:** Chronos is optional:

```bash
# Option 1: Install Chronos
pip install chronos-forecasting

# Option 2: Use naive baseline for testing
# (edit experiment scripts to use NaiveBaseline)
```

### Q: Validation script fails

**A:** Check dependencies:
```bash
pip install --upgrade numpy scipy matplotlib scikit-learn
python validate.py
```

If still failing, please open a GitHub issue with the full error message.

### Q: Plots not showing / saving

**A:** Matplotlib backend issue:

```bash
# Set backend explicitly
export MPLBACKEND=Agg

# Or in Python:
import matplotlib
matplotlib.use('Agg')
```

---

## Advanced Usage

### Q: Can I use custom distance metrics?

**A:** Yes! Subclass `CredalSet`:

```python
from credal_tta.core.credal_set import CredalSet

class MyCredalSet(CredalSet):
    def diameter(self):
        # Use KL divergence instead of Wasserstein
        from scipy.stats import entropy
        max_dist = 0
        for i in range(self.K):
            for j in range(i+1, self.K):
                kl = entropy(self.extremes[i].pdf_values, 
                            self.extremes[j].pdf_values)
                max_dist = max(max_dist, kl)
        return max_dist
```

### Q: Can I integrate with my own TSFM?

**A:** Absolutely! Just implement the wrapper:

```python
from credal_tta.models.wrappers import TSFMWrapper

class MyModelWrapper(TSFMWrapper):
    def __init__(self, model_path):
        self.model = load_my_model(model_path)
    
    def predict(self, context, prediction_length=1):
        # Your prediction logic
        return self.model.forecast(context)[0]

# Use it
adapter = CredalTTA(model=MyModelWrapper("path/to/model"))
```

---

## Contributing

### Q: How can I contribute?

**A:** See `CONTRIBUTING.md` for full guidelines. Quick options:

1. **Report bugs**: Open GitHub issue
2. **Add models**: Implement new TSFM wrappers
3. **Add datasets**: Extend `data_loader.py`
4. **Improve docs**: Fix typos, add examples
5. **Research**: Try new distance metrics, initialization strategies

### Q: What's the development roadmap?

**A:** Planned for future versions:

- **v1.1**: GPU acceleration for credal set operations
- **v2.0**: Multivariate time series support
- **v2.1**: Hierarchical credal sets
- **v3.0**: Integration with retrieval-augmented generation

---

## Citation

### Q: How do I cite this work?

**A:** 

```bibtex
@article{du2025credal,
  title={Breaking Context Inertia: A Neuro-Symbolic Framework for Test-Time Adaptation of Time Series Foundation Models via Credal Set Theory},
  author={Du, Yimin and Wu, Guixing},
  journal={Knowledge-Based Systems},
  year={2025}
}
```

---

**Still have questions?** 

- Check `docs/API.md` for detailed API reference
- Open a GitHub issue
- Email: sa613403@mail.ustc.edu.cn
