# Complete Installation Guide

## Quick Start (3 methods)

### Method 1: Basic Installation (No TSFM models)
```bash
cd credal-tta
pip install -e .
python validate.py  # Verify core functionality
```

**Good for:** Testing framework, understanding code
**Limitation:** Uses mock models, results won't match paper

---

### Method 2: With Chronos (Recommended)
```bash
cd credal-tta

# Install core package
pip install -e .

# Install Chronos
pip install chronos-forecasting

# Verify installation
python verify_chronos.py

# Run experiments
python experiments/synthetic.py --model chronos --num_runs 3
```

**Good for:** Reproducing paper results, real benchmarks
**Requires:** ~2GB disk space, GPU recommended but not required

---

### Method 3: Full Installation (All models)
```bash
cd credal-tta

# Install with all optional dependencies
pip install -e ".[full]"

# This includes:
# - chronos-forecasting
# - salesforce-moirai (if available)
# - All transformer dependencies

python verify_chronos.py
```

**Good for:** Complete reproduction of all experiments
**Requires:** ~5GB disk space, GPU highly recommended

---

## Step-by-Step Installation

### 1. Prerequisites

**System requirements:**
- Python 3.8 or higher
- 8GB RAM minimum (16GB+ recommended)
- GPU with 8GB+ VRAM (optional but recommended for speed)

**Check your Python version:**
```bash
python --version  # Should be 3.8+
```

### 2. Create Virtual Environment (Recommended)

```bash
# Create environment
python -m venv credal-env

# Activate (Linux/Mac)
source credal-env/bin/activate

# Activate (Windows)
credal-env\Scripts\activate
```

### 3. Install Core Package

```bash
cd credal-tta
pip install -e .
```

**What this installs:**
- numpy, scipy, matplotlib
- scikit-learn, pandas
- Core Credal-TTA framework

### 4. Install Chronos (For Real Experiments)

```bash
pip install chronos-forecasting
```

**First-time usage note:** Chronos will download model weights from HuggingFace:
- `chronos-t5-tiny`: ~6MB
- `chronos-t5-base`: ~200MB (default)
- `chronos-t5-large`: ~800MB

**Verify installation:**
```bash
python verify_chronos.py
```

Expected output:
```
[1/4] Testing Chronos import...
  ✓ chronos-forecasting installed successfully

[2/4] Testing PyTorch...
  ✓ PyTorch 2.x.x installed
  ✓ CUDA available: True

[3/4] Loading Chronos model...
  ✓ Model loaded successfully

[4/4] Testing prediction...
  ✓ Prediction successful

✓ ALL TESTS PASSED!
```

### 5. (Optional) Install Moirai

For Table 3 experiments with Moirai:

```bash
pip install salesforce-moirai
```

**Note:** Moirai installation can be tricky. If it fails, you can still:
- Use Chronos for most experiments
- Skip Moirai-specific rows in Table 3

---

## Troubleshooting

### Issue 1: "No module named 'chronos'"

**Solution:**
```bash
pip install chronos-forecasting

# If that fails, try:
pip install torch transformers
pip install chronos-forecasting
```

### Issue 2: "CUDA out of memory"

**Solution A:** Use CPU instead
```bash
# Edit experiments/synthetic.py line 95:
device = "cpu"  # Instead of "cuda"
```

**Solution B:** Use smaller model
```bash
# In credal_tta/models/wrappers.py, change:
model_name = "amazon/chronos-t5-tiny"  # Instead of base
```

**Solution C:** Reduce batch size
```bash
# Run fewer experiments at once
python experiments/synthetic.py --num_runs 1
```

### Issue 3: "Model download hangs"

**Problem:** Firewall blocking HuggingFace

**Solution:**
```bash
# Set HuggingFace cache directory
export HF_HOME=/path/to/cache

# Or download manually:
# Visit: https://huggingface.co/amazon/chronos-t5-base
# Download model files
# Place in ~/.cache/huggingface/
```

### Issue 4: ImportError for transformers

**Solution:**
```bash
pip install transformers>=4.30.0
```

### Issue 5: "RuntimeError: No CUDA GPUs available"

**This is fine!** Chronos works on CPU, just slower.

**To use CPU explicitly:**
```python
# In wrappers.py
device = "cpu"
```

---

## Verifying Your Installation

### Quick Test
```bash
python validate.py
```

Should show:
```
✓ All tests passed! The installation is working correctly.
```

### Chronos-Specific Test
```bash
python verify_chronos.py
```

### Full End-to-End Test
```bash
python examples/demo_working.py
```

If Chronos is installed, this will automatically use it instead of mock.

---

## What Gets Installed

### Core Package (`pip install -e .`)
```
numpy>=1.24.0
scipy>=1.10.0
matplotlib>=3.7.0
scikit-learn>=1.3.0
pandas>=2.0.0
```

### With Chronos (`pip install chronos-forecasting`)
```
+ torch>=2.0.0
+ transformers>=4.30.0
+ chronos-forecasting
```

### Full Installation (`pip install -e ".[full]"`)
```
+ All above
+ salesforce-moirai (if available)
+ Additional transformer dependencies
```

---

## Installation Checklist

- [ ] Python 3.8+ installed
- [ ] Virtual environment created and activated
- [ ] Core package installed (`pip install -e .`)
- [ ] `python validate.py` passes
- [ ] (Optional) Chronos installed
- [ ] (Optional) `python verify_chronos.py` passes
- [ ] (Optional) GPU with CUDA available

---

## Next Steps

After successful installation:

### 1. Run Quick Demo
```bash
# With Chronos installed:
python examples/demo_working.py

# Should now show real TSFM predictions!
```

### 2. Run Small Experiment
```bash
python experiments/synthetic.py --model chronos --num_runs 3
```

Expected runtime:
- With GPU: ~5 minutes
- With CPU: ~15 minutes

### 3. Reproduce Paper Results
```bash
# Full reproduction (~1 hour with GPU)
bash run_all.sh
```

---

## GPU vs CPU Performance

| Task | CPU Time | GPU Time | Speedup |
|------|----------|----------|---------|
| Single prediction | 100ms | 10ms | 10x |
| Synthetic experiment (1 run) | 5 min | 30 sec | 10x |
| Full experiments | 8 hours | 1 hour | 8x |

**Recommendation:**
- **Have GPU?** Use it! Set `device="cuda"`
- **CPU only?** Still works, just takes longer
- **Limited time?** Run smaller experiments (--num_runs 3 instead of 10)

---

## Common Installation Paths

### Successful Installation
```
1. pip install -e .                    ✓
2. pip install chronos-forecasting     ✓
3. python verify_chronos.py            ✓ ALL TESTS PASSED
4. python experiments/synthetic.py     ✓ Shows improvements
```

### Partial Installation (OK for learning)
```
1. pip install -e .                    ✓
2. Skip Chronos installation           ⚠
3. python examples/demo_working.py     ✓ Works with mock
4. Read documentation                  ✓ Understand concepts
```

### Minimal Installation (Code review only)
```
1. Extract tarball                     ✓
2. Read code                           ✓
3. No pip install needed               ✓
```

---

## Getting Help

If installation fails:

1. **Check Python version:** Must be 3.8+
2. **Update pip:** `pip install --upgrade pip`
3. **Try clean install:**
   ```bash
   pip uninstall chronos-forecasting
   pip cache purge
   pip install chronos-forecasting
   ```
4. **Check disk space:** Need ~2GB free
5. **Check error logs:** Copy full error message
6. **Open GitHub issue** with:
   - Python version
   - OS (Linux/Mac/Windows)
   - Full error message
   - Output of `pip list`

---

## Summary

**Minimum (works but limited):**
```bash
pip install -e .
```

**Recommended (paper results):**
```bash
pip install -e .
pip install chronos-forecasting
python verify_chronos.py
```

**Complete (all experiments):**
```bash
pip install -e ".[full]"
python verify_chronos.py
bash run_all.sh
```

Choose based on your goals and available resources!
