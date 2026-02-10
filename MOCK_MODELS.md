# ⚠️ IMPORTANT: Mock Models vs. Real TSFMs

## What You're Seeing

If you see this warning:
```
WARNING: chronos-forecasting not installed. Using mock model.
```

**Your experiments are running with a simplified moving average model, NOT the actual Chronos/Moirai foundation models.**

## Why This Matters

### The Problem

**Mock models are fundamentally limited:**

1. **Moving average is too simple**: While it exhibits context inertia (the problem we're solving), it doesn't produce the complex predictions that real TSFMs make.

2. **Detection challenges**: Credal-TTA monitors distributional changes. With moving average:
   - Raw value monitoring: ❌ Credal set diameter stays near zero
   - Prediction error monitoring: ✅ Works, but not representative of real TSFM behavior

3. **Results won't match paper**: Paper results use Chronos/Moirai, which have very different characteristics.

### Expected Behavior with Mock Models

When running experiments with mock models, you might see:

```bash
python experiments/synthetic.py --model chronos --num_runs 3

Results:
Standard:      MAE: 0.713
Credal-TTA:    MAE: 0.713  ← No improvement!
```

**This is expected!** The mock model is so simple that:
- Standard approach: Uses 512-point moving average
- Credal-TTA: Uses 10-50 point moving average (after reset)
- Both converge to similar predictions quickly

## How to Get Real Results

### Option 1: Install Real TSFM Models (Recommended)

```bash
# Install Chronos
pip install chronos-forecasting

# Or install Moirai
pip install salesforce-moirai

# Then run experiments
python experiments/synthetic.py --model chronos --num_runs 10
```

**Expected improvement:** 30-60% reduction in recovery time, 20-40% MAE reduction

### Option 2: Use Error-Based Monitoring Demo

If you can't install Chronos, use the special demo that monitors prediction errors:

```bash
python examples/demo_working.py
```

This should show regime shift detection even with the mock model.

### Option 3: Use Naive Baseline Explicitly

The experiments also support a "naive" baseline that's even simpler:

```bash
python experiments/synthetic.py --model naive --num_runs 3
```

This uses explicit `NaiveBaseline` class instead of mock, making the behavior more predictable.

## Understanding Mock Model Behavior

### What the Mock Does

```python
# In wrappers.py
def predict(self, context):
    if self.pipeline is None:  # Chronos not installed
        # Mock: 50-point moving average
        window = min(50, len(context))
        return np.mean(context[-window:])
```

**Key properties:**
- Context-dependent: ✓ (uses last 50 points)
- Exhibits inertia: ✓ (slow to adapt)
- Complex enough: ❌ (too simple for realistic testing)

### Why Results Are Identical

With mock models, all three methods (Standard, Variance-Trigger, Credal-TTA) end up doing similar things:

1. **Standard (W=512)**: Average of last 512 points
2. **Variance-Trigger**: Resets to last 10 points when variance spikes
   - But variance spike detection also fails with smooth data
3. **Credal-TTA**: Resets to last 10 points when diameter expands
   - But diameter expansion also fails with smooth predictions

**Result:** All methods converge to ~50-point moving average, giving identical predictions.

## Testing Your Installation

Run this quick test:

```bash
python test_mock.py
```

This will:
1. Check if mock model predictions actually adapt
2. Verify if Credal-TTA can detect shifts
3. Diagnose what's working and what isn't
4. Give specific recommendations

## Real-World Analogy

Think of it this way:

### Mock Model (Moving Average)
```
Task: Detect when a car changes lanes
Sensor: Measures average position over last 50 seconds
Problem: By the time you notice the average shifted, the car finished changing lanes!
```

### Real TSFM (Chronos)
```
Task: Detect when a car changes lanes  
Sensor: High-resolution camera tracking moment-to-moment position
Solution: Immediately notice the lateral movement starting
```

**Credal-TTA is designed for the second scenario** - complex models that make rich predictions. With simple moving average, it's like using a hammer designed for precision work on a crude task.

## What To Do Now

### If You Want Paper-Quality Results

**Install Chronos:**
```bash
pip install torch transformers chronos-forecasting
```

Then run:
```bash
python experiments/synthetic.py --model chronos --num_runs 10
bash run_all.sh  # Full reproduction
```

### If You Just Want to See It Work

**Use the working demo:**
```bash
python examples/demo_working.py
```

This monitors prediction errors instead of raw values, which works even with simple models.

### If You're Okay with Conceptual Understanding

Read the code and paper to understand:
- Why context inertia is a problem (conceptual)
- How credal sets track uncertainty (mathematical)
- How Credal-TTA would work with real models (extrapolate)

Then cite the paper and trust the reported results, or wait until you can access a GPU to install real TSFMs.

## Summary

| Scenario | Mock Model | Real TSFM |
|----------|------------|-----------|
| **Installation** | pip install -e . | + pip install chronos-forecasting |
| **Runtime** | ~5 min | ~30 min (GPU) |
| **Detection works?** | ❌ Not reliably | ✅ Yes |
| **Results match paper?** | ❌ No | ✅ Yes |
| **Good for** | Code testing | Benchmarking |

**Bottom line:** Mock models are for **code testing**, not **performance evaluation**. For the latter, you need real TSFMs.
