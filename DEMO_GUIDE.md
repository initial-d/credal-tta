# Demo Guide: Which Script Should I Run?

## TL;DR

**Want to see Credal-TTA actually work?**

```bash
# Recommended: Working demo that monitors prediction errors
python examples/demo_working.py
```

---

## Understanding the Demo Scripts

### 🟢 `demo_working.py` - **RECOMMENDED**

**What it does:** Monitors **prediction errors** to detect when the model fails

**Why it works:** 
- Prediction errors spike dramatically during regime shifts
- Even simple models like moving average fail predictably at regime changes
- Credal set diameter tracks these error patterns

**Expected output:**
```
✓ Detected 1-3 regime shifts at: [502, ...]
Recovery time reduced by 30-50%
```

**Run it:**
```bash
python examples/demo_working.py
```

---

### 🟡 `demo_enhanced.py` - **Educational**

**What it does:** Monitors **raw time series values** with strong regime shifts

**Why it struggles:**
- Moving average predictions are too smooth
- Raw value monitoring works better with complex models
- Good for understanding the data generation process

**Expected output:**
```
⚠ Detected 0 regime shifts
Max diameter: 0.0051  ← Nearly zero!
```

**Use it for:** Understanding what strong regime shifts look like in the data

---

### 🟡 `demo.py` - **Legacy**

**What it does:** Original demo with moderate regime shifts

**Status:** Same issues as `demo_enhanced.py`, kept for backward compatibility

**Expected output:** No detections with simple baseline

---

## Why the Original Demos Don't Work

### The Problem

When HCA monitors **raw time series values** with a **moving average model**:

1. **Data changes**: `x_t` jumps from mean=0 to mean=5 ✓
2. **Predictions change**: But predictions are *smoothed averages*, change gradually
3. **Credal set**: Tracks the *smoothed predictions*, not the raw jumps
4. **Result**: Diameter barely changes → No detection

### The Solution

Monitor **prediction errors** instead:

1. **Before shift**: Model predicts well, errors ~1.0
2. **At shift**: Model uses old context, errors spike to ~5.0
3. **Credal set**: Tracks these *error spikes* 
4. **Result**: Diameter explodes → Detection!

### Visualization

```
Raw values approach (demo_enhanced.py):
Time:  0    100   200   300   400   500   600   700
Data:  [0.1  0.2  -0.1  0.3  0.0  |5.1  5.2  4.9  5.3]
Pred:  [0.1  0.15  0.0  0.2  0.1  |0.8  1.5  2.3  3.1]  ← Smooth!
HCA:   [0.1  0.15  0.0  0.2  0.1  |0.8  1.5  2.3  3.1]  ← Monitors smooth predictions
Diam:   ─────────────────────────────────────────────  ← No spike!

Error-based approach (demo_working.py):
Time:  0    100   200   300   400   500   600   700
Data:  [0.1  0.2  -0.1  0.3  0.0  |5.1  5.2  4.9  5.3]
Pred:  [0.1  0.15  0.0  0.2  0.1  |0.8  1.5  2.3  3.1]
Error: [0.1  0.05  0.1  0.1  0.1  |4.3  3.7  2.6  2.2]  ← Spike!
HCA:   [0.1  0.05  0.1  0.1  0.1  |4.3  3.7  2.6  2.2]  ← Monitors errors
Diam:   ─────────────────────────┐                     
                                 └───────────────────  ← BIG spike!
```

---

## For Paper Reproduction

**Don't use the demos!** They're educational tools, not benchmarks.

For actual performance numbers from the paper:

```bash
# Table 2: Synthetic benchmarks
python experiments/synthetic.py --model chronos --num_runs 10

# Table 3: UCI Electricity
python experiments/electricity.py --model chronos --num_series 20

# Table 4: Financial crises
python experiments/finance.py --dataset sp500 --model chronos

# All experiments (~1 hour)
bash run_all.sh
```

These use **real TSFM models** (Chronos, Moirai) which:
- Make complex, informative predictions
- Have larger errors during regime shifts
- Work correctly with raw value monitoring

---

## Quick Comparison

| Metric | demo.py | demo_enhanced.py | **demo_working.py** | experiments/*.py |
|--------|---------|------------------|---------------------|------------------|
| **Detection works?** | ❌ | ❌ | ✅ | ✅ |
| **Model** | Moving avg | Moving avg | Moving avg | Chronos/Moirai |
| **Monitors** | Raw values | Raw values | **Errors** | Raw values |
| **Runtime** | 10 sec | 10 sec | 10 sec | 5-60 min |
| **Purpose** | Education | Education | **Demo** | **Benchmarks** |
| **Paper results?** | No | No | No | **Yes** |

---

## Still Having Issues?

### Expected behavior after fixes:

**`demo_working.py` should show:**
```
Detected 1+ regime shifts at: [502, ...]
Recovery time reduced by 30-50%
```

**If it still shows 0 detections:**

1. Check the diagnostic output:
   ```
   Max diameter: ???      ← Should be >1.0
   Max ratio: ???         ← Should be >1.2
   ```

2. If diameter is still tiny (<0.01):
   - The burn-in period (20 samples) may still be too long
   - Try editing `credal_tta/core/hca.py` line 93: change 20 to 10

3. If diameter is large but ratio <1.2:
   - Lower the threshold manually in the script:
     ```python
     adapter = ErrorBasedCredalTTA(lambda_reset=1.1)
     ```

4. Open a GitHub issue with:
   - Full diagnostic output
   - Your Python version
   - NumPy/SciPy versions

---

## Summary

- **Quick demo:** `python examples/demo_working.py`
- **Understand why:** Read this guide
- **Real benchmarks:** `python experiments/synthetic.py --model chronos`
- **Paper reproduction:** `bash run_all.sh`

The key insight: **simple models need error-based monitoring, complex models work with value-based monitoring**.
