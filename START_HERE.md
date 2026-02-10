# 🚀 Quick Start: From Installation to Results in 10 Minutes

## Option A: Automatic Installation (Recommended)

```bash
# 1. Extract and navigate
tar -xzf credal-tta-v2.tar.gz
cd credal-tta

# 2. Run installation script
bash install.sh

# 3. Run demo
python examples/demo_working.py
```

That's it! The script will guide you through everything.

---

## Option B: Manual Installation

### Step 1: Install Core Package (30 seconds)
```bash
cd credal-tta
pip install -e .
```

### Step 2: Install Chronos (2-5 minutes, optional but recommended)
```bash
pip install chronos-forecasting
```

### Step 3: Verify Installation (30 seconds)
```bash
python verify_chronos.py
```

Expected output:
```
✓ ALL TESTS PASSED!
Chronos is ready to use.
```

### Step 4: Run Your First Experiment (2 minutes)
```bash
python experiments/synthetic.py --model chronos --num_runs 3
```

---

## What You Should See

### With Chronos Installed ✅

**Demo output:**
```
✓ Detected 2-3 regime shifts at: [502, 518, ...]
Recovery time reduced by 45%
MAE improvement: 35%
```

**Experiment output:**
```
Standard:      MAE: 0.420  RT: 78
Credal-TTA:    MAE: 0.280  RT: 37  ← Clear improvement!
```

### Without Chronos (Mock Models) ⚠️

**Demo output:**
```
⚠ Detected 0 regime shifts
Max diameter: 0.0051  ← Very small!
```

**Experiment output:**
```
Standard:      MAE: 0.713  RT: 499
Credal-TTA:    MAE: 0.713  RT: 499  ← No difference
```

**Why?** See `MOCK_MODELS.md` for detailed explanation.

---

## Troubleshooting in 60 Seconds

### Problem 1: "ModuleNotFoundError: No module named 'credal_tta'"
```bash
# Solution:
cd credal-tta
pip install -e .
```

### Problem 2: "WARNING: chronos-forecasting not installed"
```bash
# Solution:
pip install chronos-forecasting

# Verify:
python verify_chronos.py
```

### Problem 3: Experiments show no improvement
```bash
# Check if using mock:
python experiments/synthetic.py --model chronos --num_runs 1

# If you see "WARNING: Using mock model":
pip install chronos-forecasting

# If Chronos is installed but results identical:
python test_mock.py  # Diagnostic tool
```

### Problem 4: CUDA out of memory
```bash
# Solution: Use CPU
# Edit credal_tta/models/wrappers.py line 42:
device = "cpu"  # Instead of "cuda"
```

---

## Quick Comparison: What Each Script Does

| Script | Model | Runtime | Detection | Purpose |
|--------|-------|---------|-----------|---------|
| `demo.py` | Mock | 10s | ❌ | Understanding code |
| `demo_enhanced.py` | Mock | 10s | ❌ | See data patterns |
| **`demo_working.py`** | Mock | 10s | ✅ | **Quick demo** |
| `experiments/synthetic.py` | Chronos | 5min | ✅ | **Paper results** |

**Recommendation:** Start with `demo_working.py`, then move to real experiments.

---

## 10-Minute Workflow

```bash
# Minute 1-2: Installation
tar -xzf credal-tta-v2.tar.gz
cd credal-tta
pip install -e .

# Minute 3-7: Install Chronos
pip install chronos-forecasting
python verify_chronos.py

# Minute 8-9: Quick demo
python examples/demo_working.py

# Minute 10: First real experiment
python experiments/synthetic.py --model chronos --num_runs 1
```

---

## What to Expect: Results Summary

### Synthetic Benchmarks (Table 2)

**SinFreq Dataset:**
| Method | MAE | Recovery Time | Improvement |
|--------|-----|---------------|-------------|
| Standard | 0.42 | 78 steps | baseline |
| **Credal-TTA** | **0.28** | **37 steps** | **52% faster** |

**StepMean Dataset:**
| Method | MAE | Recovery Time | ATE |
|--------|-----|---------------|-----|
| Standard | 1.52 | 95 steps | 128.3 |
| **Credal-TTA** | **1.12** | **42 steps** | **51.8** |

### Runtime Expectations

| Experiment | With GPU | With CPU | Runs |
|------------|----------|----------|------|
| Single run | 30 sec | 2 min | 1 |
| Paper (10 runs) | 5 min | 20 min | 10 |
| All experiments | 1 hour | 8 hours | Full |

---

## Decision Tree: Which Path to Take?

```
Do you have GPU?
├── Yes
│   ├── Install Chronos → Run full experiments → Get paper results
│   └── Estimated time: 1-2 hours total
└── No
    ├── Do you need paper-quality results?
    │   ├── Yes → Install Chronos anyway (slower but works)
    │   │   └── Estimated time: 6-8 hours
    │   └── No → Use demo_working.py for quick demo
    │       └── Estimated time: 5 minutes
    └── Just want to understand the code?
        └── Read documentation + run validate.py
            └── Estimated time: 30 minutes
```

---

## Recommended Learning Path

### Path 1: Quick Learner (30 min)
1. `pip install -e .`
2. `python examples/demo_working.py`
3. Read `DEMO_GUIDE.md`
4. Understand the concept

### Path 2: Thorough Evaluator (2 hours)
1. `bash install.sh` (installs Chronos)
2. `python verify_chronos.py`
3. `python experiments/synthetic.py --model chronos --num_runs 3`
4. Compare results with paper

### Path 3: Full Reproducer (4 hours)
1. Complete installation with Chronos
2. `bash run_all.sh`
3. Reproduce all tables
4. Generate all figures

---

## Getting Help

**Installation issues?**
- See `INSTALL.md` for detailed guide
- Run `python validate.py` for diagnostics

**Demo not detecting shifts?**
- See `DEMO_GUIDE.md` for explanation
- Try `python demo_working.py` (monitors errors)
- Check `MOCK_MODELS.md` if using mock

**Experiment results wrong?**
- Verify Chronos installed: `python verify_chronos.py`
- Check for "WARNING: Using mock model" in output
- Run `python test_mock.py` for diagnostics

**Still stuck?**
- Check `FAQ.md` (50+ questions answered)
- Open GitHub issue
- Email: sa613403@mail.ustc.edu.cn

---

## Success Checklist

After installation, you should be able to:

- [ ] `python validate.py` passes
- [ ] `python verify_chronos.py` passes (if Chronos installed)
- [ ] `python examples/demo_working.py` shows improvements
- [ ] `python experiments/synthetic.py --model chronos --num_runs 1` shows different results for Standard vs Credal-TTA
- [ ] No "WARNING: Using mock model" messages (if Chronos installed)

If all checked, you're ready to reproduce the paper!

---

## Summary

**Fastest path to results:**
```bash
bash install.sh                                    # Answer Y to Chronos
python experiments/synthetic.py --model chronos    # 5-10 min
```

**Most thorough validation:**
```bash
bash install.sh
python verify_chronos.py
bash run_all.sh    # ~1 hour, reproduces entire paper
```

**Just exploring:**
```bash
pip install -e .
python examples/demo_working.py
cat DEMO_GUIDE.md
```

Choose based on your goals and time available!
