#!/bin/bash

# Credal-TTA - Run All Experiments
# This script reproduces all results from the paper

echo "======================================"
echo "Credal-TTA: Reproducing Paper Results"
echo "======================================"
echo ""

# Create results directory
mkdir -p results/{synthetic,electricity,finance,ablation}

# ===== Synthetic Benchmarks (Table 2) =====
echo "[1/4] Running Synthetic Benchmarks..."
echo "--------------------------------------"

python experiments/synthetic.py --dataset SinFreq --model chronos --num_runs 10
python experiments/synthetic.py --dataset StepMean --model chronos --num_runs 10

echo "✓ Synthetic benchmarks complete"
echo ""

# ===== UCI Electricity (Table 3) =====
echo "[2/4] Running UCI Electricity Experiments..."
echo "--------------------------------------"

python experiments/electricity.py --model chronos --num_series 20
python experiments/electricity.py --model moirai --num_series 20

echo "✓ Electricity experiments complete"
echo ""

# ===== Financial Crisis (Table 4) =====
echo "[3/4] Running Financial Crisis Experiments..."
echo "--------------------------------------"

python experiments/finance.py --dataset sp500 --model chronos --num_runs 5
python experiments/finance.py --dataset bitcoin --model patchtst --num_runs 5

echo "✓ Financial crisis experiments complete"
echo ""

# ===== Ablation Studies (Tables 5-7) =====
echo "[4/4] Running Ablation Studies..."
echo "--------------------------------------"

python experiments/ablation.py --study variance_vs_credal --num_series 5
python experiments/ablation.py --study num_extremes
python experiments/ablation.py --study threshold_sensitivity

echo "✓ Ablation studies complete"
echo ""

# ===== Summary =====
echo "======================================"
echo "All experiments completed!"
echo "======================================"
echo ""
echo "Results saved to:"
echo "  - results/synthetic/"
echo "  - results/electricity/"
echo "  - results/finance/"
echo "  - results/ablation/"
echo ""
echo "To visualize results, see examples/quickstart.ipynb"
