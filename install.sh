#!/bin/bash

# Credal-TTA Automatic Installation Script
# This script installs Credal-TTA with Chronos TSFM

set -e  # Exit on error

echo "======================================================================"
echo "Credal-TTA Installation Script"
echo "======================================================================"
echo ""

# Check Python version
echo "[1/6] Checking Python version..."
python_version=$(python --version 2>&1 | awk '{print $2}')
echo "  Found Python $python_version"

# Basic version check (need 3.8+)
major=$(echo $python_version | cut -d. -f1)
minor=$(echo $python_version | cut -d. -f2)

if [ "$major" -lt 3 ] || ([ "$major" -eq 3 ] && [ "$minor" -lt 8 ]); then
    echo "  ✗ Error: Python 3.8+ required, found $python_version"
    exit 1
fi
echo "  ✓ Python version OK"
echo ""

# Install core package
echo "[2/6] Installing core Credal-TTA package..."
pip install -e . > /dev/null 2>&1
if [ $? -eq 0 ]; then
    echo "  ✓ Core package installed"
else
    echo "  ✗ Failed to install core package"
    exit 1
fi
echo ""

# Ask about Chronos installation
echo "[3/6] Chronos TSFM installation"
read -p "  Install Chronos? (Recommended, ~200MB download) [Y/n]: " install_chronos
install_chronos=${install_chronos:-Y}

if [[ $install_chronos =~ ^[Yy]$ ]]; then
    echo "  Installing Chronos (this may take a few minutes)..."
    pip install chronos-forecasting > /dev/null 2>&1
    if [ $? -eq 0 ]; then
        echo "  ✓ Chronos installed successfully"
        chronos_installed=true
    else
        echo "  ✗ Chronos installation failed"
        echo "  Continuing with mock models..."
        chronos_installed=false
    fi
else
    echo "  Skipping Chronos installation (will use mock models)"
    chronos_installed=false
fi
echo ""

# Validate installation
echo "[4/6] Validating installation..."
python validate.py > /dev/null 2>&1
if [ $? -eq 0 ]; then
    echo "  ✓ Core validation passed"
else
    echo "  ✗ Validation failed"
    echo "  Run 'python validate.py' for details"
fi
echo ""

# Verify Chronos (if installed)
if [ "$chronos_installed" = true ]; then
    echo "[5/6] Verifying Chronos installation..."
    python verify_chronos.py > /dev/null 2>&1
    if [ $? -eq 0 ]; then
        echo "  ✓ Chronos working correctly"
    else
        echo "  ⚠ Chronos installed but verification failed"
        echo "  Run 'python verify_chronos.py' for details"
    fi
else
    echo "[5/6] Skipping Chronos verification (not installed)"
fi
echo ""

# Optional: Moirai
echo "[6/6] Optional: Moirai TSFM"
read -p "  Install Moirai? (Optional, ~500MB) [y/N]: " install_moirai
install_moirai=${install_moirai:-N}

if [[ $install_moirai =~ ^[Yy]$ ]]; then
    echo "  Installing Moirai..."
    pip install salesforce-moirai > /dev/null 2>&1
    if [ $? -eq 0 ]; then
        echo "  ✓ Moirai installed"
    else
        echo "  ✗ Moirai installation failed (this is optional)"
    fi
else
    echo "  Skipping Moirai installation"
fi
echo ""

# Summary
echo "======================================================================"
echo "Installation Complete!"
echo "======================================================================"
echo ""
echo "What was installed:"
echo "  ✓ Core Credal-TTA framework"
if [ "$chronos_installed" = true ]; then
    echo "  ✓ Chronos TSFM (for real experiments)"
else
    echo "  ⚠ Chronos NOT installed (will use mock models)"
fi
echo ""

echo "Next steps:"
echo ""

if [ "$chronos_installed" = true ]; then
    echo "  1. Run demo with real TSFM:"
    echo "     python examples/demo_working.py"
    echo ""
    echo "  2. Run small experiment:"
    echo "     python experiments/synthetic.py --model chronos --num_runs 3"
    echo ""
    echo "  3. Reproduce paper results:"
    echo "     bash run_all.sh"
else
    echo "  1. Run demo with mock model:"
    echo "     python examples/demo_working.py"
    echo ""
    echo "  2. To install Chronos later:"
    echo "     pip install chronos-forecasting"
    echo "     python verify_chronos.py"
    echo ""
    echo "  3. For paper-quality results, Chronos is required."
fi

echo ""
echo "Documentation:"
echo "  - INSTALL.md      : Detailed installation guide"
echo "  - QUICKSTART.md   : Quick start tutorial"
echo "  - DEMO_GUIDE.md   : Choosing the right demo"
echo "  - FAQ.md          : Common questions"
echo ""
