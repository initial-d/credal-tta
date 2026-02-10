#!/usr/bin/env python
"""
Verify Chronos Installation
"""

print("=" * 70)
print("Chronos Installation Verification")
print("=" * 70)
print()

# Test 1: Import chronos
print("[1/4] Testing Chronos import...")
try:
    from chronos import ChronosPipeline
    print("  ✓ chronos-forecasting installed successfully")
except ImportError as e:
    print(f"  ✗ Failed to import chronos: {e}")
    print("\n  Install with: pip install chronos-forecasting")
    exit(1)

# Test 2: Import torch
print("\n[2/4] Testing PyTorch...")
try:
    import torch
    print(f"  ✓ PyTorch {torch.__version__} installed")
    print(f"  ✓ CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  ✓ CUDA device: {torch.cuda.get_device_name(0)}")
except ImportError as e:
    print(f"  ✗ PyTorch not found: {e}")
    exit(1)

# Test 3: Load tiny model for testing
print("\n[3/4] Loading Chronos model (this may take a minute)...")
try:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Using device: {device}")
    
    # Load smallest model for quick test
    pipeline = ChronosPipeline.from_pretrained(
        "amazon/chronos-t5-tiny",  # Smallest model (~6MB)
        device_map=device,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32
    )
    print("  ✓ Model loaded successfully")
except Exception as e:
    print(f"  ✗ Failed to load model: {e}")
    print("\n  This might be due to:")
    print("    - Network issues (model downloads from HuggingFace)")
    print("    - Insufficient memory")
    print("    - Missing transformers library")
    exit(1)

# Test 4: Make a prediction
print("\n[4/4] Testing prediction...")
try:
    import numpy as np
    
    # Create simple test data
    context = torch.tensor(np.random.randn(1, 100), dtype=torch.float32)
    
    # Generate forecast
    forecast = pipeline.predict(context, prediction_length=1)
    
    print(f"  ✓ Prediction successful")
    print(f"  ✓ Output shape: {forecast.shape}")
    print(f"  ✓ Forecast value: {forecast.numpy()[0, 0, 0]:.4f}")
    
except Exception as e:
    print(f"  ✗ Prediction failed: {e}")
    exit(1)

# Summary
print("\n" + "=" * 70)
print("✓ ALL TESTS PASSED!")
print("=" * 70)
print()
print("Chronos is ready to use. You can now run:")
print("  python experiments/synthetic.py --model chronos --num_runs 3")
print()
print("Available Chronos models (from smallest to largest):")
print("  - amazon/chronos-t5-tiny   (~6MB, fastest)")
print("  - amazon/chronos-t5-mini   (~20MB)")
print("  - amazon/chronos-t5-small  (~46MB)")
print("  - amazon/chronos-t5-base   (~200MB, used in paper)")
print("  - amazon/chronos-t5-large  (~800MB)")
print()
print("The experiments use 'chronos-t5-base' by default.")
