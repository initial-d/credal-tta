# Credal-TTA API Reference

## Core Classes

### `CredalTTA`

Main framework for test-time adaptation.

```python
from credal_tta import CredalTTA
from credal_tta.models.wrappers import ChronosWrapper

# Initialize
model = ChronosWrapper()
adapter = CredalTTA(
    model=model,
    K=3,                   # Number of extreme distributions
    lambda_reset=1.2,      # Detection threshold
    lambda_caution=0.95,   # Stable regime threshold
    W_max=512,             # Maximum context length
    L_min=10,              # Minimum context buffer
    sigma_noise=None,      # Auto-estimated if None
    smoothing_alpha=0.2    # EMA smoothing factor
)

# Single-step prediction
pred = adapter.predict_step(x_t)

# Batch prediction
preds = adapter.predict_sequence(time_series)

# With diagnostics
preds, diagnostics = adapter.predict_sequence(time_series, return_diagnostics=True)

# Reset for new episode
adapter.reset()
```

**Parameters:**
- `model`: TSFM wrapper implementing `predict(context)` method
- `K` (int): Number of extreme distributions in credal set (default: 3)
- `lambda_reset` (float): Detection threshold for regime shifts (default: 1.2)
- `lambda_caution` (float): Threshold for stable regime (default: 0.95)
- `W_max` (int): Maximum context window length (default: 512)
- `L_min` (int): Minimum context buffer for cold-start (default: 10)
- `sigma_noise` (float, optional): Observation noise std (auto-estimated if None)
- `smoothing_alpha` (float): EMA smoothing for contraction ratio (default: 0.2)

**Returns (from predict_step):**
- `prediction` (float): Next-step forecast
- `diagnostics` (dict, optional): Diagnostic information including:
  - `regime_shift` (bool): Whether shift detected
  - `diameter` (float): Credal set diameter
  - `ratio` (float): Contraction ratio
  - `context_length` (int): Current context size

---

### `HausdorffContextAdapter`

Epistemic uncertainty detector based on credal set theory.

```python
from credal_tta.core.hca import HausdorffContextAdapter

hca = HausdorffContextAdapter(
    K=3,
    sigma_noise=0.1,
    lambda_reset=1.2,
    lambda_caution=0.95,
    smoothing_alpha=0.2
)

# Initialize from burn-in data
hca.initialize(burn_in_data)

# Update with new observation
output = hca.update(x_obs)
# output keys: regime_shift, diameter, ratio, smoothed_ratio

# Get uncertainty metrics
metrics = hca.get_uncertainty_metrics()
# metrics keys: epistemic_uncertainty, aleatoric_uncertainty, total_uncertainty
```

---

### `ContextManager`

Dynamic context window management with reset-and-grow strategy.

```python
from credal_tta.core.context_manager import ContextManager

manager = ContextManager(W_max=512, L_min=10)

# Update context
context, info = manager.update(
    x_new=x_t,
    regime_shift=detected,
    t=current_time
)

# info keys: context_length, origin, reset_occurred, t_start
```

---

### `CredalSet`

Finitely-generated credal set operations.

```python
from credal_tta.core.credal_set import (
    GaussianDistribution,
    CredalSet,
    initialize_credal_set,
    wasserstein_2_gaussian
)

# Initialize credal set
credal_set = initialize_credal_set(burn_in_data, K=3)

# Compute diameter
diam = credal_set.diameter()

# Bayesian update
credal_set_updated = credal_set.update(x_obs, sigma_noise=0.1)

# Contraction ratio
ratio = credal_set_updated.contraction_ratio(prev_diameter=diam)
```

---

## Model Wrappers

### `ChronosWrapper`

```python
from credal_tta.models.wrappers import ChronosWrapper

model = ChronosWrapper(
    model_name="amazon/chronos-t5-base",
    device="cuda"  # or "cpu"
)

prediction = model.predict(context, prediction_length=1)
```

### `MoiraiWrapper`

```python
from credal_tta.models.wrappers import MoiraiWrapper

model = MoiraiWrapper(
    model_name="Salesforce/moirai-1.0-R-small",
    device="cuda"
)

prediction = model.predict(context)
```

### `PatchTSTWrapper`

```python
from credal_tta.models.wrappers import PatchTSTWrapper

model = PatchTSTWrapper(
    model_path="path/to/checkpoint.pt",
    seq_len=512,
    patch_len=16,
    stride=8
)

prediction = model.predict(context)
```

---

## Utilities

### Data Loading

```python
from credal_tta.utils.data_loader import (
    generate_sin_freq_shift,
    generate_step_mean_shift,
    load_uci_electricity,
    load_sp500_crisis,
    get_dataset
)

# Synthetic data
data, shift_point = generate_sin_freq_shift(T=1000, shift_point=500)

# Real-world data
electricity = load_uci_electricity(customer_id=0)
sp500, crash_point = load_sp500_crisis()
```

### Metrics

```python
from credal_tta.utils.metrics import (
    mae, rmse, mape,
    recovery_time,
    accumulated_transition_error,
    compute_all_metrics
)

# Standard metrics
error = mae(y_true, y_pred)

# Adaptation-specific metrics
rt = recovery_time(y_true, y_pred, shift_point=500)
ate = accumulated_transition_error(y_true, y_pred, shift_point=500)

# All metrics at once
metrics = compute_all_metrics(y_true, y_pred, shift_points=[500])
# Returns: {MAE, RMSE, MAPE, Avg_RT, Avg_ATE}
```

---

## Configuration Examples

### Conservative (Low False Positives)

```python
adapter = CredalTTA(
    model=model,
    K=3,
    lambda_reset=1.5,      # Higher threshold
    lambda_caution=0.9,
    smoothing_alpha=0.3,   # More smoothing
    W_max=512
)
```

### Aggressive (Fast Adaptation)

```python
adapter = CredalTTA(
    model=model,
    K=5,                   # More extremes
    lambda_reset=1.1,      # Lower threshold
    lambda_caution=0.95,
    smoothing_alpha=0.1,   # Less smoothing
    W_max=256              # Shorter max context
)
```

### Resource-Constrained

```python
adapter = CredalTTA(
    model=model,
    K=2,                   # Fewer extremes
    lambda_reset=1.2,
    W_max=128,             # Shorter context
    L_min=5
)
```

---

## Advanced Usage

### Custom Distance Metric

```python
from credal_tta.core.credal_set import CredalSet

class CustomCredalSet(CredalSet):
    def diameter(self):
        # Use custom metric (e.g., KL divergence)
        max_dist = 0.0
        for i in range(self.K):
            for j in range(i + 1, self.K):
                dist = custom_distance(self.extremes[i], self.extremes[j])
                max_dist = max(max_dist, dist)
        return max_dist
```

### Hybrid Detection Strategy

```python
# Combine credal-based and variance-based signals
hca_output = hca.update(x_t)
variance = np.var(recent_window)

regime_shift = (
    hca_output['smoothed_ratio'] > lambda_reset or
    variance > variance_threshold
)
```

### Multi-Horizon Forecasting

```python
# Predict multiple steps ahead
def predict_multi_horizon(adapter, context, horizon=10):
    predictions = []
    current_context = context.copy()
    
    for h in range(horizon):
        pred = adapter.model.predict(current_context)
        predictions.append(pred)
        current_context = np.append(current_context[1:], pred)
    
    return np.array(predictions)
```
