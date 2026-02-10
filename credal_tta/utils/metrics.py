"""
Evaluation Metrics - Section 5.4 in paper
"""

import numpy as np
from typing import List, Tuple, Dict


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error"""
    return np.mean(np.abs(y_true - y_pred))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error"""
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def mape(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-8) -> float:
    """Mean Absolute Percentage Error"""
    mask = np.abs(y_true) > epsilon
    return 100 * np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask]))


def recovery_time(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    shift_point: int,
    baseline_window: int = 100,
    recovery_window: int = 10,
    threshold_multiplier: float = 1.5
) -> int:
    """
    Recovery Time (RT) - Number of steps to adapt after regime shift
    
    Defined in Section 5.4:
    RT(τ) = min{t - τ : (1/10)Σ|ŷ_s - y_s| ≤ 1.5 × baseline_error}
    
    Args:
        y_true: Ground truth values
        y_pred: Predictions
        shift_point: Regime shift time τ
        baseline_window: Window for computing pre-shift error
        recovery_window: Window for computing post-shift error
        threshold_multiplier: Recovery threshold (1.5 in paper)
        
    Returns:
        Number of steps until recovery
    """
    # Compute baseline error (pre-shift)
    pre_shift_start = max(0, shift_point - baseline_window)
    baseline_error = mae(
        y_true[pre_shift_start:shift_point],
        y_pred[pre_shift_start:shift_point]
    )
    
    recovery_threshold = threshold_multiplier * baseline_error
    
    # Search for recovery point
    for t in range(shift_point + recovery_window, len(y_true)):
        window_error = mae(
            y_true[t - recovery_window:t],
            y_pred[t - recovery_window:t]
        )
        if window_error <= recovery_threshold:
            return t - shift_point
    
    # No recovery detected
    return len(y_true) - shift_point


def accumulated_transition_error(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    shift_point: int,
    transition_window: int = 100
) -> float:
    """
    Accumulated Transition Error (ATE) - Cumulative error during recovery
    
    Defined in Section 5.4:
    ATE(τ, T_w) = Σ_{t=τ+1}^{τ+T_w} |ŷ_t - y_t|
    
    Args:
        y_true: Ground truth values
        y_pred: Predictions
        shift_point: Regime shift time τ
        transition_window: Recovery window length T_w (100 in paper)
        
    Returns:
        Cumulative absolute error during transition
    """
    end_point = min(shift_point + transition_window, len(y_true))
    transition_errors = np.abs(y_true[shift_point:end_point] - y_pred[shift_point:end_point])
    
    return np.sum(transition_errors)


def compute_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    shift_points: List[int] = None
) -> Dict:
    """
    Compute all standard and adaptation-specific metrics
    
    Args:
        y_true: Ground truth
        y_pred: Predictions
        shift_points: List of regime shift times (for RT and ATE)
        
    Returns:
        Dictionary of metrics
    """
    metrics = {
        'MAE': mae(y_true, y_pred),
        'RMSE': rmse(y_true, y_pred),
        'MAPE': mape(y_true, y_pred)
    }
    
    if shift_points:
        recovery_times = []
        ates = []
        
        for tau in shift_points:
            if tau < len(y_true) - 20:  # Ensure sufficient post-shift data
                rt = recovery_time(y_true, y_pred, tau)
                ate = accumulated_transition_error(y_true, y_pred, tau)
                recovery_times.append(rt)
                ates.append(ate)
        
        if recovery_times:
            metrics['Avg_RT'] = np.mean(recovery_times)
            metrics['Avg_ATE'] = np.mean(ates)
    
    return metrics


def detect_regime_shifts(
    diagnostics: List[Dict],
    threshold: float = 1.2
) -> List[int]:
    """
    Extract regime shift times from Credal-TTA diagnostics
    
    Args:
        diagnostics: List of diagnostic dicts from CredalTTA
        threshold: Detection threshold (lambda_reset)
        
    Returns:
        List of time indices where shifts were detected
    """
    shift_points = []
    for i, diag in enumerate(diagnostics):
        if diag.get('regime_shift', False):
            shift_points.append(i)
    
    return shift_points


def compare_methods(
    y_true: np.ndarray,
    predictions_dict: Dict[str, np.ndarray],
    shift_points: List[int] = None
) -> Dict:
    """
    Compare multiple methods on same dataset
    
    Args:
        y_true: Ground truth
        predictions_dict: {method_name: predictions} mapping
        shift_points: Known regime shift times
        
    Returns:
        DataFrame-like dict with metrics for each method
    """
    results = {}
    
    for method_name, y_pred in predictions_dict.items():
        results[method_name] = compute_all_metrics(y_true, y_pred, shift_points)
    
    return results
