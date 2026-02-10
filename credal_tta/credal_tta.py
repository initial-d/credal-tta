"""
Credal-TTA Framework - Main Entry Point
Integrates HCA, Context Manager, and TSFM Interface
"""

import numpy as np
from typing import Optional, List, Dict
from .core.hca import HausdorffContextAdapter
from .core.context_manager import ContextManager


class CredalTTA:
    """
    Credal Test-Time Adaptation Framework
    
    Usage:
        model = ChronosWrapper(...)
        adapter = CredalTTA(model=model)
        
        for t, x_t in enumerate(time_series):
            prediction = adapter.predict_step(x_t)
    """
    
    def __init__(
        self,
        model,  # TSFM wrapper (Chronos/Moirai/PatchTST)
        K: int = 3,
        lambda_reset: float = 1.2,
        lambda_caution: float = 0.95,
        W_max: int = 512,
        L_min: int = 10,
        sigma_noise: Optional[float] = None,
        smoothing_alpha: float = 0.2
    ):
        """
        Args:
            model: TSFM wrapper implementing predict(context) method
            K: Number of extreme distributions in credal set
            lambda_reset: Detection threshold for regime shifts
            lambda_caution: Threshold for stable regime
            W_max: Maximum context length
            L_min: Minimum context buffer
            sigma_noise: Observation noise (auto-estimated if None)
            smoothing_alpha: EMA smoothing for contraction ratio
        """
        self.model = model
        
        # Initialize modules
        self.hca = HausdorffContextAdapter(
            K=K,
            sigma_noise=sigma_noise,
            lambda_reset=lambda_reset,
            lambda_caution=lambda_caution,
            smoothing_alpha=smoothing_alpha
        )
        
        self.context_manager = ContextManager(
            W_max=W_max,
            L_min=L_min
        )
        
        # Tracking
        self.t: int = 0
        self.predictions: List[float] = []
        self.diagnostics: List[Dict] = []
        
    def predict_step(self, x_t: float, return_diagnostics: bool = False):
        """
        Single-step prediction with adaptive context (Algorithm 1)
        
        Args:
            x_t: Current observation
            return_diagnostics: Whether to return diagnostic info
            
        Returns:
            prediction: Next-step forecast
            diagnostics (optional): Dict with HCA and context info
        """
        # HCA update and regime detection
        hca_output = self.hca.update(x_t)
        regime_shift = hca_output['regime_shift']
        
        # Context management
        context, context_info = self.context_manager.update(
            x_new=x_t,
            regime_shift=regime_shift,
            t=self.t
        )
        
        # Optional: Reset credal set after shift
        if regime_shift and len(context) >= 10:
            self.hca.reset_credal_set(context[-20:])
        
        # TSFM prediction
        if len(context) >= self.context_manager.L_min:
            prediction = self.model.predict(context)
        else:
            # Fallback: naive forecast for insufficient context
            prediction = x_t if len(context) == 0 else context[-1]
        
        self.predictions.append(prediction)
        
        # Increment time
        self.t += 1
        
        # Diagnostics
        diagnostics = {
            't': self.t,
            **hca_output,
            **context_info,
            'prediction': prediction
        }
        self.diagnostics.append(diagnostics)
        
        if return_diagnostics:
            return prediction, diagnostics
        else:
            return prediction
    
    def predict_sequence(
        self,
        time_series: np.ndarray,
        return_diagnostics: bool = False
    ):
        """
        Batch prediction over entire time series
        
        Args:
            time_series: Input observations
            return_diagnostics: Whether to return full diagnostic trace
            
        Returns:
            predictions: Array of forecasts
            diagnostics (optional): List of diagnostic dicts per time step
        """
        predictions = []
        
        for x_t in time_series:
            pred = self.predict_step(x_t, return_diagnostics=False)
            predictions.append(pred)
        
        if return_diagnostics:
            return np.array(predictions), self.diagnostics
        else:
            return np.array(predictions)
    
    def reset(self):
        """
        Reset adapter for new time series / episode
        """
        self.hca = HausdorffContextAdapter(
            K=self.hca.K,
            sigma_noise=self.hca.sigma_noise,
            lambda_reset=self.hca.lambda_reset,
            lambda_caution=self.hca.lambda_caution,
            smoothing_alpha=self.hca.smoothing_alpha
        )
        self.context_manager.clear_history()
        self.t = 0
        self.predictions = []
        self.diagnostics = []
    
    def get_uncertainty_summary(self) -> Dict:
        """
        Aggregate uncertainty metrics over trajectory
        """
        if not self.hca.is_initialized:
            return {}
        
        return {
            'mean_diameter': np.mean(self.hca.diameter_history),
            'std_diameter': np.std(self.hca.diameter_history),
            'mean_ratio': np.mean(self.hca.ratio_history),
            'num_resets': sum(d['reset_occurred'] for d in self.diagnostics),
            'avg_context_length': np.mean([d['context_length'] for d in self.diagnostics])
        }
