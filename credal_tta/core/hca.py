"""
Hausdorff Context Adapter (HCA) - Section 4.1 in paper
Lightweight epistemic uncertainty detector based on credal set theory
"""

import numpy as np
from typing import Optional, List
from .credal_set import CredalSet, initialize_credal_set


class HausdorffContextAdapter:
    """
    HCA monitors distributional stability via credal set diameter dynamics
    
    Key properties (Theorems 3.1-3.2):
    - Stable regime → geometric diameter contraction (ρ_t < 1)
    - Regime shift → explosive diameter expansion (ρ_t >> 1)
    """
    
    def __init__(
        self,
        K: int = 3,
        sigma_noise: float = None,
        lambda_reset: float = 1.2,
        lambda_caution: float = 0.95,
        smoothing_alpha: float = 1,
        epsilon: float = 0.5
    ):
        """
        Args:
            K: Number of extreme distributions
            sigma_noise: Observation noise std (auto-estimated if None)
            lambda_reset: Threshold for regime detection (Eq. 8)
            lambda_caution: Threshold for stable regime
            smoothing_alpha: EMA smoothing for contraction ratio
            epsilon: Small constant for numerical stability
        """
        self.K = K
        self.sigma_noise = sigma_noise
        self.lambda_reset = lambda_reset
        self.lambda_caution = lambda_caution
        self.smoothing_alpha = smoothing_alpha
        self.epsilon = epsilon
        
        # State variables
        self.credal_set: Optional[CredalSet] = None
        self.prev_diameter: float = 0.0
        self.diameter_history: List[float] = []
        self.ratio_history: List[float] = []
        self.smoothed_ratio: float = 1.0
        
        self.is_initialized: bool = False
        self.burn_in_data: List[float] = []
        
    def initialize(self, burn_in_data: np.ndarray):
        """
        Initialize from burn-in period (Section 5.3 in paper)
        
        Args:
            burn_in_data: Initial observations (length T_0 = 50-100)
        """
        if self.sigma_noise is None:
            # Estimate observation noise from high-frequency residuals
            if len(burn_in_data) > 10:
                diffs = np.diff(burn_in_data)
                self.sigma_noise = 0.1 * np.std(diffs)
            else:
                self.sigma_noise = 0.1 * np.std(burn_in_data)
        
        self.credal_set = initialize_credal_set(burn_in_data, K=self.K)
        self.prev_diameter = self.credal_set.diameter()
        self.diameter_history = [self.prev_diameter]
        self.ratio_history = [1.0]
        self.is_initialized = True
        
    def update(self, x_obs: float) -> dict:
        """
        Online Bayesian update and regime detection (Algorithm 1, lines 4-9)
        
        Args:
            x_obs: New observation
            
        Returns:
            dict with keys:
                - 'regime_shift': bool, whether shift detected
                - 'diameter': current diameter
                - 'ratio': contraction ratio
                - 'smoothed_ratio': EMA-smoothed ratio
        """
        if not self.is_initialized:
            # Accumulate burn-in data (reduced from 50 to 20 for faster initialization)
            self.burn_in_data.append(x_obs)
            if len(self.burn_in_data) >= 20:
                self.initialize(np.array(self.burn_in_data))
            return {
                'regime_shift': False,
                'diameter': 0.0,
                'ratio': 1.0,
                'smoothed_ratio': 1.0
            }
        
        # Bayesian update of credal set
        self.credal_set = self.credal_set.update(x_obs, self.sigma_noise)
        
        # Compute diameter and contraction ratio
        curr_diameter = self.credal_set.diameter()
        ratio = self.credal_set.contraction_ratio(self.prev_diameter, self.epsilon)
        
        # Exponential smoothing (optional, reduces false positives)
        #self.smoothed_ratio = (
        #    self.smoothing_alpha * ratio +
        #    (1 - self.smoothing_alpha) * self.smoothed_ratio
        #)
        self.smoothed_ratio = ratio
        
        # Detection rule (Eq. 8 in paper)
        regime_shift = self.smoothed_ratio > self.lambda_reset
        
        # Update history
        self.diameter_history.append(curr_diameter)
        self.ratio_history.append(ratio)
        self.prev_diameter = curr_diameter
        
        return {
            'regime_shift': regime_shift,
            'diameter': curr_diameter,
            'ratio': ratio,
            'smoothed_ratio': self.smoothed_ratio,
            'stable_regime': self.smoothed_ratio < self.lambda_caution
        }
    
    def reset_credal_set(self, recent_data: np.ndarray):
        """
        Reinitialize credal set after regime shift (Algorithm 1, line 12)
        
        Args:
            recent_data: Recent observations from new regime
        """
        if len(recent_data) >= 10:
            self.credal_set = initialize_credal_set(recent_data, K=self.K)
            self.prev_diameter = self.credal_set.diameter()
            self.smoothed_ratio = 1.0
    
    def get_uncertainty_metrics(self) -> dict:
        """
        Extract interpretable uncertainty metrics for monitoring
        
        Returns:
            dict with epistemic and aleatoric uncertainty measures
        """
        if not self.is_initialized:
            return {
                'epistemic_uncertainty': 0.0,
                'aleatoric_uncertainty': 0.0,
                'total_uncertainty': 0.0
            }
        
        # Epistemic: credal set diameter
        epistemic = self.credal_set.diameter()
        
        # Aleatoric: average variance within extremes
        aleatoric = np.mean([e.sigma ** 2 for e in self.credal_set.extremes])
        
        return {
            'epistemic_uncertainty': epistemic,
            'aleatoric_uncertainty': aleatoric,
            'total_uncertainty': epistemic + aleatoric
        }
