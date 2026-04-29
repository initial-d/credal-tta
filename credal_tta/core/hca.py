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
        epsilon: float = 0.5,
        T0: int = 50,
        lambda_hard: float = 2.5,
        use_burn_in_health_check: bool = True
    ):
        """
        Args:
            K: Number of extreme distributions
            sigma_noise: Observation noise std (auto-estimated if None)
            lambda_reset: Threshold for regime detection (Eq. 8)
            lambda_caution: Threshold for stable regime
            smoothing_alpha: EMA smoothing for contraction ratio
            epsilon: Small constant for numerical stability
            T0: Burn-in window length
            lambda_hard: Conservative fixed threshold for burn-in health check
            use_burn_in_health_check: Whether to enable burn-in health check (R1-W2)
        """
        self.K = K
        self.sigma_noise = sigma_noise
        self.lambda_reset = lambda_reset
        self.lambda_caution = lambda_caution
        self.smoothing_alpha = smoothing_alpha
        self.epsilon = epsilon
        self.T0 = T0
        self.lambda_hard = lambda_hard
        self.use_burn_in_health_check = use_burn_in_health_check
        
        # State variables
        self.credal_set: Optional[CredalSet] = None
        self.prev_diameter: float = 0.0
        self.diameter_history: List[float] = []
        self.ratio_history: List[float] = []
        self.smoothed_ratio: float = 1.0
        
        self.is_initialized: bool = False
        self.burn_in_data: List[float] = []
        self.burn_in_restart_count: int = 0
        
        # Adaptive threshold (estimated after burn-in)
        self.mu_rho: float = 1.0
        self.sigma_rho: float = 0.0
        self.lambda_adaptive: Optional[float] = None
        
    def initialize(self, burn_in_data: np.ndarray):
        """
        Initialize from burn-in period with optional health check (Algorithm 1, Lines 3-10)
        
        If use_burn_in_health_check is True, performs incremental Bayesian updates
        during burn-in and monitors for structural breaks using lambda_hard.
        If a break is detected (at most one restart), the burn-in is restarted
        with diffuse priors from recent observations.
        
        Args:
            burn_in_data: Initial observations (length T_0 = 50-100)
        """
        if self.sigma_noise is None:
            if len(burn_in_data) > 10:
                diffs = np.diff(burn_in_data)
                self.sigma_noise = 0.1 * np.std(diffs)
            else:
                self.sigma_noise = 0.1 * np.std(burn_in_data)
        
        if self.use_burn_in_health_check and len(burn_in_data) >= 10:
            burn_in_data = self._burn_in_with_health_check(burn_in_data)
        
        self.credal_set = initialize_credal_set(burn_in_data, K=self.K)
        self.prev_diameter = self.credal_set.diameter()
        self.diameter_history = [self.prev_diameter]
        self.ratio_history = [1.0]
        self.is_initialized = True
        
        # Estimate adaptive threshold from burn-in contraction ratios
        if len(self.ratio_history) > 5:
            self.mu_rho = float(np.mean(self.ratio_history))
            self.sigma_rho = float(np.std(self.ratio_history))
            self.lambda_adaptive = self.mu_rho + 3 * self.sigma_rho
    
    def _burn_in_with_health_check(self, burn_in_data: np.ndarray) -> np.ndarray:
        """
        Burn-in health check (Algorithm 1, Lines 3-10, Remark 3)
        
        Monitors contraction ratio during burn-in using conservative lambda_hard.
        If a structural break is detected, restarts burn-in with diffuse priors.
        At most one restart is allowed to prevent infinite cycling.
        
        Args:
            burn_in_data: Raw burn-in observations
            
        Returns:
            Cleaned burn-in data (possibly truncated after restart)
        """
        restart_count = 0
        temp_credal = initialize_credal_set(burn_in_data[:5], K=self.K)
        prev_diam = temp_credal.diameter()
        ratios = [1.0]
        
        for t in range(5, len(burn_in_data)):
            x_t = burn_in_data[t]
            temp_credal = temp_credal.update(x_t, self.sigma_noise)
            curr_diam = temp_credal.diameter()
            rho = curr_diam / (prev_diam + self.epsilon)
            ratios.append(rho)
            
            if rho > self.lambda_hard and restart_count == 0:
                # Burn-in contaminated: restart with diffuse priors
                restart_count += 1
                self.burn_in_restart_count = restart_count
                
                # Use last min(t,10) observations for restart
                lookback = min(t, 10)
                recent = burn_in_data[t - lookback + 1:t + 1]
                x_bar = float(np.mean(recent))
                s_hat = float(np.std(recent)) + 1e-8
                
                # Reinitialize with maximally diffuse priors (Remark 3)
                from .credal_set import GaussianDistribution, CredalSet
                temp_credal = CredalSet([
                    GaussianDistribution(x_bar - 3 * s_hat, s_hat, sensitivity=0.1),
                    GaussianDistribution(x_bar + 3 * s_hat, s_hat, sensitivity=5.0),
                    GaussianDistribution(x_bar, 3 * s_hat, sensitivity=1.0),
                ])
                prev_diam = temp_credal.diameter()
                
                # Return remaining data from restart point
                remaining = burn_in_data[t:]
                if len(remaining) < 20:
                    # Extend: use what we have plus pad
                    return burn_in_data[max(0, t - 10):]
                return remaining
            
            prev_diam = curr_diam
        
        # Store burn-in ratios for adaptive threshold estimation
        self.ratio_history = ratios
        return burn_in_data
        
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
