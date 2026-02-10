"""
Improved Hausdorff Context Adapter Implementation
Integrates user's implementation with framework fixes
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional


@dataclass
class GaussianBelief:
    """Represents a single extreme distribution P_k = N(mu, sigma)."""
    mu: float
    sigma: float
    precision: float  # tau = 1/sigma^2 for efficient updates
    
    @classmethod
    def from_stats(cls, mu: float, sigma: float):
        """Create belief from mean and std, computing precision"""
        return cls(mu, sigma, 1.0 / (sigma**2 + 1e-6))
    
    def copy(self):
        """Deep copy of belief"""
        return GaussianBelief(self.mu, self.sigma, self.precision)


class HausdorffContextAdapter:
    """
    Implements Section 4.1: Hausdorff Context Adapter (HCA).
    Tracks the diameter of the Credal Set to detect regime shifts.
    
    Key improvements over basic implementation:
    1. Burn-in period handling
    2. EMA smoothing for robustness
    3. Better initialization strategy
    4. Diagnostic tracking
    """
    
    def __init__(
        self,
        K: int = 3,
        noise_std: Optional[float] = None,
        lambda_reset: float = 1.2,
        lambda_caution: float = 0.95,
        smoothing_alpha: float = 0.2,
        epsilon: float = 1e-6,
        burn_in_size: int = 20
    ):
        """
        Args:
            K: Number of extreme distributions
            noise_std: Observation noise (auto-estimated if None)
            lambda_reset: Threshold for regime detection
            lambda_caution: Threshold for stable regime
            smoothing_alpha: EMA smoothing factor for ratio
            epsilon: Small constant for numerical stability
            burn_in_size: Number of samples for initialization
        """
        self.K = K
        self.noise_std = noise_std
        self.lambda_reset = lambda_reset
        self.lambda_caution = lambda_caution
        self.smoothing_alpha = smoothing_alpha
        self.epsilon = epsilon
        self.burn_in_size = burn_in_size
        
        # State
        self.extremes: Optional[List[GaussianBelief]] = None
        self.noise_precision: Optional[float] = None
        self.prev_diameter: float = 0.0
        self.smoothed_ratio: float = 1.0
        
        # Tracking
        self.diameter_history: List[float] = []
        self.ratio_history: List[float] = []
        self.is_initialized: bool = False
        self.burn_in_buffer: List[float] = []
        
    def _initialize_extremes(self, data: np.ndarray) -> List[GaussianBelief]:
        """
        Sec 4.1.1: Initialize K extreme distributions
        Strategy: Pessimistic, Optimistic, Neutral (+ extras)
        """
        mu_init = np.mean(data)
        std_init = np.std(data) + 1e-6
        
        priors = []
        
        if self.K == 2:
            # Pessimistic and Optimistic only
            priors.append(GaussianBelief.from_stats(mu_init - std_init, 0.5 * std_init))
            priors.append(GaussianBelief.from_stats(mu_init + std_init, 0.5 * std_init))
            
        elif self.K == 3:
            # Paper's default: Pessimistic, Optimistic, Neutral
            priors.append(GaussianBelief.from_stats(mu_init - std_init, 0.5 * std_init))
            priors.append(GaussianBelief.from_stats(mu_init + std_init, 0.5 * std_init))
            priors.append(GaussianBelief.from_stats(mu_init, std_init))
            
        elif self.K == 5:
            # Extended: more coverage
            priors.append(GaussianBelief.from_stats(mu_init - 1.5 * std_init, 0.4 * std_init))
            priors.append(GaussianBelief.from_stats(mu_init - 0.5 * std_init, 0.5 * std_init))
            priors.append(GaussianBelief.from_stats(mu_init, std_init))
            priors.append(GaussianBelief.from_stats(mu_init + 0.5 * std_init, 0.5 * std_init))
            priors.append(GaussianBelief.from_stats(mu_init + 1.5 * std_init, 0.4 * std_init))
            
        else:
            # General case: uniform spread
            offsets = np.linspace(-std_init, std_init, self.K)
            for offset in offsets:
                priors.append(GaussianBelief.from_stats(mu_init + offset, 0.5 * std_init))
        
        return priors
    
    def initialize(self, context_data: np.ndarray):
        """
        Initialize HCA from burn-in data
        
        Args:
            context_data: Initial observations for parameter estimation
        """
        # Estimate noise if not provided
        if self.noise_std is None:
            if len(context_data) > 10:
                diffs = np.diff(context_data)
                self.noise_std = 0.1 * np.std(diffs)
            else:
                self.noise_std = 0.1 * np.std(context_data)
        
        self.noise_precision = 1.0 / (self.noise_std**2 + 1e-6)
        
        # Initialize extreme distributions
        self.extremes = self._initialize_extremes(context_data)
        
        # Initialize diameter tracking
        self.prev_diameter = self.compute_diameter()
        self.diameter_history = [self.prev_diameter]
        self.ratio_history = [1.0]
        self.is_initialized = True
    
    def update(self, x_t: float) -> dict:
        """
        Sec 4.1.2: Online Bayesian Update + Detection
        
        Args:
            x_t: New observation
            
        Returns:
            dict with detection results and diagnostics
        """
        # Handle burn-in period
        if not self.is_initialized:
            self.burn_in_buffer.append(x_t)
            if len(self.burn_in_buffer) >= self.burn_in_size:
                self.initialize(np.array(self.burn_in_buffer))
            return {
                'regime_shift': False,
                'diameter': 0.0,
                'ratio': 1.0,
                'smoothed_ratio': 1.0,
                'stable_regime': True
            }
        
        # Update each extreme distribution
        for belief in self.extremes:
            # Conjugate update for Gaussian with known variance
            old_precision = belief.precision
            new_precision = old_precision + self.noise_precision
            
            # Posterior mean: precision-weighted average
            belief.mu = (old_precision * belief.mu + self.noise_precision * x_t) / new_precision
            belief.precision = new_precision
            belief.sigma = np.sqrt(1.0 / new_precision)
        
        # Compute diameter and contraction ratio
        curr_diameter = self.compute_diameter()
        ratio = curr_diameter / (self.prev_diameter + self.epsilon)
        
        # EMA smoothing for robustness
        self.smoothed_ratio = (
            self.smoothing_alpha * ratio +
            (1 - self.smoothing_alpha) * self.smoothed_ratio
        )
        
        # Detection
        regime_shift = self.smoothed_ratio > self.lambda_reset
        stable_regime = self.smoothed_ratio < self.lambda_caution
        
        # Update history
        self.diameter_history.append(curr_diameter)
        self.ratio_history.append(ratio)
        self.prev_diameter = curr_diameter
        
        return {
            'regime_shift': regime_shift,
            'diameter': curr_diameter,
            'ratio': ratio,
            'smoothed_ratio': self.smoothed_ratio,
            'stable_regime': stable_regime
        }
    
    def _wasserstein_distance_1d(self, p: GaussianBelief, q: GaussianBelief) -> float:
        """
        Eq 7 (Simplified for 1D Gaussians):
        W2^2 = (mu_p - mu_q)^2 + (sigma_p - sigma_q)^2
        
        Returns: W2 distance (not squared)
        """
        return np.sqrt((p.mu - q.mu)**2 + (p.sigma - q.sigma)**2)
    
    def compute_diameter(self) -> float:
        """
        Definition 2.3: Hausdorff diameter
        Max pairwise Wasserstein distance among extremes
        """
        if self.extremes is None or len(self.extremes) < 2:
            return 0.0
        
        max_dist = 0.0
        for i in range(len(self.extremes)):
            for j in range(i + 1, len(self.extremes)):
                dist = self._wasserstein_distance_1d(self.extremes[i], self.extremes[j])
                if dist > max_dist:
                    max_dist = dist
        return max_dist
    
    def detect_shift(self, threshold: Optional[float] = None) -> Tuple[bool, float, float]:
        """
        Legacy interface for compatibility
        Calculates Contraction Ratio and checks threshold
        
        Args:
            threshold: Detection threshold (uses lambda_reset if None)
            
        Returns:
            (is_shift, current_diameter, rho_t)
        """
        if threshold is None:
            threshold = self.lambda_reset
        
        curr_diameter = self.compute_diameter()
        rho_t = curr_diameter / (self.prev_diameter + self.epsilon)
        is_shift = rho_t > threshold
        
        return is_shift, curr_diameter, rho_t
    
    def reset_credal_set(self, recent_data: np.ndarray):
        """
        Reinitialize credal set after regime shift
        
        Args:
            recent_data: Recent observations from new regime
        """
        if len(recent_data) >= 10:
            self.extremes = self._initialize_extremes(recent_data)
            self.prev_diameter = self.compute_diameter()
            self.smoothed_ratio = 1.0
    
    def get_uncertainty_metrics(self) -> dict:
        """
        Extract interpretable uncertainty metrics
        
        Returns:
            Epistemic and aleatoric uncertainty measures
        """
        if not self.is_initialized:
            return {
                'epistemic_uncertainty': 0.0,
                'aleatoric_uncertainty': 0.0,
                'total_uncertainty': 0.0
            }
        
        # Epistemic: credal set diameter
        epistemic = self.compute_diameter()
        
        # Aleatoric: average variance within extremes
        aleatoric = np.mean([e.sigma ** 2 for e in self.extremes])
        
        return {
            'epistemic_uncertainty': epistemic,
            'aleatoric_uncertainty': aleatoric,
            'total_uncertainty': epistemic + aleatoric
        }
    
    def get_extreme_stats(self) -> List[dict]:
        """
        Get statistics of all extreme distributions
        Useful for debugging and visualization
        """
        if not self.is_initialized:
            return []
        
        return [
            {
                'mu': e.mu,
                'sigma': e.sigma,
                'precision': e.precision
            }
            for e in self.extremes
        ]


# ==================== Usage Example ====================

def example_usage():
    """Demonstrate improved HCA usage"""
    print("=" * 70)
    print("Improved HCA Usage Example")
    print("=" * 70)
    print()
    
    # Generate synthetic data with regime shift
    np.random.seed(42)
    regime1 = np.random.normal(0, 1, 500)
    regime2 = np.random.normal(5, 2, 500)
    data = np.concatenate([regime1, regime2])
    
    # Initialize HCA
    hca = HausdorffContextAdapter(
        K=3,
        lambda_reset=1.2,
        lambda_caution=0.95,
        burn_in_size=20
    )
    
    print(f"Initialized HCA with K={hca.K} extremes")
    print(f"Detection threshold: λ_reset={hca.lambda_reset}")
    print()
    
    # Process data
    shift_detected_at = []
    
    for t, x_t in enumerate(data):
        result = hca.update(x_t)
        
        if result['regime_shift']:
            shift_detected_at.append(t)
            print(f"[t={t}] 🔴 Regime shift detected!")
            print(f"  Diameter: {result['diameter']:.4f}")
            print(f"  Ratio: {result['ratio']:.4f}")
            print(f"  Smoothed ratio: {result['smoothed_ratio']:.4f}")
            print()
    
    # Summary
    print("=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"True shift at: t=500")
    print(f"Detected {len(shift_detected_at)} shifts at: {shift_detected_at[:5]}")
    
    if len(shift_detected_at) > 0:
        lag = shift_detected_at[0] - 500
        print(f"Detection lag: {lag} steps")
    
    # Uncertainty evolution
    print()
    print("Uncertainty Evolution:")
    metrics = hca.get_uncertainty_metrics()
    print(f"  Epistemic: {metrics['epistemic_uncertainty']:.4f}")
    print(f"  Aleatoric: {metrics['aleatoric_uncertainty']:.4f}")
    
    # Extreme distributions
    print()
    print("Current Extreme Distributions:")
    for i, stats in enumerate(hca.get_extreme_stats()):
        print(f"  P{i+1}: μ={stats['mu']:.2f}, σ={stats['sigma']:.2f}")


if __name__ == "__main__":
    example_usage()
