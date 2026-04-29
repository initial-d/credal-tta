"""
Multivariate HCA with Diagonal Covariance Approximation (R1-W3)
Reduces complexity from O(K²d³) to O(K²d)
"""

import numpy as np
from typing import Dict, List


class MultivariateDiagonalHCA:
    """
    Multivariate HCA using diagonal covariance approximation
    W2(P_k, P_k') = ||μ_k - μ_k'||² + Σ_j(σ_kj - σ_k'j)²
    """
    
    def __init__(self, d: int, K: int = 3, lambda_reset: float = 1.3):
        self.d = d
        self.K = K
        self.lambda_reset = lambda_reset
        
        # K Gaussian distributions, each with d-dim mean and d variances
        self.means = [np.zeros(d) for _ in range(K)]
        self.stds = [np.ones(d) for _ in range(K)]
        
        self.is_initialized = False
        self.t = 0
        
    def update(self, x_t: np.ndarray) -> Dict:
        """Update with d-dimensional observation"""
        assert x_t.shape[0] == self.d
        
        if not self.is_initialized:
            # Initialize with diffuse priors
            for k in range(self.K):
                self.means[k] = x_t + np.random.randn(self.d) * 0.1
                self.stds[k] = np.ones(self.d)
            self.is_initialized = True
            return {'regime_shift': False, 'diameter': 0.0}
        
        # Bayesian update (dimension-wise)
        for k in range(self.K):
            for j in range(self.d):
                # Simple Kalman-like update per dimension
                prior_var = self.stds[k][j] ** 2
                obs_var = 0.1  # Assumed observation noise
                
                K_gain = prior_var / (prior_var + obs_var)
                self.means[k][j] += K_gain * (x_t[j] - self.means[k][j])
                self.stds[k][j] = np.sqrt((1 - K_gain) * prior_var)
        
        # Compute diameter using diagonal W2
        diameter = self._compute_diameter()
        
        self.t += 1
        
        return {
            'regime_shift': False,  # Simplified
            'diameter': diameter
        }
    
    def _compute_diameter(self) -> float:
        """Compute credal set diameter with diagonal W2"""
        max_dist = 0.0
        
        for i in range(self.K):
            for j in range(i + 1, self.K):
                # ||μ_i - μ_j||²
                mean_dist = np.sum((self.means[i] - self.means[j]) ** 2)
                
                # Σ_d(σ_id - σ_jd)²
                std_dist = np.sum((self.stds[i] - self.stds[j]) ** 2)
                
                w2_dist = np.sqrt(mean_dist + std_dist)
                max_dist = max(max_dist, w2_dist)
        
        return max_dist


# Alias for validation
MultivariateHCA = MultivariateDiagonalHCA
