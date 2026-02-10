"""
Credal Set Operations and Wasserstein Distance Computation
"""

import numpy as np
from scipy.stats import norm
from typing import List, Tuple


class GaussianDistribution:
    """Single Gaussian distribution as extreme point of credal set"""
    
    # [修改点 1]: __init__ 增加 sensitivity 参数，默认值为 1.0 保证兼容性
    def __init__(self, mu: float, sigma: float, sensitivity: float = 1.0):
        self.mu = mu
        self.sigma = max(sigma, 1e-6)  # Prevent numerical issues
        self.sensitivity = sensitivity # <--- 新增属性：敏感度 (学习率因子)
    
    def pdf(self, x: float) -> float:
        """Probability density function"""
        return norm.pdf(x, loc=self.mu, scale=self.sigma)
    
    def sample(self, n: int = 1) -> np.ndarray:
        """Generate samples"""
        return np.random.normal(self.mu, self.sigma, n)
    
    def bayesian_update(self, x_obs: float, sigma_noise: float) -> 'GaussianDistribution':
        """
        Bayesian update with Gaussian likelihood
        
        Args:
            x_obs: Observed value
            sigma_noise: Observation noise standard deviation
            
        Returns:
            Updated posterior Gaussian distribution
        """
        # [修改点 2]: 计算有效噪声 (Effective Noise)
        # sensitivity < 1.0 (敏感粒子): 认为噪声小 -> 权重高 -> 更新快
        # sensitivity > 1.0 (迟钝粒子): 认为噪声大 -> 权重低 -> 更新慢
        effective_sigma_noise = sigma_noise * self.sensitivity 

        # Posterior precision (inverse variance)
        # 注意这里用了 effective_sigma_noise
        #kappa_inv = 1.0 / (self.sigma ** 2) + 1.0 / (effective_sigma_noise ** 2)
        #kappa = 1.0 / kappa_inv
        
        # Posterior mean (precision-weighted average)
        # 注意这里用了 effective_sigma_noise
        #mu_post = kappa * (self.mu / (self.sigma ** 2) + x_obs / (effective_sigma_noise ** 2))

        epsilon = 1e-8
        kappa_inv = 1.0 / (self.sigma ** 2 + epsilon) + 1.0 / (effective_sigma_noise ** 2 + epsilon)
        kappa = 1.0 / (kappa_inv + epsilon)

        mu_post = kappa * (
            self.mu / (self.sigma ** 2 + epsilon) +
            x_obs / (effective_sigma_noise ** 2 + epsilon)
        )
        
        # Posterior standard deviation
        sigma_post = np.sqrt(kappa)

        global_min_sigma = sigma_noise * 0.2
        sigma_post = max(sigma_post, global_min_sigma)

        # [修改点 3]: 将 sensitivity 传递给更新后的分布，保持其性格不变
        return GaussianDistribution(mu_post, sigma_post, sensitivity=self.sensitivity)
    
    def __repr__(self):
        # [修改点 4]: 打印时显示 sensitivity 方便调试
        return f"N(μ={self.mu:.3f}, σ={self.sigma:.3f}, sen={self.sensitivity:.2f})"


def wasserstein_2_gaussian(p: GaussianDistribution, q: GaussianDistribution) -> float:
    """
    Wasserstein-2 distance between two univariate Gaussians
    
    Closed-form formula (Eq. 7 in paper):
    W_2^2 = (μ_p - μ_q)^2 + (σ_p - σ_q)^2
    
    Args:
        p, q: Gaussian distributions
        
    Returns:
        Wasserstein-2 distance
    """
    mean_diff_sq = (p.mu - q.mu) ** 2
    std_diff_sq = (p.sigma - q.sigma) ** 2
    
    return np.sqrt(mean_diff_sq + std_diff_sq)


class CredalSet:
    """
    Finitely-Generated Credal Set (FGCS) - Definition 2.2 in paper
    
    Maintains K extreme Gaussian distributions and computes Hausdorff diameter
    """
    
    def __init__(self, extremes: List[GaussianDistribution]):
        self.extremes = extremes
        self.K = len(extremes)
    
    def diameter(self) -> float:
        """
        Hausdorff diameter (Definition 2.3 in paper)
        
        Returns:
            Maximum pairwise Wasserstein-2 distance among extremes
        """
        if self.K == 1:
            return 0.0
        
        max_dist = 0.0
        for i in range(self.K):
            for j in range(i + 1, self.K):
                dist = wasserstein_2_gaussian(self.extremes[i], self.extremes[j])
                max_dist = max(max_dist, dist)
        
        return max_dist
    
    def update(self, x_obs: float, sigma_noise: float) -> 'CredalSet':
        """
        Online Bayesian update of all extreme distributions
        
        Args:
            x_obs: Observed value
            sigma_noise: Observation noise
            
        Returns:
            Updated credal set
        """
        updated_extremes = [
            extreme.bayesian_update(x_obs, sigma_noise)
            for extreme in self.extremes
        ]
        return CredalSet(updated_extremes)
    
    def contraction_ratio(self, prev_diameter: float, epsilon: float = 1e-6) -> float:
        """
        Compute empirical contraction ratio (Definition 2.4 in paper)
        
        Args:
            prev_diameter: Diameter at previous time step
            epsilon: Small constant to prevent division by zero
            
        Returns:
            ρ_t = diam(K_t) / (diam(K_{t-1}) + ε)
        """
        curr_diameter = self.diameter()
        return curr_diameter / (prev_diameter + epsilon)
    
    def __repr__(self):
        return f"CredalSet(K={self.K}, diam={self.diameter():.4f})"


def initialize_credal_set(
    burn_in_data: np.ndarray,
    K: int = 3,
    divergence_factor: float = 1.0
) -> CredalSet:
    """
    Initialize credal set from burn-in data (Section 4.1 in paper)
    
    Creates K extreme distributions:
    - P_1: Pessimistic (mean - std)
    - P_2: Optimistic (mean + std)
    - P_3: Neutral (mean)
    
    Args:
        burn_in_data: Initial observations for parameter estimation
        K: Number of extreme distributions (default: 3)
        divergence_factor: Controls initial spread
        
    Returns:
        Initialized credal set
    """
    mean = np.mean(burn_in_data)
    std = np.std(burn_in_data)
    
    # [修改点 5]: 初始化逻辑全面升级，赋予不同的 sensitivity
    
    if K == 2:
        extremes = [
            # 一个快 (0.1)，一个慢 (5.0)
            GaussianDistribution(mean - divergence_factor * std, 0.5 * std, sensitivity=0.1),
            GaussianDistribution(mean + divergence_factor * std, 0.5 * std, sensitivity=5.0),
        ]
    elif K == 3:
        extremes = [
            # P1 (悲观): 极其敏锐 (Fast Learner) -> 遇到突变先跑
            GaussianDistribution(mean - divergence_factor * std, 0.5 * std, sensitivity=0.1),  
            # P2 (乐观): 极其迟钝 (Slow Learner) -> 遇到突变不动
            GaussianDistribution(mean + divergence_factor * std, 0.5 * std, sensitivity=5.0),  
            # P3 (中立): 正常速度
            GaussianDistribution(mean, std, sensitivity=1.0),  
        ]
    elif K == 5:
        # 保留了你原本的 K=5 逻辑，但增加了 sensitivity 的梯度分布
        # 梯度: [0.1, 0.5, 1.0, 2.0, 5.0]
        extremes = [
            GaussianDistribution(mean - 1.5 * divergence_factor * std, 0.4 * std, sensitivity=0.1),
            GaussianDistribution(mean - 0.5 * divergence_factor * std, 0.5 * std, sensitivity=0.5),
            GaussianDistribution(mean, std, sensitivity=1.0),
            GaussianDistribution(mean + 0.5 * divergence_factor * std, 0.5 * std, sensitivity=2.0),
            GaussianDistribution(mean + 1.5 * divergence_factor * std, 0.4 * std, sensitivity=5.0),
        ]
    else:
        # General case: uniform spread
        extremes = []
        offsets = np.linspace(-divergence_factor * std, divergence_factor * std, K)
        # 动态生成 sensitivity 梯度
        sensitivities = np.linspace(0.1, 5.0, K)
        
        for i, offset in enumerate(offsets):
            extremes.append(GaussianDistribution(
                mean + offset, 
                0.5 * std, 
                sensitivity=sensitivities[i] # 动态分配
            ))
    
    return CredalSet(extremes)

