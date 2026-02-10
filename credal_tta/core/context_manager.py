"""
Reset-and-Grow Context Manager - Section 4.2 in paper
Dynamic context pruning to eliminate context inertia
"""

import numpy as np
from typing import List, Tuple


class ContextManager:
    """
    Manages dynamic context window with reset-and-grow strategy
    
    Key operations:
    - Reset: Prune toxic historical context upon regime shift
    - Grow: Gradually expand context in stable post-shift phase
    """
    
    def __init__(
        self,
        W_max: int = 512,
        L_min: int = 10
    ):
        """
        Args:
            W_max: Maximum context length (TSFM capacity)
            L_min: Minimum context buffer (cold-start safeguard)
        """
        self.W_max = W_max
        self.L_min = L_min
        
        # Context origin (tracks last reset point)
        self.S: int = 0
        
        # Full historical data (for windowing)
        self.history: List[float] = []
        
    def update(self, x_new: float, regime_shift: bool, t: int) -> Tuple[np.ndarray, dict]:
        """
        Update context window (Algorithm 1, lines 10-15)
        
        Args:
            x_new: New observation
            regime_shift: Whether HCA detected regime shift
            t: Current time step
            
        Returns:
            context: Effective context array for TSFM
            info: Diagnostic information
        """
        # Append to history
        self.history.append(x_new)
        
        # Reset origin upon regime shift (Eq. 10)
        if regime_shift:
            self.S = t
        
        # Construct effective context (Eq. 11)
        t_start = max(self.S, t - self.W_max, t - len(self.history) + 1)
        t_start = max(t_start, t - self.L_min)  # Ensure minimum context
        
        # Extract context window
        start_idx = max(0, len(self.history) - (t - t_start + 1))
        context = np.array(self.history[start_idx:])
        
        info = {
            'context_length': len(context),
            'origin': self.S,
            'reset_occurred': regime_shift,
            't_start': t_start
        }
        
        return context, info
    
    def get_context_stats(self) -> dict:
        """
        Return context statistics for monitoring
        """
        return {
            'history_length': len(self.history),
            'origin': self.S,
            'age_since_reset': len(self.history) - self.S if self.S > 0 else len(self.history)
        }
    
    def clear_history(self):
        """
        Clear all historical data (for new episode/dataset)
        """
        self.history = []
        self.S = 0
