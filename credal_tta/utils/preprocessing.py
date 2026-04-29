"""
Two-Track Data Flow Preprocessing (R1-W1)
Winsorization for HCA only, raw data for TSFM
"""

import numpy as np
from typing import Tuple


class TwoTrackPreprocessor:
    """
    Implements two-track data flow:
    - TSFM path: raw unmodified data
    - HCA path: log-transform + winsorization
    """
    
    def __init__(self, clip_percentile: float = 99.9):
        self.clip_percentile = clip_percentile
        self.clip_min = None
        self.clip_max = None
        
    def preprocess_for_hca(self, x: np.ndarray) -> np.ndarray:
        """Apply log + clip for HCA internal updates"""
        x_log = np.sign(x) * np.log1p(np.abs(x))
        
        if self.clip_min is None:
            self.clip_min = np.percentile(x_log, 100 - self.clip_percentile)
            self.clip_max = np.percentile(x_log, self.clip_percentile)
        
        return np.clip(x_log, self.clip_min, self.clip_max)
    
    def preprocess_for_tsfm(self, x: np.ndarray) -> np.ndarray:
        """Return raw data unchanged for TSFM"""
        return x


# Module-level convenience functions
_default_preprocessor = TwoTrackPreprocessor()

def preprocess_for_hca(x: np.ndarray) -> np.ndarray:
    """Module-level convenience: log + clip for HCA"""
    return _default_preprocessor.preprocess_for_hca(x)

def two_track_preprocess(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return (tsfm_data, hca_data) tuple"""
    return _default_preprocessor.preprocess_for_tsfm(x), _default_preprocessor.preprocess_for_hca(x)
