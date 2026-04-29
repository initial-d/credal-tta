# Credal-TTA Package Initialization

from .credal_tta import CredalTTA
from .core.hca import HausdorffContextAdapter
from .core.context_manager import ContextManager
from .core.credal_set import CredalSet, GaussianDistribution
from .core.hca_multivariate import MultivariateDiagonalHCA
from .utils.preprocessing import TwoTrackPreprocessor
from .models.tta_baselines import LoRATTA, TENT_TTA

__version__ = "1.1.0"
__all__ = [
    'CredalTTA',
    'HausdorffContextAdapter',
    'ContextManager',
    'CredalSet',
    'GaussianDistribution',
    'MultivariateDiagonalHCA',
    'TwoTrackPreprocessor',
    'LoRATTA',
    'TENT_TTA',
]
