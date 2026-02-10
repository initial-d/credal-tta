# Credal-TTA Package Initialization

from .credal_tta import CredalTTA
from .core.hca import HausdorffContextAdapter
from .core.context_manager import ContextManager
from .core.credal_set import CredalSet, GaussianDistribution

__version__ = "1.0.0"
__all__ = [
    'CredalTTA',
    'HausdorffContextAdapter',
    'ContextManager',
    'CredalSet',
    'GaussianDistribution'
]
