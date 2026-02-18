"""
Evaluation strategies module

This module implements holdout, random subsampling, and K-fold
cross-validation strategies using the Strategy design pattern
"""

from .strategies import (
    BaseValidationStrategy,
    HoldoutValidation,
    RandomSubsampling,
    KFoldCrossValidation,
    ValidationStrategyFactory
)

__all__ = [
    'BaseValidationStrategy',
    'HoldoutValidation',
    'RandomSubsampling',
    'KFoldCrossValidation',
    'ValidationStrategyFactory'
]
