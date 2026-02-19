"""
Evaluation strategies module

This module implements various cross-validation and holdout strategies
using the Strategy design pattern
"""

from .strategies import (
    BaseValidationStrategy,
    HoldoutValidation,
    RandomSubsampling,
    KFoldCrossValidation,
    LeaveOneOutCrossValidation,
    LeavePOutCrossValidation,
    StratifiedCrossValidation,
    StratifiedShuffleSplit,
    Bootstrap,
    ValidationStrategyFactory
)

__all__ = [
    'BaseValidationStrategy',
    'HoldoutValidation',
    'RandomSubsampling',
    'KFoldCrossValidation',
    'LeaveOneOutCrossValidation',
    'LeavePOutCrossValidation',
    'StratifiedCrossValidation',
    'StratifiedShuffleSplit',
    'Bootstrap',
    'ValidationStrategyFactory'
]
