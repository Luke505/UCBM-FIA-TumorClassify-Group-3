"""
Data preprocessing utilities

This module provides functions for min-max feature normalization.
It supports both a simple convenience function and a fit/transform
pattern to avoid data leakage during cross-validation.
"""

from typing import Tuple

import numpy as np


def compute_normalization_params(features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute per-feature min and max values for min-max normalization

    These parameters should be computed on the training set only, then
    applied to both training and test sets to avoid data leakage.

    Parameters:
    -----------
    features : np.ndarray
        Feature matrix of shape (n_samples, n_features), typically the
        training set

    Returns:
    --------
    Tuple[np.ndarray, np.ndarray]
        (min_vals, max_vals) — per-feature minimum and maximum values,
        each of shape (n_features,)
    """
    min_vals = np.min(features, axis=0)
    max_vals = np.max(features, axis=0)
    return min_vals, max_vals


def apply_normalization(features: np.ndarray, min_vals: np.ndarray,
    max_vals: np.ndarray) -> np.ndarray:
    """
    Apply min-max normalization using pre-computed parameters

    For each feature: normalized = (value - min) / (max - min)
    Features where max == min are set to 0 (constant features).

    Parameters:
    -----------
    features : np.ndarray
        Feature matrix of shape (n_samples, n_features) to normalize
    min_vals : np.ndarray
        Per-feature minimum values of shape (n_features,), computed on
        the training set via compute_normalization_params()
    max_vals : np.ndarray
        Per-feature maximum values of shape (n_features,), computed on
        the training set via compute_normalization_params()

    Returns:
    --------
    np.ndarray
        Normalized feature matrix of shape (n_samples, n_features) with
        values in approximately [0, 1] (test values may exceed this range)
    """
    if features.shape[0] == 0:
        return np.array(features, dtype=np.float64)

    features_normalized = np.zeros_like(features, dtype=np.float64)
    ranges = max_vals - min_vals

    for feature_idx in range(features.shape[1]):
        if ranges[feature_idx] > 0:
            features_normalized[:, feature_idx] = (
                (features[:, feature_idx] - min_vals[feature_idx]) / ranges[feature_idx]
            )
        else:
            # Constant feature: set to 0
            features_normalized[:, feature_idx] = 0

    return features_normalized


def normalize_features(features: np.ndarray) -> np.ndarray:
    """
    Normalize feature matrix to [0, 1] range using min-max normalization

    Convenience function that computes parameters and applies normalization
    in a single call. For cross-validation, prefer using
    compute_normalization_params() + apply_normalization() separately to
    avoid data leakage.

    For each feature: feature_normalized = (feature - min) / (max - min)

    Parameters:
    -----------
    features : np.ndarray
        Feature matrix of shape (n_samples, n_features)

    Returns:
    --------
    np.ndarray
        Normalized feature matrix with values in [0, 1]
    """
    if features.shape[0] == 0:
        return np.array(features, dtype=np.float64)

    min_vals, max_vals = compute_normalization_params(features)
    return apply_normalization(features, min_vals, max_vals)
