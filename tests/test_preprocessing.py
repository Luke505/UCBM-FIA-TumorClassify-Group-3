"""
Unit tests for preprocessing module

Tests feature normalization including the fit/transform pattern
used to avoid data leakage during cross-validation
"""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.utils.preprocessing import (
    normalize_features,
    compute_normalization_params,
    apply_normalization
)


class TestNormalizeFeatures(unittest.TestCase):
    """Test cases for the convenience normalize_features function"""

    def test_normalization_to_zero_one(self):
        """Test that features are normalized to the range [0, 1]"""
        features = np.array([
            [1, 10],
            [2, 20],
            [3, 30]
        ])

        features_norm = normalize_features(features)

        self.assertTrue(np.all(features_norm >= 0))
        self.assertTrue(np.all(features_norm <= 1))

        np.testing.assert_almost_equal(features_norm[:, 0].min(), 0)
        np.testing.assert_almost_equal(features_norm[:, 0].max(), 1)
        np.testing.assert_almost_equal(features_norm[:, 1].min(), 0)
        np.testing.assert_almost_equal(features_norm[:, 1].max(), 1)

    def test_normalization_constant_feature(self):
        """Test normalization of constant features (all the same value)"""
        features = np.array([
            [1, 5],
            [2, 5],
            [3, 5]
        ])

        features_norm = normalize_features(features)
        np.testing.assert_array_equal(features_norm[:, 1], np.zeros(3))

    def test_normalization_empty_array(self):
        """Test normalization of empty array"""
        features = np.array([]).reshape(0, 9)
        features_norm = normalize_features(features)
        self.assertEqual(features_norm.shape, (0, 9))

    def test_normalization_single_sample(self):
        """Test normalization with a single sample"""
        features = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9]])
        features_norm = normalize_features(features)

        self.assertEqual(features_norm.shape, features.shape)
        self.assertTrue(np.all(np.isfinite(features_norm)))
        np.testing.assert_array_equal(features_norm, np.zeros_like(features, dtype=np.float64))

    def test_normalization_preserves_shape(self):
        """Test that normalization preserves the shape"""
        features = np.array([
            [1, 5, 3, 7, 2, 8, 4, 6, 9],
            [2, 6, 4, 8, 3, 9, 5, 7, 10],
            [3, 7, 5, 9, 4, 10, 6, 8, 1]
        ])
        features_norm = normalize_features(features)
        self.assertEqual(features_norm.shape, features.shape)


class TestFitTransformNormalization(unittest.TestCase):
    """Test cases for the fit/transform normalization pattern
    
    These functions are used during cross-validation to compute
    normalization parameters on training data only (avoiding data leakage)
    and then apply them to both training and test data.
    """

    def test_compute_params_returns_correct_shape(self):
        """Test that compute_normalization_params returns per-feature min/max"""
        features = np.array([
            [1, 10, 5],
            [3, 20, 5],
            [5, 30, 5]
        ])

        min_vals, max_vals = compute_normalization_params(features)

        self.assertEqual(min_vals.shape, (3,))
        self.assertEqual(max_vals.shape, (3,))
        np.testing.assert_array_equal(min_vals, [1, 10, 5])
        np.testing.assert_array_equal(max_vals, [5, 30, 5])

    def test_apply_normalization_with_train_params(self):
        """Test applying training params normalizes training data to [0, 1]"""
        train = np.array([
            [1, 10],
            [3, 20],
            [5, 30]
        ])

        min_vals, max_vals = compute_normalization_params(train)
        train_norm = apply_normalization(train, min_vals, max_vals)

        np.testing.assert_almost_equal(train_norm[:, 0], [0.0, 0.5, 1.0])
        np.testing.assert_almost_equal(train_norm[:, 1], [0.0, 0.5, 1.0])

    def test_test_data_can_exceed_zero_one(self):
        """Test that test data normalized with train params can go outside [0, 1]"""
        train = np.array([
            [2, 10],
            [4, 20]
        ])
        test = np.array([
            [1, 25],  # feature 0 below train min, feature 1 above train max
            [6, 15]   # feature 0 above train max
        ])

        min_vals, max_vals = compute_normalization_params(train)
        test_norm = apply_normalization(test, min_vals, max_vals)

        # Feature 0: (1-2)/(4-2) = -0.5, (6-2)/(4-2) = 2.0
        self.assertAlmostEqual(test_norm[0, 0], -0.5)
        self.assertAlmostEqual(test_norm[1, 0], 2.0)

    def test_constant_feature_handled(self):
        """Test that constant features (max == min) are set to 0"""
        features = np.array([
            [1, 5],
            [3, 5],
            [5, 5]
        ])

        min_vals, max_vals = compute_normalization_params(features)
        features_norm = apply_normalization(features, min_vals, max_vals)

        # Constant feature (column 1) should be all zeros
        np.testing.assert_array_equal(features_norm[:, 1], [0.0, 0.0, 0.0])

    def test_fit_transform_equivalent_to_convenience(self):
        """Test that fit/transform gives same result as normalize_features"""
        features = np.array([
            [1, 5, 3, 7, 2, 8, 4, 6, 9],
            [2, 6, 4, 8, 3, 9, 5, 7, 10],
            [3, 7, 5, 9, 4, 10, 6, 8, 1]
        ])

        # Convenience function
        result_convenience = normalize_features(features)

        # Fit/transform
        min_vals, max_vals = compute_normalization_params(features)
        result_fit_transform = apply_normalization(features, min_vals, max_vals)

        np.testing.assert_array_almost_equal(result_convenience, result_fit_transform)

    def test_apply_normalization_empty_array(self):
        """Test apply_normalization with empty array"""
        features = np.array([]).reshape(0, 3)
        min_vals = np.array([1, 2, 3])
        max_vals = np.array([5, 6, 7])

        result = apply_normalization(features, min_vals, max_vals)
        self.assertEqual(result.shape, (0, 3))


if __name__ == '__main__':
    unittest.main()
