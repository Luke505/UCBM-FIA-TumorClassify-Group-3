"""
Unit tests for basic validation strategy splitting behavior

Tests that Holdout and Random Subsampling strategies produce valid
train/test splits with correct index properties
"""

import os
import sys
import unittest

import numpy as np

# Add src to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.evaluation.strategies import HoldoutValidation, RandomSubsampling


class TestBasicStrategySplits(unittest.TestCase):
    """
    Test cases for basic validation strategy split behavior

    Verifies that Holdout and Random Subsampling produce valid
    non-overlapping train/test index splits covering all samples
    """

    def setUp(self):
        """Set up test fixtures with small feature matrix and labels"""
        np.random.seed(42)
        self.features = np.random.rand(5, 2)
        self.labels = np.array([2, 4, 2, 4, 2])

    def test_holdout_split_indices_are_valid(self):
        """Test that Holdout produces valid non-overlapping train/test indices"""
        strategy = HoldoutValidation(test_size=0.4, random_state=42)

        for train_idx, test_idx in strategy.split(self.features, self.labels):
            # Train and test indices should not overlap
            overlap = set(train_idx).intersection(set(test_idx))
            self.assertEqual(len(overlap), 0, "Train and test indices must not overlap")

            # Together they should cover all samples
            all_indices = set(train_idx).union(set(test_idx))
            self.assertEqual(all_indices, set(range(len(self.features))),
                             "Train + test indices must cover all samples")

            # Test size should match the requested proportion
            expected_test_size = int(len(self.features) * 0.4)
            self.assertEqual(len(test_idx), expected_test_size)

    def test_random_subsampling_produces_correct_number_of_splits(self):
        """Test that Random Subsampling produces the correct number of splits"""
        n_splits = 3
        strategy = RandomSubsampling(n_splits=n_splits, test_size=0.4, random_state=42)

        splits = list(strategy.split(self.features, self.labels))
        self.assertEqual(len(splits), n_splits,
                         f"Should produce exactly {n_splits} splits")

        # Each split should have non-overlapping train/test indices
        for train_idx, test_idx in splits:
            overlap = set(train_idx).intersection(set(test_idx))
            self.assertEqual(len(overlap), 0, "Train and test indices must not overlap")

            all_indices = set(train_idx).union(set(test_idx))
            self.assertEqual(all_indices, set(range(len(self.features))),
                             "Train + test indices must cover all samples")


if __name__ == '__main__':
    unittest.main()
