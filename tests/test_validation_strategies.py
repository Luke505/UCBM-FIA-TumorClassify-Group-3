"""
Unit tests for validation strategies

Tests holdout and K-fold cross-validation strategies.
"""

import os
import sys
import unittest

import numpy as np

# Add src to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.evaluation.strategies import (
    HoldoutValidation,
    KFoldCrossValidation,
    ValidationStrategyFactory
)


class TestHoldoutValidation(unittest.TestCase):
    """Test cases for Holdout Validation"""

    def test_single_split(self):
        """Test that holdout produces single split"""
        features = np.random.rand(100, 5)
        labels = np.random.choice([2, 4], 100)

        strategy = HoldoutValidation(test_size=0.3, random_state=42)
        n_splits = strategy.get_n_splits(features, labels)

        self.assertEqual(n_splits, 1)

    def test_split_sizes(self):
        """Test that split sizes are correct"""
        features = np.random.rand(100, 5)
        labels = np.random.choice([2, 4], 100)

        strategy = HoldoutValidation(test_size=0.3, random_state=42)

        for train_idx, test_idx in strategy.split(features, labels):
            self.assertEqual(len(test_idx), 30)
            self.assertEqual(len(train_idx), 70)

    def test_reproducibility(self):
        """Test that random_state makes splits reproducible"""
        features = np.random.rand(100, 5)
        labels = np.random.choice([2, 4], 100)

        strategy1 = HoldoutValidation(test_size=0.3, random_state=42)
        strategy2 = HoldoutValidation(test_size=0.3, random_state=42)

        splits1 = list(strategy1.split(features, labels))
        splits2 = list(strategy2.split(features, labels))

        np.testing.assert_array_equal(splits1[0][0], splits2[0][0])
        np.testing.assert_array_equal(splits1[0][1], splits2[0][1])


class TestKFoldCrossValidation(unittest.TestCase):
    """Test cases for K-Fold Cross-Validation"""

    def test_number_of_splits(self):
        """Test that K-Fold produces K splits"""
        features = np.random.rand(100, 5)
        labels = np.random.choice([2, 4], 100)

        strategy = KFoldCrossValidation(n_splits=5, random_state=42)
        n_splits = strategy.get_n_splits(features, labels)

        self.assertEqual(n_splits, 5)

    def test_all_samples_used(self):
        """Test that all samples are used exactly once as test"""
        features = np.random.rand(100, 5)
        labels = np.random.choice([2, 4], 100)

        strategy = KFoldCrossValidation(n_splits=5, random_state=42)

        all_test_indices = []
        for train_idx, test_idx in strategy.split(features, labels):
            all_test_indices.extend(test_idx)

        # All indices should appear exactly once
        self.assertEqual(len(all_test_indices), 100)
        self.assertEqual(len(set(all_test_indices)), 100)

    def test_no_overlap(self):
        """Test that train and test sets don't overlap"""
        features = np.random.rand(100, 5)
        labels = np.random.choice([2, 4], 100)

        strategy = KFoldCrossValidation(n_splits=5, random_state=42)

        for train_idx, test_idx in strategy.split(features, labels):
            overlap = set(train_idx).intersection(set(test_idx))
            self.assertEqual(len(overlap), 0)


class TestValidationStrategyFactory(unittest.TestCase):
    """Test cases for ValidationStrategyFactory"""

    def test_create_holdout(self):
        """Test creation of holdout strategy"""
        strategy = ValidationStrategyFactory.create_strategy('holdout', test_size=0.3)
        self.assertIsInstance(strategy, HoldoutValidation)

    def test_create_kfold(self):
        """Test creation of K-Fold strategy"""
        strategy = ValidationStrategyFactory.create_strategy('kfold', n_splits=5)
        self.assertIsInstance(strategy, KFoldCrossValidation)

    def test_unknown_strategy(self):
        """Test that unknown strategy raises ValueError"""
        with self.assertRaises(ValueError):
            ValidationStrategyFactory.create_strategy('unknown_strategy')

    def test_list_strategies(self):
        """Test listing available strategies"""
        strategies = ValidationStrategyFactory.list_strategies()

        self.assertIsInstance(strategies, list)
        self.assertIn('holdout', strategies)
        self.assertIn('kfold', strategies)


if __name__ == '__main__':
    unittest.main()
