"""
Unit tests for validation strategies

Tests all validation strategies including Holdout, K-Fold, LOOCV,
Stratified CV, Bootstrap, etc.
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
    LeaveOneOutCrossValidation,
    LeavePOutCrossValidation,
    StratifiedCrossValidation,
    StratifiedShuffleSplit,
    Bootstrap,
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


class TestLeaveOneOutCrossValidation(unittest.TestCase):
    """Test cases for Leave-One-Out Cross-Validation"""

    def test_number_of_splits(self):
        """Test that LOOCV produces n_samples splits"""
        features = np.random.rand(50, 5)
        labels = np.random.choice([2, 4], 50)

        strategy = LeaveOneOutCrossValidation()
        n_splits = strategy.get_n_splits(features, labels)

        self.assertEqual(n_splits, 50)

    def test_single_test_sample(self):
        """Test that each split has exactly one test sample"""
        features = np.random.rand(50, 5)
        labels = np.random.choice([2, 4], 50)

        strategy = LeaveOneOutCrossValidation()

        for train_idx, test_idx in strategy.split(features, labels):
            self.assertEqual(len(test_idx), 1)
            self.assertEqual(len(train_idx), 49)


class TestLeavePOutCrossValidation(unittest.TestCase):
    """Test cases for Leave-P-Out Cross-Validation"""

    def test_p_test_samples(self):
        """Test that each split has p test samples"""
        features = np.random.rand(50, 5)
        labels = np.random.choice([2, 4], 50)

        p = 3
        strategy = LeavePOutCrossValidation(p=p, max_splits=10, random_state=42)

        for train_idx, test_idx in strategy.split(features, labels):
            self.assertEqual(len(test_idx), p)
            self.assertEqual(len(train_idx), 50 - p)

    def test_max_splits_limit(self):
        """Test that max_splits limits the number of splits"""
        features = np.random.rand(50, 5)
        labels = np.random.choice([2, 4], 50)

        max_splits = 20
        strategy = LeavePOutCrossValidation(p=2, max_splits=max_splits, random_state=42)
        n_splits = strategy.get_n_splits(features, labels)

        self.assertLessEqual(n_splits, max_splits)


class TestStratifiedCrossValidation(unittest.TestCase):
    """Test cases for Stratified Cross-Validation"""

    def test_class_proportions(self):
        """Test that class proportions are preserved in each fold"""
        # Create imbalanced dataset
        features = np.random.rand(100, 5)
        labels = np.array([2] * 30 + [4] * 70)

        strategy = StratifiedCrossValidation(n_splits=5, random_state=42)

        for train_idx, test_idx in strategy.split(features, labels):
            labels_test = labels[test_idx]

            # Count classes in test set
            class_counts = dict(zip(*np.unique(labels_test, return_counts=True)))

            # Check that both classes are present
            self.assertIn(2, class_counts)
            self.assertIn(4, class_counts)

            # Check approximate proportions (30% class 2, 70% class 4)
            total_test = len(labels_test)
            prop_class_2 = class_counts[2] / total_test

            # Allow some tolerance due to rounding
            self.assertAlmostEqual(prop_class_2, 0.3, delta=0.15)


class TestStratifiedShuffleSplit(unittest.TestCase):
    """Test cases for Stratified Shuffle Split"""

    def test_number_of_splits(self):
        """Test correct number of splits"""
        features = np.random.rand(100, 5)
        labels = np.array([2] * 50 + [4] * 50)

        n_splits = 10
        strategy = StratifiedShuffleSplit(n_splits=n_splits, test_size=0.3, random_state=42)

        self.assertEqual(strategy.get_n_splits(features, labels), n_splits)

    def test_class_proportions_maintained(self):
        """Test that class proportions are maintained"""
        features = np.random.rand(100, 5)
        labels = np.array([2] * 30 + [4] * 70)

        strategy = StratifiedShuffleSplit(n_splits=5, test_size=0.3, random_state=42)

        for train_idx, test_idx in strategy.split(features, labels):
            labels_test = labels[test_idx]
            class_counts = dict(zip(*np.unique(labels_test, return_counts=True)))

            # Both classes should be present
            self.assertIn(2, class_counts)
            self.assertIn(4, class_counts)


class TestBootstrap(unittest.TestCase):
    """Test cases for Bootstrap validation"""

    def test_number_of_splits(self):
        """Test correct number of bootstrap iterations"""
        features = np.random.rand(100, 5)
        labels = np.random.choice([2, 4], 100)

        n_splits = 10
        strategy = Bootstrap(n_splits=n_splits, random_state=42)

        self.assertEqual(strategy.get_n_splits(features, labels), n_splits)

    def test_train_size(self):
        """Test that training set has same size as original (with replacement)"""
        features = np.random.rand(100, 5)
        labels = np.random.choice([2, 4], 100)

        strategy = Bootstrap(n_splits=5, random_state=42)

        for train_idx, test_idx in strategy.split(features, labels):
            # Training set should have same size as original dataset
            self.assertEqual(len(train_idx), 100)
            # Test set (out-of-bag) should be non-empty
            self.assertGreater(len(test_idx), 0)


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

    def test_create_loocv(self):
        """Test creation of LOOCV strategy"""
        strategy = ValidationStrategyFactory.create_strategy('loocv')
        self.assertIsInstance(strategy, LeaveOneOutCrossValidation)

    def test_create_stratified(self):
        """Test creation of Stratified CV strategy"""
        strategy = ValidationStrategyFactory.create_strategy('stratified', n_splits=5)
        self.assertIsInstance(strategy, StratifiedCrossValidation)

    def test_create_bootstrap(self):
        """Test creation of Bootstrap strategy"""
        strategy = ValidationStrategyFactory.create_strategy('bootstrap', n_splits=10)
        self.assertIsInstance(strategy, Bootstrap)

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
        self.assertIn('bootstrap', strategies)


if __name__ == '__main__':
    unittest.main()
