"""
Unit tests for k-NN classifier

Tests the k-Nearest Neighbors classifier implementation
"""

import os
import sys
import unittest

import numpy as np

# Add src to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.classifier.knn_classifier import KNNClassifier


class TestKNNClassifier(unittest.TestCase):
    """Test cases for KNN Classifier"""

    def setUp(self):
        """Set up test fixtures"""
        # Create simple training data
        self.features_train = np.array([
            [1, 1],
            [1, 2],
            [2, 1],
            [5, 5],
            [5, 6],
            [6, 5]
        ])
        self.labels_train = np.array([2, 2, 2, 4, 4, 4])

        # Create test data
        self.features_test = np.array([
            [1.5, 1.5],  # Should be class 2 (closer to first cluster)
            [5.5, 5.5]  # Should be class 4 (closer to the second cluster)
        ])

    def test_initialization(self):
        """Test classifier initialization"""
        clf = KNNClassifier(k=3)
        self.assertEqual(clf.k, 3)
        self.assertIsNone(clf.features_train)
        self.assertIsNone(clf.labels_train)

    def test_initialization_invalid_k(self):
        """Test that invalid k values raise ValueError"""
        with self.assertRaises(ValueError):
            KNNClassifier(k=0)
        with self.assertRaises(ValueError):
            KNNClassifier(k=-1)

    def test_fit(self):
        """Test fitting the classifier"""
        clf = KNNClassifier(k=3)
        clf.fit(self.features_train, self.labels_train)

        np.testing.assert_array_equal(clf.features_train, self.features_train)
        np.testing.assert_array_equal(clf.labels_train, self.labels_train)

    def test_predict_without_fit(self):
        """Test that predict raises error if not fitted"""
        clf = KNNClassifier(k=3)
        with self.assertRaises(ValueError):
            clf.predict(self.features_test)

    def test_predict(self):
        """Test prediction"""
        clf = KNNClassifier(k=3)
        clf.fit(self.features_train, self.labels_train)
        predictions = clf.predict(self.features_test)

        self.assertEqual(len(predictions), 2)
        self.assertEqual(predictions[0], 2)  # The first test sample should be class 2
        self.assertEqual(predictions[1], 4)  # The second test sample should be class 4

    def test_predict_proba(self):
        """Test probability prediction"""
        clf = KNNClassifier(k=3)
        clf.fit(self.features_train, self.labels_train)
        probabilities = clf.predict_proba(self.features_test)

        self.assertEqual(probabilities.shape, (2, 2))  # 2 samples, 2 classes

        # Probabilities should sum to 1
        np.testing.assert_almost_equal(probabilities.sum(axis=1), [1.0, 1.0])

        # Probabilities should be between 0 and 1
        self.assertTrue(np.all(probabilities >= 0))
        self.assertTrue(np.all(probabilities <= 1))

    def test_calculate_distances(self):
        """Test Euclidean distance calculation"""
        clf = KNNClassifier(k=3)
        clf.fit(self.features_train, self.labels_train)

        test_sample = np.array([1, 1])
        distances = clf._calculate_distances(test_sample)

        # Distance to first training sample should be 0
        self.assertAlmostEqual(distances[0], 0)

        # Distances should be non-negative
        self.assertTrue(np.all(distances >= 0))

    def test_different_k_values(self):
        """Test classifier with different k values"""
        for k in [1, 3, 5]:
            clf = KNNClassifier(k=k)
            clf.fit(self.features_train, self.labels_train)
            predictions = clf.predict(self.features_test)

            self.assertEqual(len(predictions), len(self.features_test))
            # All predictions should be valid class labels
            self.assertTrue(all(pred in [2, 4] for pred in predictions))


class TestKNNClassifierEdgeCases(unittest.TestCase):
    """Test edge cases for KNN Classifier"""

    def test_single_neighbor(self):
        """Test with k=1"""
        features_train = np.array([[1, 1], [2, 2], [3, 3]])
        labels_train = np.array([2, 4, 2])
        features_test = np.array([[1.1, 1.1]])

        clf = KNNClassifier(k=1)
        clf.fit(features_train, labels_train)
        prediction = clf.predict(features_test)

        self.assertEqual(prediction[0], 2)  # Closest to [1, 1]

    def test_all_neighbors_same_class(self):
        """Test when all neighbors belong to the same class"""
        features_train = np.array([[1, 1], [1, 2], [2, 1]])
        labels_train = np.array([2, 2, 2])
        features_test = np.array([[1.5, 1.5]])

        clf = KNNClassifier(k=3)
        clf.fit(features_train, labels_train)
        prediction = clf.predict(features_test)

        self.assertEqual(prediction[0], 2)


if __name__ == '__main__':
    unittest.main()
