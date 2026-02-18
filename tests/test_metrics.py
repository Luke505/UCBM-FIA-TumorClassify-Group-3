"""
Unit tests for evaluation metrics

Tests all metric calculations including accuracy, sensitivity,
specificity, geometric mean, and AUC
"""

import os
import sys
import unittest

import numpy as np

# Add src to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.metrics.metrics import (
    confusion_matrix,
    accuracy_score,
    error_rate,
    sensitivity,
    specificity,
    geometric_mean,
    auc_score,
    calculate_all_metrics
)


class TestConfusionMatrix(unittest.TestCase):
    """Test cases for confusion matrix calculation"""

    def test_perfect_prediction(self):
        """Test confusion matrix with perfect predictions"""
        labels_true = np.array([2, 2, 4, 4])
        labels_pred = np.array([2, 2, 4, 4])

        TN, FP, FN, TP = confusion_matrix(labels_true, labels_pred)

        self.assertEqual(TN, 2)
        self.assertEqual(FP, 0)
        self.assertEqual(FN, 0)
        self.assertEqual(TP, 2)

    def test_all_wrong_prediction(self):
        """Test confusion matrix with all wrong predictions"""
        labels_true = np.array([2, 2, 4, 4])
        labels_pred = np.array([4, 4, 2, 2])

        TN, FP, FN, TP = confusion_matrix(labels_true, labels_pred)

        self.assertEqual(TN, 0)
        self.assertEqual(FP, 2)
        self.assertEqual(FN, 2)
        self.assertEqual(TP, 0)

    def test_mixed_prediction(self):
        """Test confusion matrix with mixed predictions"""
        labels_true = np.array([2, 2, 2, 4, 4, 4])
        labels_pred = np.array([2, 4, 2, 4, 4, 2])

        TN, FP, FN, TP = confusion_matrix(labels_true, labels_pred)

        self.assertEqual(TN, 2)  # Predicted 2, actual 2
        self.assertEqual(FP, 1)  # Predicted 4, actual 2
        self.assertEqual(FN, 1)  # Predicted 2, actual 4
        self.assertEqual(TP, 2)  # Predicted 4, actual 4


class TestAccuracyMetrics(unittest.TestCase):
    """Test cases for accuracy and error rate"""

    def test_accuracy_perfect(self):
        """Test accuracy with perfect predictions"""
        labels_true = np.array([2, 2, 4, 4])
        labels_pred = np.array([2, 2, 4, 4])

        acc = accuracy_score(labels_true, labels_pred)
        self.assertEqual(acc, 1.0)

    def test_accuracy_zero(self):
        """Test accuracy with all wrong predictions"""
        labels_true = np.array([2, 2, 4, 4])
        labels_pred = np.array([4, 4, 2, 2])

        acc = accuracy_score(labels_true, labels_pred)
        self.assertEqual(acc, 0.0)

    def test_accuracy_half(self):
        """Test accuracy with 50% correct predictions"""
        labels_true = np.array([2, 2, 4, 4])
        labels_pred = np.array([2, 4, 4, 2])

        acc = accuracy_score(labels_true, labels_pred)
        self.assertEqual(acc, 0.5)

    def test_error_rate(self):
        """Test that error rate = 1 - accuracy"""
        labels_true = np.array([2, 2, 4, 4])
        labels_pred = np.array([2, 4, 4, 2])

        acc = accuracy_score(labels_true, labels_pred)
        err = error_rate(labels_true, labels_pred)

        self.assertAlmostEqual(acc + err, 1.0)


class TestSensitivitySpecificity(unittest.TestCase):
    """Test cases for sensitivity and specificity"""

    def test_sensitivity_perfect(self):
        """Test sensitivity with perfect positive predictions"""
        labels_true = np.array([2, 2, 4, 4, 4])
        labels_pred = np.array([2, 2, 4, 4, 4])

        sens = sensitivity(labels_true, labels_pred)
        self.assertEqual(sens, 1.0)

    def test_sensitivity_zero(self):
        """Test sensitivity when no positives are detected"""
        labels_true = np.array([2, 2, 4, 4, 4])
        labels_pred = np.array([2, 2, 2, 2, 2])

        sens = sensitivity(labels_true, labels_pred)
        self.assertEqual(sens, 0.0)

    def test_specificity_perfect(self):
        """Test specificity with perfect negative predictions"""
        labels_true = np.array([2, 2, 2, 4, 4])
        labels_pred = np.array([2, 2, 2, 4, 4])

        spec = specificity(labels_true, labels_pred)
        self.assertEqual(spec, 1.0)

    def test_specificity_zero(self):
        """Test specificity when no negatives are detected"""
        labels_true = np.array([2, 2, 2, 4, 4])
        labels_pred = np.array([4, 4, 4, 4, 4])

        spec = specificity(labels_true, labels_pred)
        self.assertEqual(spec, 0.0)


class TestGeometricMean(unittest.TestCase):
    """Test cases for geometric mean"""

    def test_geometric_mean_perfect(self):
        """Test geometric mean with perfect predictions"""
        labels_true = np.array([2, 2, 4, 4])
        labels_pred = np.array([2, 2, 4, 4])

        gm = geometric_mean(labels_true, labels_pred)
        self.assertEqual(gm, 1.0)

    def test_geometric_mean_calculation(self):
        """Test geometric mean calculation"""
        labels_true = np.array([2, 2, 2, 2, 4, 4, 4, 4])
        labels_pred = np.array([2, 2, 4, 2, 4, 4, 2, 4])

        sens = sensitivity(labels_true, labels_pred)
        spec = specificity(labels_true, labels_pred)
        gm = geometric_mean(labels_true, labels_pred)

        self.assertAlmostEqual(gm, np.sqrt(sens * spec))


class TestAUCScore(unittest.TestCase):
    """Test cases for AUC score"""

    def test_auc_perfect(self):
        """Test AUC with perfect probability predictions"""
        labels_true = np.array([2, 2, 4, 4])
        labels_proba = np.array([0.1, 0.2, 0.9, 0.8])

        auc = auc_score(labels_true, labels_proba)
        self.assertEqual(auc, 1.0)

    def test_auc_random(self):
        """Test AUC with random predictions (should be ~0.5)"""
        np.random.seed(42)
        labels_true = np.array([2] * 50 + [4] * 50)
        labels_proba = np.random.rand(100)

        auc = auc_score(labels_true, labels_proba)
        self.assertGreaterEqual(auc, 0.0)
        self.assertLessEqual(auc, 1.0)

    def test_auc_with_2d_probabilities(self):
        """Test AUC with 2D probability array"""
        labels_true = np.array([2, 2, 4, 4])
        labels_proba = np.array([
            [0.9, 0.1],
            [0.8, 0.2],
            [0.1, 0.9],
            [0.2, 0.8]
        ])

        auc = auc_score(labels_true, labels_proba)
        self.assertEqual(auc, 1.0)


class TestCalculateAllMetrics(unittest.TestCase):
    """Test cases for calculate_all_metrics function"""

    def test_all_metrics_calculated(self):
        """Test that all metrics are calculated"""
        labels_true = np.array([2, 2, 4, 4])
        labels_pred = np.array([2, 2, 4, 4])
        labels_proba = np.array([0.1, 0.2, 0.9, 0.8])

        metrics = calculate_all_metrics(labels_true, labels_pred, labels_proba)

        # Check that all expected metrics are present
        self.assertIn('accuracy', metrics)
        self.assertIn('error_rate', metrics)
        self.assertIn('sensitivity', metrics)
        self.assertIn('specificity', metrics)
        self.assertIn('geometric_mean', metrics)
        self.assertIn('auc', metrics)
        self.assertIn('confusion_matrix', metrics)

    def test_all_metrics_without_probabilities(self):
        """Test metrics calculation without probabilities"""
        labels_true = np.array([2, 2, 4, 4])
        labels_pred = np.array([2, 2, 4, 4])

        metrics = calculate_all_metrics(labels_true, labels_pred)

        # AUC should not be present without probabilities
        self.assertNotIn('auc', metrics)

        # Other metrics should be present
        self.assertIn('accuracy', metrics)
        self.assertIn('confusion_matrix', metrics)


if __name__ == '__main__':
    unittest.main()
