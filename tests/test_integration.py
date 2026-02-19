"""
Integration tests using actual test data files.

These tests verify the complete pipeline works correctly with
real data from all provided test files, using per-fold normalization
to avoid data leakage.
"""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.utils.data_loader import DataLoaderFactory
from src.utils.preprocessing import compute_normalization_params, apply_normalization
from src.classifier.knn_classifier import KNNClassifier
from src.evaluation.strategies import HoldoutValidation, KFoldCrossValidation
from src.metrics.metrics import calculate_all_metrics


class TestIntegrationWithTestData(unittest.TestCase):
    """Integration tests using provided test data files"""

    @classmethod
    def setUpClass(cls):
        """Set up paths to test data files"""
        cls.test_data_dir = os.path.join(os.path.dirname(__file__), '..', 'tests_data')
        if not os.path.exists(cls.test_data_dir):
            cls.test_data_dir = 'tests_data'

        cls.test_files = {
            'csv': 'version_1.csv',
            'xlsx': 'version_2.xlsx',
            'txt': 'version_3.txt',
            'json': 'version_4.json',
            'tsv': 'version_5.tsv'
        }

    def _get_filepath(self, format_name):
        """Get the filepath for a test file"""
        return os.path.join(self.test_data_dir, self.test_files[format_name])

    @staticmethod
    def _run_pipeline(filepath, k=3):
        """
        Run a complete ML pipeline on a data file

        Normalization is performed per-fold: min/max are computed on the
        training set only and applied to both training and test sets.

        Parameters:
        -----------
        filepath : str
            Path to data file
        k : int
            Number of neighbors for k-NN

        Returns:
        --------
        dict
            Metrics from the evaluation
        """
        # Load data as TumorDataset
        dataset = DataLoaderFactory.load_data(filepath)

        # Convert to arrays (raw, not normalized yet)
        features, labels = dataset.to_arrays()

        # Create the classifier and validation strategy
        classifier = KNNClassifier(k=k)
        strategy = HoldoutValidation(test_size=0.3, random_state=42)

        # Run evaluation with per-fold normalization
        for train_idx, test_idx in strategy.split(features, labels):
            features_train, features_test = features[train_idx], features[test_idx]
            labels_train, labels_test = labels[train_idx], labels[test_idx]

            # Normalize: fit on training data only, apply to both
            min_vals, max_vals = compute_normalization_params(features_train)
            features_train = apply_normalization(features_train, min_vals, max_vals)
            features_test = apply_normalization(features_test, min_vals, max_vals)

            classifier.fit(features_train, labels_train)
            labels_pred = classifier.predict(features_test)
            labels_proba = classifier.predict_proba(features_test)

            metrics = calculate_all_metrics(labels_test, labels_pred, labels_proba)

            return metrics
        return None

    def test_csv_file_pipeline(self):
        """Test complete pipeline with a CSV file"""
        filepath = self._get_filepath('csv')
        if not os.path.exists(filepath):
            self.skipTest(f"Test file not found: {filepath}")

        metrics = self._run_pipeline(filepath)

        self.assertIn('accuracy', metrics)
        self.assertIn('sensitivity', metrics)
        self.assertIn('specificity', metrics)
        self.assertIn('auc', metrics)

        self.assertGreaterEqual(metrics['accuracy'], 0.0)
        self.assertLessEqual(metrics['accuracy'], 1.0)
        self.assertGreaterEqual(metrics['auc'], 0.0)
        self.assertLessEqual(metrics['auc'], 1.0)

    def test_txt_file_pipeline(self):
        """Test complete pipeline with a TXT file"""
        filepath = self._get_filepath('txt')
        if not os.path.exists(filepath):
            self.skipTest(f"Test file not found: {filepath}")

        metrics = self._run_pipeline(filepath)
        self.assertGreater(metrics['accuracy'], 0.5)

    def test_tsv_file_pipeline(self):
        """Test complete pipeline with a TSV file"""
        filepath = self._get_filepath('tsv')
        if not os.path.exists(filepath):
            self.skipTest(f"Test file not found: {filepath}")

        metrics = self._run_pipeline(filepath)
        self.assertIn('accuracy', metrics)
        self.assertIn('confusion_matrix', metrics)

    def test_json_file_pipeline(self):
        """Test complete pipeline with a JSON file"""
        filepath = self._get_filepath('json')
        if not os.path.exists(filepath):
            self.skipTest(f"Test file not found: {filepath}")

        metrics = self._run_pipeline(filepath)
        self.assertGreater(metrics['accuracy'], 0.5)

    def test_xlsx_file_pipeline(self):
        """Test complete pipeline with a XLSX file"""
        filepath = self._get_filepath('xlsx')
        if not os.path.exists(filepath):
            self.skipTest(f"Test file not found: {filepath}")

        metrics = self._run_pipeline(filepath)
        self.assertIn('accuracy', metrics)
        self.assertIn('geometric_mean', metrics)


class TestKFoldIntegration(unittest.TestCase):
    """Test K-Fold cross-validation with real data"""

    @classmethod
    def setUpClass(cls):
        """Set up test data"""
        cls.test_data_dir = os.path.join(os.path.dirname(__file__), '..', 'tests_data')
        if not os.path.exists(cls.test_data_dir):
            cls.test_data_dir = 'tests_data'

    def test_kfold_evaluation(self):
        """Test K-Fold cross-validation with real data and per-fold normalization"""
        filepath = os.path.join(self.test_data_dir, 'version_1.csv')
        if not os.path.exists(filepath):
            self.skipTest(f"Test file not found: {filepath}")

        dataset = DataLoaderFactory.load_data(filepath)
        features, labels = dataset.to_arrays()

        classifier = KNNClassifier(k=3)
        strategy = KFoldCrossValidation(n_splits=5, random_state=42)

        fold_metrics = []
        for train_idx, test_idx in strategy.split(features, labels):
            features_train, features_test = features[train_idx], features[test_idx]
            labels_train, labels_test = labels[train_idx], labels[test_idx]

            # Normalize per-fold: fit on training, apply to both
            min_vals, max_vals = compute_normalization_params(features_train)
            features_train = apply_normalization(features_train, min_vals, max_vals)
            features_test = apply_normalization(features_test, min_vals, max_vals)

            classifier.fit(features_train, labels_train)
            labels_pred = classifier.predict(features_test)
            labels_proba = classifier.predict_proba(features_test)

            metrics = calculate_all_metrics(labels_test, labels_pred, labels_proba)
            fold_metrics.append(metrics)

        self.assertEqual(len(fold_metrics), 5)
        avg_accuracy = np.mean([m['accuracy'] for m in fold_metrics])
        self.assertGreater(avg_accuracy, 0.5)
        self.assertLessEqual(avg_accuracy, 1.0)


class TestDifferentKValues(unittest.TestCase):
    """Test classifier with different k values"""

    @classmethod
    def setUpClass(cls):
        """Set up test data"""
        cls.test_data_dir = os.path.join(os.path.dirname(__file__), '..', 'tests_data')
        if not os.path.exists(cls.test_data_dir):
            cls.test_data_dir = 'tests_data'

    def test_varying_k_values(self):
        """Test that the classifier works with different k values"""
        filepath = os.path.join(self.test_data_dir, 'version_1.csv')
        if not os.path.exists(filepath):
            self.skipTest(f"Test file not found: {filepath}")

        dataset = DataLoaderFactory.load_data(filepath)
        features, labels = dataset.to_arrays()

        k_values = [1, 3, 5, 7, 11]
        results = {}

        for k in k_values:
            classifier = KNNClassifier(k=k)
            strategy = HoldoutValidation(test_size=0.3, random_state=42)

            for train_idx, test_idx in strategy.split(features, labels):
                features_train, features_test = features[train_idx], features[test_idx]
                labels_train, labels_test = labels[train_idx], labels[test_idx]

                # Normalize per-fold
                min_vals, max_vals = compute_normalization_params(features_train)
                features_train = apply_normalization(features_train, min_vals, max_vals)
                features_test = apply_normalization(features_test, min_vals, max_vals)

                classifier.fit(features_train, labels_train)
                labels_pred = classifier.predict(features_test)

                metrics = calculate_all_metrics(labels_test, labels_pred)
                results[k] = metrics['accuracy']

        self.assertEqual(len(results), len(k_values))
        for k, acc in results.items():
            self.assertGreater(acc, 0.5, f"Accuracy for k={k} is too low")


if __name__ == '__main__':
    unittest.main()
