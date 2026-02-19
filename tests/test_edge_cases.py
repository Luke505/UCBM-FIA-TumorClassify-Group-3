"""
Comprehensive edge case and error handling tests
"""

import os
import shutil
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.classifier.knn_classifier import KNNClassifier
from src.metrics.metrics import (
    accuracy_score,
    sensitivity,
    specificity,
    geometric_mean,
    auc_score
)
from src.model import TumorDataset, TumorSample, TumorFeatures
from src.utils.data_loader import DataLoaderFactory
from src.utils.preprocessing import normalize_features
from src.evaluation.strategies import KFoldCrossValidation


class TestDivisionByZero(unittest.TestCase):
    """Test edge cases that could cause division by zero"""

    def test_accuracy_empty_predictions(self):
        """Test accuracy with empty arrays"""
        labels_true = np.array([])
        labels_pred = np.array([])

        result = accuracy_score(labels_true, labels_pred)
        self.assertEqual(result, 0.0)

    def test_sensitivity_no_positives(self):
        """Test sensitivity when there are no positive samples"""
        labels_true = np.array([2, 2, 2, 2])
        labels_pred = np.array([2, 2, 2, 2])

        result = sensitivity(labels_true, labels_pred)
        self.assertEqual(result, 0.0)

    def test_specificity_no_negatives(self):
        """Test specificity when there are no negative samples"""
        labels_true = np.array([4, 4, 4, 4])
        labels_pred = np.array([4, 4, 4, 4])

        result = specificity(labels_true, labels_pred)
        self.assertEqual(result, 0.0)

    def test_geometric_mean_zero_sensitivity(self):
        """Test geometric mean when sensitivity is zero"""
        labels_true = np.array([2, 2, 2, 2, 4])
        labels_pred = np.array([2, 2, 2, 2, 2])

        result = geometric_mean(labels_true, labels_pred)
        self.assertEqual(result, 0.0)

    def test_auc_all_same_class(self):
        """Test AUC when all samples are from one class"""
        labels_true = np.array([2, 2, 2, 2])
        labels_proba = np.array([0.1, 0.2, 0.3, 0.4])
        result = auc_score(labels_true, labels_proba)
        self.assertEqual(result, 0.5)

        labels_true = np.array([4, 4, 4, 4])
        labels_proba = np.array([0.6, 0.7, 0.8, 0.9])
        result = auc_score(labels_true, labels_proba)
        self.assertEqual(result, 0.5)


class TestDataLoadingEdgeCases(unittest.TestCase):
    """Test edge cases in data loading"""

    def setUp(self):
        """Create a temporary directory"""
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up"""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_load_nonexistent_file(self):
        """Test loading a file that doesn't exist"""
        with self.assertRaises(FileNotFoundError):
            DataLoaderFactory.load_data('/nonexistent/file.csv')

    def test_load_unsupported_format(self):
        """Test loading unsupported file format"""
        test_file = os.path.join(self.test_dir, 'test.pdf')
        with open(test_file, 'w') as f:
            f.write("dummy")

        with self.assertRaises(ValueError):
            DataLoaderFactory.load_data(test_file)

    def test_load_empty_csv(self):
        """Test loading empty CSV file (headers only)"""
        test_file = os.path.join(self.test_dir, 'empty.csv')
        with open(test_file, 'w') as f:
            f.write("sample_code_number,clump_thickness,uniformity_cell_size,"
                    "uniformity_cell_shape,marginal_adhesion,"
                    "single_epithelial_cell_size,bare_nuclei,"
                    "bland_chromatin,normal_nucleoli,mitoses,class\n")

        dataset = DataLoaderFactory.load_data(test_file)
        self.assertIsInstance(dataset, TumorDataset)
        self.assertEqual(len(dataset), 0)

    def test_load_csv_with_all_invalid_samples(self):
        """Test loading CSV where all samples have invalid data"""
        test_file = os.path.join(self.test_dir, 'all_invalid.csv')
        data = ("sample_code_number,clump_thickness,uniformity_cell_size,"
                "uniformity_cell_shape,marginal_adhesion,"
                "single_epithelial_cell_size,bare_nuclei,"
                "bland_chromatin,normal_nucleoli,mitoses,class\n")
        for i in range(5):
            # class=3 is invalid, should be filtered
            data += f"{i},5,5,5,5,5,5,5,5,5,3\n"

        with open(test_file, 'w') as f:
            f.write(data)

        dataset = DataLoaderFactory.load_data(test_file)
        self.assertIsInstance(dataset, TumorDataset)
        self.assertEqual(len(dataset), 0)

    def test_load_csv_filters_duplicate_ids(self):
        """Test that duplicate IDs are filtered out"""
        test_file = os.path.join(self.test_dir, 'dupes.csv')
        data = ("sample_code_number,clump_thickness,uniformity_cell_size,"
                "uniformity_cell_shape,marginal_adhesion,"
                "single_epithelial_cell_size,bare_nuclei,"
                "bland_chromatin,normal_nucleoli,mitoses,class\n")
        data += "1,5,5,5,5,5,5,5,5,5,2\n"
        data += "1,3,3,3,3,3,3,3,3,3,4\n"  # duplicate id=1
        data += "2,5,5,5,5,5,5,5,5,5,4\n"

        with open(test_file, 'w') as f:
            f.write(data)

        dataset = DataLoaderFactory.load_data(test_file)
        self.assertEqual(len(dataset), 2)  # only 2 unique IDs
        # First occurrence is kept
        sample = dataset.get_by_id(1)
        self.assertEqual(sample.tumor_class, 2)


class TestClassifierEdgeCases(unittest.TestCase):
    """Test edge cases in k-NN classifier"""

    def test_predict_with_k_larger_than_training_set(self):
        """Test prediction when k is larger than the training set"""
        np.random.seed(42)
        features_train = np.random.rand(3, 9)
        labels_train = np.array([2, 4, 2])
        features_test = np.random.rand(2, 9)

        clf = KNNClassifier(k=10)
        clf.fit(features_train, labels_train)

        predictions = clf.predict(features_test)
        self.assertEqual(len(predictions), 2)
        self.assertTrue(all(p in [2, 4] for p in predictions))

    def test_predict_single_sample(self):
        """Test prediction with a single test sample"""
        np.random.seed(42)
        features_train = np.random.rand(10, 9)
        labels_train = np.array([2] * 5 + [4] * 5)
        features_test = np.random.rand(1, 9)

        clf = KNNClassifier(k=3)
        clf.fit(features_train, labels_train)

        predictions = clf.predict(features_test)
        self.assertEqual(len(predictions), 1)
        self.assertIn(predictions[0], [2, 4])

    def test_predict_proba_edge_cases(self):
        """Test probability prediction with single class"""
        np.random.seed(42)
        features_train = np.random.rand(10, 9)
        labels_train = np.array([2] * 10)  # All benign
        features_test = np.random.rand(2, 9)

        clf = KNNClassifier(k=3)
        clf.fit(features_train, labels_train)

        proba = clf.predict_proba(features_test)
        self.assertEqual(proba.shape[0], 2)
        self.assertTrue(np.all((proba >= 0) & (proba <= 1)))

    def test_fit_with_tumor_dataset(self):
        """Test fitting classifier with TumorDataset directly"""
        samples = [
            TumorSample(id=i, features=TumorFeatures(5, 5, 5, 5, 5, 5, 5, 5, 5),
                tumor_class=2 if i < 5 else 4)
            for i in range(10)
        ]
        dataset = TumorDataset(samples)

        clf = KNNClassifier(k=3)
        clf.fit(dataset)
        self.assertIsNotNone(clf.features_train)
        self.assertIsNotNone(clf.labels_train)


class TestPreprocessingEdgeCases(unittest.TestCase):
    """Test edge cases in preprocessing"""

    def test_normalize_constant_feature(self):
        """Test normalization when a feature has a constant value"""
        features = np.array([
            [1, 5, 5, 5, 5, 5, 5, 5, 5],
            [2, 5, 5, 5, 5, 5, 5, 5, 5],
            [3, 5, 5, 5, 5, 5, 5, 5, 5]
        ])

        features_norm = normalize_features(features)
        self.assertEqual(features_norm.shape, features.shape)
        self.assertTrue(np.all(np.isfinite(features_norm)))

    def test_normalize_single_sample(self):
        """Test normalization with a single sample"""
        features = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9]])
        features_norm = normalize_features(features)

        self.assertEqual(features_norm.shape, features.shape)
        self.assertTrue(np.all(np.isfinite(features_norm)))


class TestTumorModelEdgeCases(unittest.TestCase):
    """Test edge cases in tumor data model"""

    def test_tumor_features_invalid_values(self):
        """Test that invalid feature values raise ValueError"""
        with self.assertRaises(ValueError):
            TumorFeatures(0, 5, 5, 5, 5, 5, 5, 5, 5)  # 0 out of range
        with self.assertRaises(ValueError):
            TumorFeatures(11, 5, 5, 5, 5, 5, 5, 5, 5)  # 11 out of range

    def test_tumor_sample_invalid_class(self):
        """Test that invalid tumor class raises ValueError"""
        features = TumorFeatures(5, 5, 5, 5, 5, 5, 5, 5, 5)
        with self.assertRaises(ValueError):
            TumorSample(id=1, features=features, tumor_class=3)

    def test_tumor_dataset_duplicate_ids(self):
        """Test that duplicate IDs are rejected"""
        features = TumorFeatures(5, 5, 5, 5, 5, 5, 5, 5, 5)
        s1 = TumorSample(id=1, features=features, tumor_class=2)
        s2 = TumorSample(id=1, features=features, tumor_class=4)

        dataset = TumorDataset()
        self.assertTrue(dataset.add_sample(s1))
        self.assertFalse(dataset.add_sample(s2))  # duplicate rejected
        self.assertEqual(len(dataset), 1)

    def test_tumor_dataset_empty(self):
        """Test empty dataset to_arrays"""
        dataset = TumorDataset()
        features, labels = dataset.to_arrays()
        self.assertEqual(features.shape, (0, 9))
        self.assertEqual(labels.shape, (0,))

    def test_tumor_sample_from_dict_invalid(self):
        """Test from_dict returns None for invalid data"""
        result = TumorSample.from_dict({'id': 1, 'class': 3})  # invalid class
        self.assertIsNone(result)

    def test_tumor_features_from_array(self):
        """Test creating features from numpy array"""
        arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
        features = TumorFeatures.from_array(arr)
        self.assertEqual(features.clump_thickness, 1)
        self.assertEqual(features.mitoses, 9)

    def test_tumor_features_from_array_wrong_length(self):
        """Test from_array with wrong length raises error"""
        with self.assertRaises(ValueError):
            TumorFeatures.from_array(np.array([1, 2, 3]))

    def test_tumor_dataset_filter_by_class(self):
        """Test filtering dataset by class"""
        samples = [
            TumorSample(id=1, features=TumorFeatures(5, 5, 5, 5, 5, 5, 5, 5, 5), tumor_class=2),
            TumorSample(id=2, features=TumorFeatures(5, 5, 5, 5, 5, 5, 5, 5, 5), tumor_class=4),
            TumorSample(id=3, features=TumorFeatures(5, 5, 5, 5, 5, 5, 5, 5, 5), tumor_class=2),
        ]
        dataset = TumorDataset(samples)

        benign = dataset.filter_by_class(2)
        self.assertEqual(len(benign), 2)

        malignant = dataset.filter_by_class(4)
        self.assertEqual(len(malignant), 1)

    def test_tumor_sample_properties(self):
        """Test is_malignant and is_benign properties"""
        features = TumorFeatures(5, 5, 5, 5, 5, 5, 5, 5, 5)
        benign = TumorSample(id=1, features=features, tumor_class=2)
        malignant = TumorSample(id=2, features=features, tumor_class=4)

        self.assertTrue(benign.is_benign)
        self.assertFalse(benign.is_malignant)
        self.assertTrue(malignant.is_malignant)
        self.assertFalse(malignant.is_benign)


# noinspection SpellCheckingInspection
class TestValidationStrategyEdgeCases(unittest.TestCase):
    """Test edge cases in validation strategies"""

    def test_kfold_with_k_larger_than_samples(self):
        """Test K-Fold when K is larger than the number of samples"""
        np.random.seed(42)
        features = np.random.rand(5, 9)
        labels = np.array([2, 4, 2, 4, 2])

        strategy = KFoldCrossValidation(n_splits=10, random_state=42)
        n_splits = strategy.get_n_splits(features, labels)
        self.assertIsInstance(n_splits, int)
        self.assertGreater(n_splits, 0)

    def test_kfold_with_minimum_samples(self):
        """Test K-Fold with a minimum number of samples"""
        np.random.seed(42)
        features = np.random.rand(10, 9)
        labels = np.array([2] * 5 + [4] * 5)

        strategy = KFoldCrossValidation(n_splits=3, random_state=42)
        folds = list(strategy.split(features, labels))
        self.assertEqual(len(folds), 3)

        for train_idx, test_idx in folds:
            self.assertGreater(len(train_idx), 0)
            self.assertGreater(len(test_idx), 0)


class TestIntegrationEdgeCases(unittest.TestCase):
    """Test edge cases in full pipeline integration"""

    def test_pipeline_with_imbalanced_data(self):
        """Test pipeline with a highly imbalanced dataset"""
        np.random.seed(42)
        features = np.random.rand(100, 9)
        labels = np.array([2] * 90 + [4] * 10)

        clf = KNNClassifier(k=3)
        strategy = KFoldCrossValidation(n_splits=3, random_state=42)

        for train_idx, test_idx in strategy.split(features, labels):
            features_train, features_test = features[train_idx], features[test_idx]
            labels_train, labels_test = labels[train_idx], labels[test_idx]

            clf.fit(features_train, labels_train)
            predictions = clf.predict(features_test)
            self.assertEqual(len(predictions), len(labels_test))

    def test_pipeline_with_very_small_dataset(self):
        """Test pipeline with a very small dataset"""
        np.random.seed(42)
        features = np.random.rand(6, 9)
        labels = np.array([2, 4, 2, 4, 2, 4])

        clf = KNNClassifier(k=1)
        strategy = KFoldCrossValidation(n_splits=2, random_state=42)

        fold_count = 0
        for train_idx, test_idx in strategy.split(features, labels):
            features_train, features_test = features[train_idx], features[test_idx]
            labels_train, labels_test = labels[train_idx], labels[test_idx]

            clf.fit(features_train, labels_train)
            predictions = clf.predict(features_test)
            self.assertEqual(len(predictions), len(labels_test))
            fold_count += 1

        self.assertEqual(fold_count, 2)


if __name__ == '__main__':
    unittest.main()
