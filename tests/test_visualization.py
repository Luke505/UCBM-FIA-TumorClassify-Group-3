"""
Unit tests for visualization functions
"""

import os
import shutil
import tempfile
import unittest

import numpy as np

from src.visualization.plots import (
    create_output_directory,
    plot_confusion_matrix,
    plot_roc_curve,
    plot_metrics_comparison,
    plot_accuracy_distribution,
    plot_accuracy_vs_k,
    plot_error_rate_vs_k,
    save_results_to_file,
    save_fold_results
)


class TestCreateOutputDirectory(unittest.TestCase):
    """Test output directory creation"""

    def setUp(self):
        """Create a temporary directory for tests"""
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Remove the temporary directory after tests"""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_create_output_directory_without_subfolder(self):
        """Test creating an output directory without a subfolder"""
        output_dir = create_output_directory(
            base_dir=self.test_dir,
            data_filename='test_data.csv',
            use_subfolder=False
        )

        self.assertEqual(output_dir, self.test_dir)
        self.assertTrue(os.path.exists(output_dir))

    def test_create_output_directory_with_subfolder(self):
        """Test creating an output directory with a subfolder"""
        output_dir = create_output_directory(
            base_dir=self.test_dir,
            data_filename='test_data.csv',
            use_subfolder=True
        )

        self.assertTrue(output_dir.startswith(self.test_dir))
        self.assertIn('test_data', output_dir)
        self.assertTrue(os.path.exists(output_dir))

    def test_create_output_directory_no_filename(self):
        """Test creating an output directory with no filename"""
        output_dir = create_output_directory(
            base_dir=self.test_dir,
            data_filename=None,
            use_subfolder=True
        )

        self.assertTrue(output_dir.startswith(self.test_dir))
        self.assertIn('experiment', output_dir)
        self.assertTrue(os.path.exists(output_dir))


class TestPlotConfusionMatrix(unittest.TestCase):
    """Test confusion matrix plotting"""

    def setUp(self):
        """Create a temporary directory for test outputs"""
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Remove a temporary directory after tests"""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_plot_confusion_matrix_basic(self):
        """Test basic confusion matrix plotting"""
        cm_dict = {
            'TN': 50,
            'FP': 5,
            'FN': 3,
            'TP': 42
        }

        output_path = os.path.join(self.test_dir, 'confusion_matrix.png')
        plot_confusion_matrix(cm_dict, output_path)

        self.assertTrue(os.path.exists(output_path))
        self.assertGreater(os.path.getsize(output_path), 0)

    def test_plot_confusion_matrix_perfect(self):
        """Test confusion matrix with perfect predictions"""
        cm_dict = {
            'TN': 60,
            'FP': 0,
            'FN': 0,
            'TP': 40
        }

        output_path = os.path.join(self.test_dir, 'confusion_matrix_perfect.png')
        plot_confusion_matrix(cm_dict, output_path)

        self.assertTrue(os.path.exists(output_path))


class TestPlotROCCurve(unittest.TestCase):
    """Test ROC curve plotting"""

    def setUp(self):
        """Create a temporary directory for test outputs"""
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Remove a temporary directory after tests"""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_plot_roc_curve_basic(self):
        """Test basic ROC curve plotting"""
        np.random.seed(42)
        labels_true = np.array([2, 4, 2, 4, 4, 2, 4, 2, 4, 4] * 10)
        labels_proba = np.random.rand(100)
        auc = 0.85

        output_path = os.path.join(self.test_dir, 'roc_curve.png')
        plot_roc_curve(labels_true, labels_proba, auc, output_path)

        self.assertTrue(os.path.exists(output_path))
        self.assertGreater(os.path.getsize(output_path), 0)

    def test_plot_roc_curve_2d_proba(self):
        """Test ROC curve with 2D probability array"""
        np.random.seed(42)
        labels_true = np.array([2, 4, 2, 4, 4, 2, 4, 2, 4, 4] * 10)
        labels_proba = np.random.rand(100, 2)
        auc = 0.80

        output_path = os.path.join(self.test_dir, 'roc_curve_2d.png')
        plot_roc_curve(labels_true, labels_proba, auc, output_path)

        self.assertTrue(os.path.exists(output_path))

    def test_plot_roc_curve_perfect(self):
        """Test ROC curve with perfect predictions"""
        labels_true = np.array([2, 2, 2, 2, 4, 4, 4, 4])
        labels_proba = np.array([0.1, 0.2, 0.15, 0.25, 0.9, 0.95, 0.88, 0.92])
        auc = 1.0

        output_path = os.path.join(self.test_dir, 'roc_curve_perfect.png')
        plot_roc_curve(labels_true, labels_proba, auc, output_path)

        self.assertTrue(os.path.exists(output_path))


class TestPlotAccuracyDistribution(unittest.TestCase):
    """Test accuracy distribution plotting"""

    def setUp(self):
        """Create a temporary directory for test outputs"""
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Remove a temporary directory after tests"""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_plot_accuracy_distribution_boxplot(self):
        """Test accuracy distribution with boxplot"""
        accuracies = [0.95, 0.96, 0.94, 0.97, 0.95, 0.96, 0.93, 0.95, 0.96, 0.94]

        output_path = os.path.join(self.test_dir, 'accuracy_dist_boxplot.png')
        plot_accuracy_distribution(accuracies, output_path, plot_type='boxplot')

        self.assertTrue(os.path.exists(output_path))
        self.assertGreater(os.path.getsize(output_path), 0)

    def test_plot_accuracy_distribution_histogram(self):
        """Test accuracy distribution with histogram"""
        accuracies = [0.95, 0.96, 0.94, 0.97, 0.95, 0.96, 0.93, 0.95, 0.96, 0.94]

        output_path = os.path.join(self.test_dir, 'accuracy_dist_hist.png')
        plot_accuracy_distribution(accuracies, output_path, plot_type='histogram')

        self.assertTrue(os.path.exists(output_path))
        self.assertGreater(os.path.getsize(output_path), 0)

    def test_plot_accuracy_distribution_few_values(self):
        """Test accuracy distribution with few values"""
        accuracies = [0.95, 0.96, 0.94]

        output_path = os.path.join(self.test_dir, 'accuracy_dist_few.png')
        plot_accuracy_distribution(accuracies, output_path, plot_type='boxplot')

        self.assertTrue(os.path.exists(output_path))


class TestPlotAccuracyVsK(unittest.TestCase):
    """Test accuracy vs. k plotting"""

    def setUp(self):
        """Create a temporary directory for test outputs"""
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Remove a temporary directory after tests"""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_plot_accuracy_vs_k_basic(self):
        """Test basic accuracy vs. k plotting"""
        k_values = [1, 3, 5, 7, 9, 11]
        accuracies = [0.92, 0.95, 0.97, 0.96, 0.95, 0.94]

        output_path = os.path.join(self.test_dir, 'accuracy_vs_k.png')
        plot_accuracy_vs_k(k_values, accuracies, output_path)

        self.assertTrue(os.path.exists(output_path))
        self.assertGreater(os.path.getsize(output_path), 0)

    def test_plot_accuracy_vs_k_increasing(self):
        """Test accuracy vs. k with increasing trend"""
        k_values = [1, 3, 5, 7, 9]
        accuracies = [0.90, 0.92, 0.94, 0.96, 0.98]

        output_path = os.path.join(self.test_dir, 'accuracy_vs_k_increasing.png')
        plot_accuracy_vs_k(k_values, accuracies, output_path)

        self.assertTrue(os.path.exists(output_path))

    def test_plot_accuracy_vs_k_decreasing(self):
        """Test accuracy vs. k with a decreasing trend"""
        k_values = [1, 3, 5, 7, 9]
        accuracies = [0.98, 0.96, 0.94, 0.92, 0.90]

        output_path = os.path.join(self.test_dir, 'accuracy_vs_k_decreasing.png')
        plot_accuracy_vs_k(k_values, accuracies, output_path)

        self.assertTrue(os.path.exists(output_path))


class TestPlotErrorRateVsK(unittest.TestCase):
    """Test error rate vs. k plotting"""

    def setUp(self):
        """Create a temporary directory for test outputs"""
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Remove a temporary directory after tests"""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_plot_error_rate_vs_k_basic(self):
        """Test basic error rate vs. k plotting"""
        k_values = [1, 3, 5, 7, 9, 11]
        error_rates = [0.08, 0.05, 0.03, 0.04, 0.05, 0.06]

        output_path = os.path.join(self.test_dir, 'error_rate_vs_k.png')
        plot_error_rate_vs_k(k_values, error_rates, output_path)

        self.assertTrue(os.path.exists(output_path))
        self.assertGreater(os.path.getsize(output_path), 0)

    def test_plot_error_rate_vs_k_from_accuracies(self):
        """Test error rate vs. k computed from accuracies"""
        k_values = [1, 3, 5, 7, 9]
        accuracies = [0.92, 0.95, 0.97, 0.96, 0.95]
        error_rates = [1 - acc for acc in accuracies]

        output_path = os.path.join(self.test_dir, 'error_rate_vs_k_computed.png')
        plot_error_rate_vs_k(k_values, error_rates, output_path)

        self.assertTrue(os.path.exists(output_path))


class TestSaveResultsToFile(unittest.TestCase):
    """Test saving results to files"""

    def setUp(self):
        """Create a temporary directory for test outputs"""
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Remove a temporary directory after tests"""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_save_results_to_csv(self):
        """Test saving results to CSV file"""
        # noinspection SpellCheckingInspection
        results = {
            'strategy': 'kfold',
            'k': 5,
            'accuracy': 0.95,
            'confusion_matrix': {'TN': 50, 'FP': 5, 'FN': 3, 'TP': 42}
        }

        output_path = os.path.join(self.test_dir, 'results.csv')
        save_results_to_file(results, output_path)

        self.assertTrue(os.path.exists(output_path))
        self.assertGreater(os.path.getsize(output_path), 0)

    def test_save_results_to_xlsx(self):
        """Test saving results to an Excel file"""
        results = {
            'strategy': 'holdout',
            'k': 3,
            'accuracy': 0.96
        }

        output_path = os.path.join(self.test_dir, 'results.xlsx')
        save_results_to_file(results, output_path)

        self.assertTrue(os.path.exists(output_path))
        self.assertGreater(os.path.getsize(output_path), 0)


class TestPlotMetricsComparison(unittest.TestCase):
    """Test metrics comparison bar chart plotting"""

    def setUp(self):
        """Create a temporary directory for test outputs"""
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Remove a temporary directory after tests"""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_plot_metrics_comparison_basic(self):
        """Test basic metrics comparison plotting"""
        metrics = {
            'avg_accuracy': 0.95,
            'avg_error_rate': 0.05,
            'avg_sensitivity': 0.93,
            'avg_specificity': 0.96,
            'avg_geometric_mean': 0.945
        }

        output_path = os.path.join(self.test_dir, 'metrics_comparison.png')
        plot_metrics_comparison(metrics, output_path)

        self.assertTrue(os.path.exists(output_path))
        self.assertGreater(os.path.getsize(output_path), 0)

    def test_plot_metrics_comparison_with_prefix(self):
        """Test metrics comparison with avg_ prefix"""
        metrics = {
            'accuracy': 0.97,
            'error_rate': 0.03,
            'sensitivity': 0.95,
            'specificity': 0.98,
            'geometric_mean': 0.965
        }

        output_path = os.path.join(self.test_dir, 'metrics_comparison_no_prefix.png')
        plot_metrics_comparison(metrics, output_path)

        self.assertTrue(os.path.exists(output_path))

    def test_plot_metrics_comparison_missing_metrics(self):
        """Test metrics comparison with some missing metrics"""
        metrics = {
            'avg_accuracy': 0.95,
            'avg_error_rate': 0.05
            # Missing sensitivity, specificity, geometric_mean
        }

        output_path = os.path.join(self.test_dir, 'metrics_comparison_partial.png')
        plot_metrics_comparison(metrics, output_path)

        self.assertTrue(os.path.exists(output_path))

    def test_plot_metrics_comparison_perfect_scores(self):
        """Test metrics comparison with perfect scores"""
        metrics = {
            'avg_accuracy': 1.0,
            'avg_error_rate': 0.0,
            'avg_sensitivity': 1.0,
            'avg_specificity': 1.0,
            'avg_geometric_mean': 1.0
        }

        output_path = os.path.join(self.test_dir, 'metrics_comparison_perfect.png')
        plot_metrics_comparison(metrics, output_path)

        self.assertTrue(os.path.exists(output_path))


class TestSaveFoldResults(unittest.TestCase):
    """Test saving per-fold results"""

    def setUp(self):
        """Create a temporary directory for test outputs"""
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Remove a temporary directory after tests"""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_save_fold_results(self):
        """Test saving per-fold results to CSV"""
        metrics_per_fold = [
            {
                'accuracy': 0.95,
                'error_rate': 0.05,
                'confusion_matrix': {'TN': 50, 'FP': 5, 'FN': 3, 'TP': 42}
            },
            {
                'accuracy': 0.96,
                'error_rate': 0.04,
                'confusion_matrix': {'TN': 52, 'FP': 3, 'FN': 4, 'TP': 41}
            }
        ]

        output_path = os.path.join(self.test_dir, 'fold_results.csv')
        save_fold_results(metrics_per_fold, output_path)

        self.assertTrue(os.path.exists(output_path))
        self.assertGreater(os.path.getsize(output_path), 0)

    def test_save_fold_results_xlsx(self):
        """Test saving per-fold results to Excel"""
        metrics_per_fold = [
            {
                'accuracy': 0.95,
                'sensitivity': 0.93,
                'confusion_matrix': {'TN': 50, 'FP': 5, 'FN': 3, 'TP': 42}
            }
        ]

        output_path = os.path.join(self.test_dir, 'fold_results.xlsx')
        save_fold_results(metrics_per_fold, output_path)

        self.assertTrue(os.path.exists(output_path))


class TestVisualizationEdgeCases(unittest.TestCase):
    """Test edge cases and error handling in visualization"""

    def setUp(self):
        """Create a temporary directory for test outputs"""
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Remove a temporary directory after tests"""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_plot_confusion_matrix_zeros(self):
        """Test confusion matrix with all zeros"""
        cm_dict = {'TN': 0, 'FP': 0, 'FN': 0, 'TP': 0}

        output_path = os.path.join(self.test_dir, 'cm_zeros.png')
        plot_confusion_matrix(cm_dict, output_path)

        self.assertTrue(os.path.exists(output_path))

    def test_plot_accuracy_distribution_single_value(self):
        """Test accuracy distribution with a single value"""
        accuracies = [0.95]

        output_path = os.path.join(self.test_dir, 'acc_dist_single.png')
        plot_accuracy_distribution(accuracies, output_path)

        self.assertTrue(os.path.exists(output_path))

    def test_plot_accuracy_vs_k_single_k(self):
        """Test accuracy vs. k with a single k value"""
        k_values = [5]
        accuracies = [0.95]

        output_path = os.path.join(self.test_dir, 'acc_vs_k_single.png')
        plot_accuracy_vs_k(k_values, accuracies, output_path)

        self.assertTrue(os.path.exists(output_path))

    def test_save_results_empty_dict(self):
        """Test saving empty results dictionary"""
        results = {}

        output_path = os.path.join(self.test_dir, 'empty_results.csv')
        save_results_to_file(results, output_path)

        self.assertTrue(os.path.exists(output_path))

    def test_save_results_with_arrays(self):
        """Test saving results with numpy arrays"""
        # noinspection SpellCheckingInspection
        results = {
            'strategy': 'kfold',
            'predictions': np.array([2, 4, 2, 4]),
            'accuracies': [0.95, 0.96, 0.94]
        }

        output_path = os.path.join(self.test_dir, 'results_arrays.csv')
        save_results_to_file(results, output_path)

        self.assertTrue(os.path.exists(output_path))


if __name__ == '__main__':
    unittest.main()
